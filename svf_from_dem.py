"""Sky View Factor (SVF) from an existing DEM raster + building heights.

Workflow implemented here
-------------------------
The DEM (bare-earth terrain) is supplied as a ready-made raster and read
directly -- it is **not** computed here.

1. **DEM + buildings -> DSM** : rasterize building footprints by their height
   and add them onto the DEM to obtain a Digital Surface Model.
2. **DSM -> SVF**        : for every cell, cast ``n_dirs`` azimuth rays, find
   the maximum horizon (elevation) angle within ``max_radius`` in each
   direction, and combine them with the isotropic-sky approximation::

       SVF = 1 - (1 / N) * sum_{i=1..N} sin^2( horizon_angle_i )

   (Oke 1987; Watson & Johnson 1987; Zaksek et al. 2011).

Three interchangeable back-ends are provided for each stage:

* ``*_py``    - pure Python (``geopandas`` + ``rasterio`` + ``numpy``).
                No GIS install needed; this is the default.
* ``*_arcpy`` - ArcGIS Pro / ArcPy Spatial Analyst.
* ``*_qgis``  - QGIS Processing (GDAL / SAGA providers).

An optional :func:`contours_to_dem_py` helper is kept for the case where you
only have contour lines and need to build a DEM first, but the main pipeline
assumes the DEM already exists.

Example (pure Python)
---------------------
>>> from svf_from_dem import run_pipeline_py
>>> run_pipeline_py(
...     dem="dem.tif",               # existing DEM raster (read directly)
...     buildings="buildings.gpkg",  height_field="height",
...     out_dsm="dsm.tif", out_svf="svf.tif",
...     n_dirs=16, max_radius=200.0,
... )
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_STOREY_HEIGHT = 3.0
DEFAULT_BUILDING_HEIGHT = 9.0
NODATA = -9999.0


# ===========================================================================
# Optional helper  -  Contours  ->  DEM
# (Only needed if you do NOT already have a DEM raster.  The main pipeline
#  reads an existing DEM and does not call this.)
# ===========================================================================


def contours_to_dem_py(
    contours: str | Path,
    elev_field: str,
    cell_size: float,
    out_dem: str | Path,
    bounds: tuple[float, float, float, float] | None = None,
    method: str = "linear",
) -> str:
    """Interpolate elevation contour lines onto a DEM raster (pure Python).

    Vertices are extracted from every contour line, tagged with the line's
    elevation, and interpolated to a regular grid with
    ``scipy.interpolate.griddata``.  ``method="linear"`` (Delaunay/TIN) is a
    good default; ``"cubic"`` is smoother, ``"nearest"`` fills gaps.

    Parameters
    ----------
    contours:
        Path to a line vector (Shapefile / GeoPackage) with an elevation
        attribute.  Must be in a projected (metric) CRS.
    elev_field:
        Name of the elevation attribute (metres).
    cell_size:
        Output pixel size in metres.
    out_dem:
        Output GeoTIFF path.
    bounds:
        Optional ``(minx, miny, maxx, maxy)`` extent; defaults to the
        contour extent.
    method:
        ``griddata`` interpolation method (``"linear"``, ``"cubic"``,
        ``"nearest"``).

    Returns
    -------
    Path to the written DEM.
    """
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_origin
    from scipy.interpolate import griddata
    from shapely.geometry import LineString, MultiLineString

    gdf = gpd.read_file(contours)
    if gdf.crs is None:
        raise ValueError("contours must have a defined (projected) CRS.")
    if elev_field not in gdf.columns:
        raise KeyError(f"'{elev_field}' not in {list(gdf.columns)}")

    # Explode every contour into its (x, y, z) vertices.
    xs, ys, zs = [], [], []
    for geom, z in zip(gdf.geometry, gdf[elev_field]):
        if geom is None or geom.is_empty or z is None:
            continue
        parts = geom.geoms if isinstance(geom, MultiLineString) else [geom]
        for part in parts:
            if not isinstance(part, LineString):
                continue
            for x, y in part.coords:
                xs.append(x); ys.append(y); zs.append(float(z))

    xs = np.asarray(xs); ys = np.asarray(ys); zs = np.asarray(zs)
    if len(xs) < 3:
        raise ValueError("Not enough contour vertices to interpolate.")

    minx, miny, maxx, maxy = bounds if bounds else (
        xs.min(), ys.min(), xs.max(), ys.max()
    )
    ncols = int(math.ceil((maxx - minx) / cell_size))
    nrows = int(math.ceil((maxy - miny) / cell_size))

    # Cell-centre coordinates; rows go top (maxy) -> bottom (miny).
    gx = minx + (np.arange(ncols) + 0.5) * cell_size
    gy = maxy - (np.arange(nrows) + 0.5) * cell_size
    mesh_x, mesh_y = np.meshgrid(gx, gy)

    logger.info("Interpolating DEM (%d x %d, %s) from %d contour vertices...",
                nrows, ncols, method, len(xs))
    dem = griddata((xs, ys), zs, (mesh_x, mesh_y), method=method)
    # Fill any NaN gaps left by linear/cubic with nearest-neighbour.
    if np.isnan(dem).any():
        fill = griddata((xs, ys), zs, (mesh_x, mesh_y), method="nearest")
        dem = np.where(np.isnan(dem), fill, dem)

    transform = from_origin(minx, maxy, cell_size, cell_size)
    _write_raster(out_dem, dem.astype("float32"), transform, gdf.crs)
    logger.info("DEM saved -> %s", out_dem)
    return str(out_dem)


# ===========================================================================
# Stage 1  -  DEM + building heights  ->  DSM
# ===========================================================================


def add_buildings_to_dem_py(
    dem: str | Path,
    buildings: str | Path,
    height_field: str,
    out_dsm: str | Path,
    levels_field: str = "building:levels",
    storey_height: float = DEFAULT_STOREY_HEIGHT,
    default_height: float = DEFAULT_BUILDING_HEIGHT,
) -> str:
    """Rasterize building heights and add them onto the DEM -> DSM (pure Python).

    The buildings are rasterized to the DEM grid (burning each footprint's
    height in metres) and summed with the terrain elevation, so the DSM holds
    ``terrain + building`` where buildings stand and bare terrain elsewhere.

    Parameters
    ----------
    dem:
        Input DEM GeoTIFF (from :func:`contours_to_dem_py`).
    buildings:
        Building footprints (polygon vector) with a height attribute.
    height_field:
        Height attribute name (metres).
    out_dsm:
        Output DSM GeoTIFF path.
    levels_field, storey_height, default_height:
        Used to fill missing heights (levels * storey_height, else default).

    Returns
    -------
    Path to the written DSM.
    """
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    with rasterio.open(dem) as src:
        dem_arr = src.read(1).astype("float32")
        transform = src.transform
        crs = src.crs
        out_shape = dem_arr.shape
        profile = src.profile

    gdf = gpd.read_file(buildings)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    heights = _resolve_heights(
        gdf, height_field, levels_field, storey_height, default_height
    )

    # (geometry, burn_value) pairs; tallest building wins on overlap.
    shapes = sorted(zip(gdf.geometry, heights), key=lambda t: t[1])
    bld_raster = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0.0,
        merge_alg=rasterio.enums.MergeAlg.replace,
        dtype="float32",
    )

    dsm = dem_arr + bld_raster
    profile.update(dtype="float32", count=1, nodata=NODATA)
    with rasterio.open(out_dsm, "w", **profile) as dst:
        dst.write(dsm.astype("float32"), 1)
    logger.info("DSM saved -> %s (max building add = %.1f m)",
                out_dsm, float(bld_raster.max()))
    return str(out_dsm)


# ===========================================================================
# Stage 2  -  DSM  ->  SVF
# ===========================================================================


def svf_from_dsm_py(
    dsm: str | Path,
    out_svf: str | Path,
    n_dirs: int = 16,
    max_radius: float = 200.0,
) -> str:
    """Compute a Sky View Factor raster from a DSM (pure Python / numpy).

    For each azimuth direction the algorithm marches outward in pixel steps,
    tracking the maximum horizon elevation angle seen so far (a vectorised
    horizon scan over the whole array).  The per-direction horizon angles are
    combined as ``SVF = 1 - mean(sin^2(horizon))``.

    Parameters
    ----------
    dsm:
        Input DSM GeoTIFF (terrain + buildings).
    out_svf:
        Output SVF GeoTIFF (values 0..1).
    n_dirs:
        Number of azimuth directions (16 = every 22.5 degrees).
    max_radius:
        Search radius in metres.

    Returns
    -------
    Path to the written SVF raster.
    """
    import rasterio

    with rasterio.open(dsm) as src:
        z = src.read(1).astype("float64")
        transform = src.transform
        crs = src.crs
        profile = src.profile
        nodata = src.nodata

    px = abs(transform.a)          # pixel size in x (metres)
    py = abs(transform.e)          # pixel size in y (metres)
    cell = (px + py) / 2.0
    max_steps = max(int(max_radius / cell), 1)
    nrows, ncols = z.shape

    if nodata is not None:
        z = np.where(z == nodata, np.nan, z)

    logger.info(
        "SVF from DSM (%d x %d, %d dirs, %.0f m radius = %d steps)...",
        nrows, ncols, n_dirs, max_radius, max_steps,
    )

    sin2_sum = np.zeros_like(z)
    valid = ~np.isnan(z)
    z0 = np.where(valid, z, 0.0)

    for d in range(n_dirs):
        az = 2.0 * math.pi * d / n_dirs
        dx = math.cos(az)          # column direction
        dy = -math.sin(az)         # row direction (row index grows downward)

        max_tan = np.full_like(z, -np.inf)
        for step in range(1, max_steps + 1):
            r_off = int(round(dy * step))
            c_off = int(round(dx * step))
            if r_off == 0 and c_off == 0:
                continue
            shifted = _shift2d(z0, r_off, c_off, fill=np.nan)
            dist = math.hypot(c_off * px, r_off * py)
            if dist == 0:
                continue
            tan_ang = (shifted - z0) / dist
            tan_ang = np.where(np.isnan(shifted), -np.inf, tan_ang)
            np.maximum(max_tan, tan_ang, out=max_tan)

        horizon = np.arctan(np.where(np.isfinite(max_tan), max_tan, 0.0))
        horizon = np.clip(horizon, 0.0, math.pi / 2)   # sky only, ignore dips
        sin2_sum += np.sin(horizon) ** 2

    svf = 1.0 - sin2_sum / n_dirs
    svf = np.where(valid, svf, NODATA)

    profile.update(dtype="float32", count=1, nodata=NODATA)
    with rasterio.open(out_svf, "w", **profile) as dst:
        dst.write(svf.astype("float32"), 1)
    good = svf[svf != NODATA]
    logger.info("SVF saved -> %s (range %.3f - %.3f, mean %.3f)",
                out_svf, good.min(), good.max(), good.mean())
    return str(out_svf)


def run_pipeline_py(
    dem: str | Path,
    buildings: str | Path,
    height_field: str,
    out_dsm: str | Path,
    out_svf: str | Path,
    n_dirs: int = 16,
    max_radius: float = 200.0,
) -> str:
    """Run the DEM + buildings -> DSM -> SVF pipeline (pure Python).

    *dem* is an existing DEM raster and is read directly; it is not computed
    here.  If you only have contour lines, build a DEM first with
    :func:`contours_to_dem_py` and pass its output as *dem*.
    """
    add_buildings_to_dem_py(dem=dem, buildings=buildings,
                            height_field=height_field, out_dsm=out_dsm)
    return svf_from_dsm_py(out_dsm, out_svf, n_dirs=n_dirs,
                           max_radius=max_radius)


# ---------------------------------------------------------------------------
# Shared helpers (pure Python back-end)
# ---------------------------------------------------------------------------


def _parse_number(value) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return float(str(value).strip().split()[0].replace(",", "."))
        except (ValueError, IndexError):
            return None


def _resolve_heights(gdf, height_field, levels_field,
                     storey_height, default_height) -> np.ndarray:
    """Numeric height per feature: height, else levels*storey, else default."""
    h = np.full(len(gdf), np.nan)
    if height_field in gdf.columns:
        parsed = gdf[height_field].map(_parse_number).to_numpy(dtype="float64")
        h = np.where(~np.isnan(parsed), parsed, h)
    if levels_field in gdf.columns:
        need = np.isnan(h)
        lv = gdf[levels_field].map(_parse_number).to_numpy(dtype="float64")
        h = np.where(need & ~np.isnan(lv), lv * storey_height, h)
    n_missing = int(np.isnan(h).sum())
    if n_missing:
        logger.info("  %d/%d buildings default to %.1f m",
                    n_missing, len(gdf), default_height)
    return np.where(np.isnan(h), default_height, h)


def _shift2d(arr: np.ndarray, r_off: int, c_off: int, fill=np.nan) -> np.ndarray:
    """Shift a 2-D array by (r_off, c_off), padding exposed edges with *fill*."""
    out = np.full_like(arr, fill)
    r_src = slice(max(0, -r_off), arr.shape[0] - max(0, r_off))
    c_src = slice(max(0, -c_off), arr.shape[1] - max(0, c_off))
    r_dst = slice(max(0, r_off), arr.shape[0] - max(0, -r_off))
    c_dst = slice(max(0, c_off), arr.shape[1] - max(0, -c_off))
    out[r_dst, c_dst] = arr[r_src, c_src]
    return out


def _write_raster(path, array, transform, crs) -> None:
    import rasterio
    with rasterio.open(
        path, "w", driver="GTiff",
        height=array.shape[0], width=array.shape[1], count=1,
        dtype=array.dtype, crs=crs, transform=transform, nodata=NODATA,
    ) as dst:
        dst.write(array, 1)


# ===========================================================================
# ArcPy back-end (ArcGIS Pro + Spatial Analyst / 3D Analyst)
# ===========================================================================


def run_pipeline_arcpy(
    dem: str,
    buildings: str,
    height_field: str,
    out_dsm: str,
    out_svf: str,
    cell_size: float | None = None,
) -> None:
    """DEM + buildings -> DSM -> SVF in ArcPy (Spatial Analyst extension).

    Reads an existing DEM raster, adds building heights with
    PolygonToRaster + Plus, then runs the *Sky View Factor* tool.
    """
    import arcpy
    from arcpy.sa import Plus, SkyViewFactor  # type: ignore

    arcpy.CheckOutExtension("Spatial")
    if cell_size is None:
        cell_size = arcpy.Describe(dem).meanCellWidth  # match the DEM grid
    arcpy.env.cellSize = cell_size
    arcpy.env.snapRaster = dem

    # 1. Buildings -> height raster, then DEM + buildings.
    bld_ras = "in_memory/bld_h"
    arcpy.conversion.PolygonToRaster(
        buildings, height_field, bld_ras, cellsize=cell_size
    )
    bld = arcpy.sa.Con(arcpy.sa.IsNull(bld_ras), 0, bld_ras)
    Plus(dem, bld).save(out_dsm)
    # 2. DSM -> SVF.
    SkyViewFactor(out_dsm, zenith_divisions=8, azimuth_divisions=16).save(out_svf)

    arcpy.CheckInExtension("Spatial")
    logger.info("ArcPy SVF saved -> %s", out_svf)


# ===========================================================================
# QGIS back-end (PyQGIS Processing: GDAL / SAGA providers)
# ===========================================================================


def run_pipeline_qgis(
    dem: str,
    buildings: str,
    height_field: str,
    out_dsm: str,
    out_svf: str,
    cell_size: float = 2.0,
    max_radius: float = 200.0,
) -> None:
    """DEM + buildings -> DSM -> SVF in QGIS Processing (QGIS Python console).

    Reads an existing DEM, rasterizes building heights and adds them with the
    raster calculator, then runs SAGA *Sky View Factor*.
    """
    import processing  # type: ignore
    from pathlib import Path as _P

    # 1. Buildings -> height raster aligned to the DEM.
    bld_ras = str(_P(out_dsm).with_name(_P(out_dsm).stem + "_bld.tif"))
    processing.run("gdal:rasterize", {
        "INPUT": buildings, "FIELD": height_field,
        "UNITS": 1, "WIDTH": cell_size, "HEIGHT": cell_size,
        "INIT": 0, "OUTPUT": bld_ras,
    })
    # 2. DSM = DEM + buildings.
    processing.run("gdal:rastercalculator", {
        "INPUT_A": dem, "BAND_A": 1,
        "INPUT_B": bld_ras, "BAND_B": 1,
        "FORMULA": "A + B", "OUTPUT": out_dsm,
    })
    # 3. DSM -> SVF (SAGA).
    processing.run("sagang:skyviewfactor", {
        "DEM": out_dsm, "RADIUS": max_radius,
        "METHOD": 1, "NDIRS": 8, "SVF": out_svf,
    })
    logger.info("QGIS/SAGA SVF saved -> %s", out_svf)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    # Folder layout: everything lives under one BASE folder, with each
    # input and each output stage in its own subfolder.  The DEM is an
    # existing raster that is read directly (not computed here).
    #
    #   BASE/
    #     01_buildings/ buildings.gpkg    <- input: footprints with height
    #     02_DEM/       dem.tif           <- input: existing DEM raster
    #     03_DSM/       dsm.tif           <- output: terrain + buildings
    #     04_SVF/       svf.tif           <- output: sky view factor
    #
    # Edit BASE (and the field name below) to match your data.
    # ------------------------------------------------------------------ #
    BASE = Path(r"D:\SVF_project")

    buildings = BASE / "01_buildings" / "buildings.gpkg"
    dem       = BASE / "02_DEM" / "dem.tif"     # read directly
    out_dsm   = BASE / "03_DSM" / "dsm.tif"
    out_svf   = BASE / "04_SVF" / "svf.tif"

    # Create the output subfolders if they don't exist yet
    for _p in (out_dsm, out_svf):
        _p.parent.mkdir(parents=True, exist_ok=True)

    run_pipeline_py(
        dem=dem,
        buildings=buildings,
        height_field="height",    # building height attribute (m)
        out_dsm=out_dsm,
        out_svf=out_svf,
        n_dirs=16,                # azimuth directions
        max_radius=200.0,         # search radius (m)
    )
