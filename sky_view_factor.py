"""Sky View Factor (SVF) calculation from building footprints.

The Sky View Factor is the fraction of the visible sky hemisphere at a
given ground location, ranging from 0 (fully obstructed) to 1 (open sky).
It is a key morphological indicator in urban-climate and thermal-comfort
studies (e.g. urban heat island, mean radiant temperature).

This module implements a **2.5-D vector ray-casting** method that works
directly on the OSM building footprints produced by
``osmnx_polygon_download.py`` (a GeoPackage ``buildings`` layer with a
height attribute).  For each observation point it:

1. casts ``n_rays`` azimuth rays out to ``max_radius`` metres,
2. finds, along each ray, the building edge that subtends the largest
   vertical (elevation) angle ``beta`` = atan(building_height / distance),
3. combines the per-ray obstruction angles into an SVF estimate using the
   widely used isotropic-sky approximation (Oke, 1987; Watson & Johnson,
   1987)::

       SVF = 1 - (1 / N) * sum_{i=1..N} sin^2(beta_i)

Everything is pure ``geopandas`` / ``shapely`` — no ArcPy or QGIS runtime
required — but see the module docstrings at the bottom for equivalent
ArcPy and QGIS entry points if you prefer a raster (DSM-based) workflow.

Typical usage
-------------
>>> import geopandas as gpd
>>> from sky_view_factor import svf_for_points, sample_points_in_polygon
>>> buildings = gpd.read_file("Seoul_osm.gpkg", layer="buildings")
>>> pts = sample_points_in_polygon(buildings.union_all(), spacing=100)
>>> svf = svf_for_points(pts, buildings, n_rays=36, max_radius=150)
>>> svf.to_file("Seoul_svf.gpkg", layer="svf", driver="GPKG")
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, MultiPolygon, Polygon
from shapely.strtree import STRtree

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Assumed storey height (metres) when a building only has a levels count.
DEFAULT_STOREY_HEIGHT = 3.0
# Fallback height (metres) when neither height nor levels is available.
DEFAULT_BUILDING_HEIGHT = 9.0


# ---------------------------------------------------------------------------
# Height attribute handling
# ---------------------------------------------------------------------------


def _parse_number(value) -> float | None:
    """Best-effort parse of an OSM height/levels tag into a float.

    Handles strings such as ``"12"``, ``"12 m"``, ``"12.5"`` and returns
    ``None`` for missing or unparseable values.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        # Strip a trailing unit like " m" and retry on the first token.
        try:
            token = str(value).strip().split()[0].replace(",", ".")
            return float(token)
        except (ValueError, IndexError):
            return None


def ensure_height(
    buildings: gpd.GeoDataFrame,
    height_col: str = "height",
    levels_col: str = "building:levels",
    storey_height: float = DEFAULT_STOREY_HEIGHT,
    default_height: float = DEFAULT_BUILDING_HEIGHT,
) -> gpd.GeoDataFrame:
    """Return a copy of *buildings* with a numeric ``_height_m`` column.

    Height is resolved in priority order:

    1. ``height_col`` (metres),
    2. ``levels_col`` * *storey_height*,
    3. *default_height*.

    Parameters
    ----------
    buildings:
        Building footprints (any CRS).
    height_col, levels_col:
        Attribute names to look for.
    storey_height:
        Metres per storey used to convert levels to height.
    default_height:
        Height assigned when no attribute is available.
    """
    gdf = buildings.copy()

    heights = np.full(len(gdf), np.nan)

    if height_col in gdf.columns:
        parsed = gdf[height_col].map(_parse_number).astype("float64")
        heights = np.where(parsed.notna(), parsed, heights)

    if levels_col in gdf.columns:
        need = np.isnan(heights)
        levels = gdf[levels_col].map(_parse_number).astype("float64")
        from_levels = levels * storey_height
        heights = np.where(need & from_levels.notna(), from_levels, heights)

    n_missing = int(np.isnan(heights).sum())
    if n_missing:
        logger.info(
            "  %d/%d buildings have no height/levels -> default %.1f m",
            n_missing, len(gdf), default_height,
        )
    heights = np.where(np.isnan(heights), default_height, heights)

    gdf["_height_m"] = heights
    return gdf


# ---------------------------------------------------------------------------
# Observation point generation
# ---------------------------------------------------------------------------


def sample_points_in_polygon(
    boundary: Polygon | MultiPolygon,
    spacing: float,
    crs: str | int | None = None,
) -> gpd.GeoDataFrame:
    """Create a regular grid of observation points inside *boundary*.

    Parameters
    ----------
    boundary:
        Study-area polygon.  **Must be in a projected (metric) CRS** so that
        *spacing* is in metres; pass *crs* if the polygon carries no CRS.
    spacing:
        Grid spacing in metres.
    crs:
        CRS to attach to the returned GeoDataFrame.

    Returns
    -------
    GeoDataFrame of Point geometries with a ``pid`` column.
    """
    minx, miny, maxx, maxy = boundary.bounds
    xs = np.arange(minx + spacing / 2, maxx, spacing)
    ys = np.arange(miny + spacing / 2, maxy, spacing)
    pts = [Point(x, y) for x in xs for y in ys]
    gdf = gpd.GeoDataFrame(geometry=pts, crs=crs)
    gdf = gdf[gdf.within(boundary)].reset_index(drop=True)
    gdf.insert(0, "pid", range(len(gdf)))
    return gdf


def points_along_lines(
    lines: gpd.GeoDataFrame,
    spacing: float,
) -> gpd.GeoDataFrame:
    """Place observation points every *spacing* metres along line geometries.

    Useful for computing SVF along a street network (the ``roads`` layer
    from ``osmnx_polygon_download.py``).

    Parameters
    ----------
    lines:
        Line GeoDataFrame in a projected (metric) CRS.
    spacing:
        Distance between consecutive points in metres.
    """
    pts: list[Point] = []
    for geom in lines.geometry:
        if geom is None or geom.is_empty:
            continue
        length = geom.length
        if length == 0:
            continue
        n = max(int(length // spacing), 1)
        for d in np.linspace(0, length, n + 1):
            pts.append(geom.interpolate(d))
    gdf = gpd.GeoDataFrame(geometry=pts, crs=lines.crs)
    gdf.insert(0, "pid", range(len(gdf)))
    return gdf


# ---------------------------------------------------------------------------
# Core SVF computation
# ---------------------------------------------------------------------------


def _obstruction_angle_along_ray(
    origin: Point,
    azimuth_rad: float,
    max_radius: float,
    tree: STRtree,
    geoms: np.ndarray,
    heights: np.ndarray,
) -> float:
    """Largest elevation angle (radians) of any building along one ray.

    A ray is a line segment from *origin* out to *max_radius* in the
    direction *azimuth_rad* (measured counter-clockwise from east, in the
    projected plane).  For every building the ray intersects, the elevation
    angle is ``atan(height / horizontal_distance)`` evaluated at the nearest
    intersection point; the maximum over all buildings is returned.
    """
    ox, oy = origin.x, origin.y
    ex = ox + max_radius * math.cos(azimuth_rad)
    ey = oy + max_radius * math.sin(azimuth_rad)
    ray = LineString([(ox, oy), (ex, ey)])

    max_beta = 0.0
    for idx in tree.query(ray):
        geom = geoms[idx]
        if not ray.intersects(geom):
            continue
        inter = ray.intersection(geom.boundary)
        # Nearest intersection distance to the origin.
        if inter.is_empty:
            continue
        if inter.geom_type == "Point":
            dist = origin.distance(inter)
        else:
            # MultiPoint / GeometryCollection: take the closest component.
            dist = min(origin.distance(g) for g in inter.geoms)
        if dist <= 0:
            # Origin lies inside a building footprint -> fully blocked here.
            return math.pi / 2
        beta = math.atan(heights[idx] / dist)
        if beta > max_beta:
            max_beta = beta
    return max_beta


def svf_at_point(
    origin: Point,
    tree: STRtree,
    geoms: np.ndarray,
    heights: np.ndarray,
    n_rays: int = 36,
    max_radius: float = 150.0,
) -> float:
    """Sky View Factor at a single point using the isotropic approximation.

    ``SVF = 1 - mean( sin^2(beta_i) )`` over *n_rays* equally spaced azimuths.

    Parameters
    ----------
    origin:
        Observation point (same projected CRS as the buildings).
    tree, geoms, heights:
        Prebuilt spatial index, building geometry array and matching height
        array (see :func:`svf_for_points`).
    n_rays:
        Number of azimuth samples (36 = every 10 degrees).
    max_radius:
        Search radius in metres; buildings beyond this are ignored.
    """
    total = 0.0
    for k in range(n_rays):
        az = 2.0 * math.pi * k / n_rays
        beta = _obstruction_angle_along_ray(
            origin, az, max_radius, tree, geoms, heights
        )
        s = math.sin(beta)
        total += s * s
    return 1.0 - total / n_rays


def svf_for_points(
    points: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    n_rays: int = 36,
    max_radius: float = 150.0,
    height_col: str = "height",
    levels_col: str = "building:levels",
    target_crs: str | int | None = None,
) -> gpd.GeoDataFrame:
    """Compute SVF for every point in *points* against *buildings*.

    Both inputs are reprojected to a common **metric** CRS.  If *target_crs*
    is not given, the estimated UTM CRS of the buildings is used so that all
    distances are in metres.

    Parameters
    ----------
    points:
        Observation points (any CRS).
    buildings:
        Building footprints with a height / levels attribute (any CRS).
    n_rays:
        Number of azimuth rays per point.
    max_radius:
        Search radius in metres.
    height_col, levels_col:
        Attribute names used by :func:`ensure_height`.
    target_crs:
        Optional projected CRS to compute in.

    Returns
    -------
    Copy of *points* (in the working CRS) with an added ``svf`` column.
    """
    if buildings.crs is None:
        raise ValueError("buildings must have a defined CRS.")

    if target_crs is None:
        target_crs = buildings.estimate_utm_crs()
    logger.info("Working CRS: %s", target_crs)

    bld = ensure_height(buildings, height_col, levels_col).to_crs(target_crs)
    bld = bld[bld.geometry.notna() & ~bld.geometry.is_empty]
    pts = points.to_crs(target_crs).reset_index(drop=True)

    geoms = np.array(bld.geometry.values, dtype=object)
    heights = bld["_height_m"].to_numpy(dtype="float64")
    tree = STRtree(geoms)

    logger.info(
        "Computing SVF for %d points (%d rays, %.0f m radius, %d buildings)...",
        len(pts), n_rays, max_radius, len(geoms),
    )

    svf_values = np.empty(len(pts))
    for i, geom in enumerate(pts.geometry):
        svf_values[i] = svf_at_point(
            geom, tree, geoms, heights, n_rays=n_rays, max_radius=max_radius
        )
        if (i + 1) % 500 == 0:
            logger.info("  %d/%d points done", i + 1, len(pts))

    out = pts.copy()
    out["svf"] = svf_values
    logger.info("Done. SVF range %.3f - %.3f (mean %.3f)",
                out["svf"].min(), out["svf"].max(), out["svf"].mean())
    return out


# ---------------------------------------------------------------------------
# Convenience: run straight from a GeoPackage
# ---------------------------------------------------------------------------


def svf_from_gpkg(
    gpkg_path: str | Path,
    buildings_layer: str = "buildings",
    spacing: float = 100.0,
    n_rays: int = 36,
    max_radius: float = 150.0,
    output_layer: str = "svf",
    output_path: str | Path | None = None,
) -> gpd.GeoDataFrame:
    """End-to-end SVF from an OSM GeoPackage created by this project.

    Loads the ``buildings`` layer, samples a regular grid of observation
    points over the buildings' extent, computes SVF, and (optionally) writes
    the result back to a GeoPackage.

    Parameters
    ----------
    gpkg_path:
        Path to the ``*_osm.gpkg`` file.
    buildings_layer:
        Name of the buildings layer.
    spacing:
        Grid spacing (metres) for observation points.
    n_rays, max_radius:
        SVF sampling parameters.
    output_layer:
        Layer name for the written result.
    output_path:
        Where to save (defaults to *gpkg_path*, appending a new layer).

    Returns
    -------
    GeoDataFrame of points with an ``svf`` column.
    """
    gpkg_path = Path(gpkg_path)
    buildings = gpd.read_file(gpkg_path, layer=buildings_layer)
    logger.info("Loaded %d buildings from %s", len(buildings), gpkg_path)

    metric = buildings.estimate_utm_crs()
    boundary = buildings.to_crs(metric).union_all()
    points = sample_points_in_polygon(boundary, spacing=spacing, crs=metric)
    logger.info("Sampled %d observation points (%.0f m grid)", len(points), spacing)

    svf = svf_for_points(
        points, buildings, n_rays=n_rays, max_radius=max_radius, target_crs=metric
    )

    out_path = Path(output_path) if output_path else gpkg_path
    svf.to_file(out_path, layer=output_layer, driver="GPKG")
    logger.info("SVF saved -> %s (layer '%s')", out_path, output_layer)
    return svf


# ---------------------------------------------------------------------------
# Alternative back-ends (raster / DSM based)
# ---------------------------------------------------------------------------


def svf_arcpy_from_dsm(dsm_raster: str, out_raster: str) -> None:
    """SVF from a Digital Surface Model using ArcPy Spatial Analyst.

    Requires the Spatial Analyst extension.  ArcGIS exposes SVF through the
    ``SkyViewFactor`` raster function / the *Sky View Factor* geoprocessing
    tool, which analyses a DSM (buildings + terrain) rather than vector
    footprints.

    >>> svf_arcpy_from_dsm("dsm.tif", "svf.tif")
    """
    import arcpy  # noqa: F401  (only importable inside ArcGIS Pro)
    from arcpy.sa import SkyViewFactor  # type: ignore

    arcpy.CheckOutExtension("Spatial")
    # Zenith divisions / azimuth divisions control angular resolution.
    result = SkyViewFactor(dsm_raster, zenith_divisions=8, azimuth_divisions=16)
    result.save(out_raster)
    arcpy.CheckInExtension("Spatial")
    logger.info("ArcPy SVF raster saved -> %s", out_raster)


def svf_qgis_from_dsm(dsm_raster: str, out_raster: str, radius: float = 10.0) -> None:
    """SVF from a DSM using QGIS + the SAGA / GDAL processing providers.

    Run inside the QGIS Python console or a standalone PyQGIS script with
    Processing initialised.  Uses SAGA's *Sky View Factor* algorithm.

    >>> svf_qgis_from_dsm("dsm.tif", "svf.tif", radius=10)
    """
    import processing  # type: ignore  (available in the QGIS environment)

    processing.run(
        "sagang:skyviewfactor",
        {
            "DEM": dsm_raster,
            "RADIUS": radius,
            "METHOD": 1,          # multi-scale
            "DLEVEL": 3.0,
            "NDIRS": 8,           # azimuth directions
            "SVF": out_raster,
        },
    )
    logger.info("QGIS/SAGA SVF raster saved -> %s", out_raster)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example: compute a 100 m SVF grid for a city GeoPackage produced by
    # osmnx_polygon_download.py.  Edit the path to match your data.
    svf_from_gpkg(
        gpkg_path=r"D:\000_SCI\10_Compact_city\OSM_data\Seoul\Seoul_osm.gpkg",
        buildings_layer="buildings",
        spacing=100.0,
        n_rays=36,
        max_radius=150.0,
        output_layer="svf",
    )
