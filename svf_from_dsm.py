"""Sky View Factor (SVF) directly from a DSM raster.

Self-contained: if you already have a DSM (Digital Surface Model = terrain +
buildings in one raster), this is the only file you need.  It reads the DSM,
runs a horizon scan, and writes an SVF GeoTIFF (values 0..1).

Method
------
For every cell (height z0) the algorithm samples ``n_dirs`` azimuth
directions.  In each direction it marches outward one pixel step at a time up
to ``max_radius`` and tracks the largest horizon (elevation) angle ``beta``.
The per-direction angles are combined with the isotropic-sky approximation
(Oke 1987; Watson & Johnson 1987)::

    SVF = 1 - (1 / N) * sum_{i=1..N} sin^2(beta_i)

0 = fully obstructed sky, 1 = completely open sky.

Requirements
------------
    pip install rasterio numpy            # matplotlib only for the preview

The DSM **must be in a projected (metric) CRS** so the pixel size and
``max_radius`` are in metres (UTM, a national grid, etc. -- not lat/lon).

Usage
-----
    python svf_from_dsm.py                 # edit the paths in __main__, or
    from svf_from_dsm import svf_from_dsm
    svf_from_dsm("dsm.tif", "svf.tif", n_dirs=16, max_radius=200)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import rasterio

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NODATA = -9999.0


def _shift2d(arr: np.ndarray, r_off: int, c_off: int, fill=np.nan) -> np.ndarray:
    """Shift a 2-D array by (r_off, c_off) pixels, padding edges with *fill*."""
    out = np.full_like(arr, fill)
    r_src = slice(max(0, -r_off), arr.shape[0] - max(0, r_off))
    c_src = slice(max(0, -c_off), arr.shape[1] - max(0, c_off))
    r_dst = slice(max(0, r_off), arr.shape[0] - max(0, -r_off))
    c_dst = slice(max(0, c_off), arr.shape[1] - max(0, -c_off))
    out[r_dst, c_dst] = arr[r_src, c_src]
    return out


def svf_from_dsm(
    dsm: str | Path,
    out_svf: str | Path,
    n_dirs: int = 16,
    max_radius: float = 200.0,
) -> str:
    """Compute a Sky View Factor raster from a DSM GeoTIFF.

    Parameters
    ----------
    dsm:
        Input DSM raster (terrain + buildings), projected/metric CRS.
    out_svf:
        Output SVF GeoTIFF path (values 0..1, ``NODATA`` where the DSM is
        nodata).
    n_dirs:
        Number of azimuth directions (16 = every 22.5 degrees).  More = finer
        but slower.
    max_radius:
        Search radius in metres; obstructions beyond this are ignored.

    Returns
    -------
    Path to the written SVF raster (as a string).
    """
    with rasterio.open(dsm) as src:
        z = src.read(1).astype("float64")
        transform = src.transform
        profile = src.profile
        nodata = src.nodata
        crs = src.crs

    if crs is not None and crs.is_geographic:
        logger.warning(
            "DSM CRS %s looks geographic (degrees). Reproject to a metric CRS "
            "(e.g. UTM) or max_radius/pixel size will be wrong.", crs,
        )

    px = abs(transform.a)          # pixel width  (m)
    py = abs(transform.e)          # pixel height (m)
    cell = (px + py) / 2.0
    max_steps = max(int(max_radius / cell), 1)
    nrows, ncols = z.shape

    if nodata is not None:
        z = np.where(z == nodata, np.nan, z)
    valid = ~np.isnan(z)
    z0 = np.where(valid, z, 0.0)

    logger.info(
        "SVF from DSM %s | %d dirs | %.0f m radius (%d steps @ %.2f m)",
        z.shape, n_dirs, max_radius, max_steps, cell,
    )

    sin2_sum = np.zeros_like(z)
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
            tan_ang = np.where(np.isnan(shifted), -np.inf, (shifted - z0) / dist)
            np.maximum(max_tan, tan_ang, out=max_tan)

        horizon = np.arctan(np.where(np.isfinite(max_tan), max_tan, 0.0))
        horizon = np.clip(horizon, 0.0, math.pi / 2)   # sky only; ignore dips
        sin2_sum += np.sin(horizon) ** 2
        logger.info("  direction %d/%d done", d + 1, n_dirs)

    svf = np.where(valid, 1.0 - sin2_sum / n_dirs, NODATA)

    profile.update(dtype="float32", count=1, nodata=NODATA)
    Path(out_svf).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_svf, "w", **profile) as dst:
        dst.write(svf.astype("float32"), 1)

    good = svf[svf != NODATA]
    logger.info("SVF saved -> %s (range %.3f - %.3f, mean %.3f)",
                out_svf, good.min(), good.max(), good.mean())
    return str(out_svf)


if __name__ == "__main__":
    # ---- edit these two paths, then run:  python svf_from_dsm.py ----------
    DSM_IN  = r"D:\SVF_project\DSM.tif"
    OUT_SVF = r"D:\SVF_project\SVF.tif"

    svf_from_dsm(
        DSM_IN,
        OUT_SVF,
        n_dirs=16,        # azimuth directions (16 = every 22.5 deg)
        max_radius=200.0, # search radius in metres
    )

    # Optional quick-look preview -----------------------------------------
    # import matplotlib.pyplot as plt
    # with rasterio.open(OUT_SVF) as src:
    #     svf = src.read(1)
    # svf = np.where(svf == NODATA, np.nan, svf)
    # plt.imshow(svf, cmap="viridis", vmin=0, vmax=1)
    # plt.colorbar(label="SVF"); plt.title("Sky View Factor"); plt.axis("off")
    # plt.show()
