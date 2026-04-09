"""Functional Urban Area (FUA) extraction workflow.

This module codifies the workflow described in section 2.1:
- data preprocessing (clip + projection)
- NTL threshold optimization using Cohen's kappa
- region growing smoothing
- inclusion of adjacent urban segments

The implementation assumes QGIS helper functions (e.g., ``Qgis_Raster_Clip``)
exist in your runtime environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from simpledbf import Dbf5


AFRICA_CITIES = {
    "Accra",
    "Addis Ababa",
    "Alexandria",
    "Cairo",
    "Cape Town",
    "Casablanca",
    "Johannesburg",
    "Lagos",
    "Luanda",
    "Nairobi",
    "Tunis",
}


@dataclass(frozen=True)
class FUAPathConfig:
    """Path configuration for the FUA extraction pipeline."""

    base_dir: Path = Path(r"D:\Final_Paper")

    @property
    def city_data_dir(self) -> Path:
        return self.base_dir / "OSM" / "Data_City"

    @property
    def ntl(self) -> Path:
        return self.base_dir / "NTL" / "ntl.tif"

    @property
    def land_use_default(self) -> Path:
        return self.base_dir / "LandUse" / "Land_use.tif"

    @property
    def land_use_africa(self) -> Path:
        return self.base_dir / "LandUse" / "Africa_Land_use.tif"

    @property
    def boundary_buffer_dir(self) -> Path:
        return self.base_dir / "OSM" / "City_Boundary_Buffer"

    @property
    def boundary_dir(self) -> Path:
        return self.base_dir / "OSM" / "City_Boundary"


def _ensure_city_dirs(city_data_dir: Path, city: str) -> Path:
    city_root = city_data_dir / city
    for d in (city_root, city_root / "law", city_root / "temp"):
        d.mkdir(parents=True, exist_ok=True)
    return city_root / "temp"


def _land_use_path(city: str, cfg: FUAPathConfig) -> Path:
    return cfg.land_use_africa if city in AFRICA_CITIES else cfg.land_use_default


def _to_qgis_path(path: Path) -> str:
    """Normalize path string for QGIS helper wrappers."""
    return str(path)


def calculate_area(
    threshold: int,
    city: str,
    cfg: FUAPathConfig,
    raster_calc_fn: Callable,
    raster_to_polygon_fn: Callable,
) -> float:
    """Calculate Cohen's kappa between thresholded NTL and land-use."""

    city_dir = cfg.city_data_dir
    input1 = city_dir / city / f"ntl_{city}.tif"
    input2 = city_dir / city / f"land_use_{city}.tif"
    output_tif = city_dir / city / "temp" / "type.tif"

    if city in AFRICA_CITIES:
        urban_cond = '"{}@1" >= 30'
        non_urban_cond = '"{}@1" < 30'
    else:
        urban_cond = '"{}@1" = 13'
        non_urban_cond = '"{}@1" != 13'

    query = (
        '("{}@1" >= {} and ' + urban_cond + ') * 11 + '
        '("{}@1" < {} and ' + non_urban_cond + ') * 00 + '
        '("{}@1" >= {} and ' + non_urban_cond + ') * 10 + '
        '("{}@1" < {} and ' + urban_cond + ') * 01'
    ).format(
        _to_qgis_path(input1),
        threshold,
        _to_qgis_path(input2),
        _to_qgis_path(input1),
        threshold,
        _to_qgis_path(input2),
        _to_qgis_path(input1),
        threshold,
        _to_qgis_path(input2),
        _to_qgis_path(input1),
        threshold,
        _to_qgis_path(input2),
    )

    raster_calc_fn(
        [_to_qgis_path(input1), _to_qgis_path(input2)],
        _to_qgis_path(output_tif),
        query,
    )

    type_shp = city_dir / city / "temp" / "type.shp"
    raster_to_polygon_fn(_to_qgis_path(output_tif), _to_qgis_path(type_shp))

    dbf = Dbf5(str(city_dir / city / "temp" / "type.dbf"))
    df = dbf.to_dataframe()
    grouped = df.groupby("VALUE").size().to_dict()

    def _safe(v: int) -> int:
        return int(grouped.get(v, 0))

    c00, c01, c10, c11 = _safe(0), _safe(1), _safe(10), _safe(11)
    total = c00 + c01 + c10 + c11
    if total == 0:
        return -1.0

    pr_a = (c11 + c00) / total
    pr_true = ((c10 + c11) / total) * ((c01 + c11) / total)
    pr_false = ((c00 + c01) / total) * ((c00 + c10) / total)
    pr_e = pr_true + pr_false

    if pr_e == 1:
        return -1.0

    return (pr_a - pr_e) / (1 - pr_e)


def threshold_cal(
    st_num: int,
    radius: int,
    sep: int,
    cache: MutableMapping[int, float],
    city: str,
    score_fn: Callable[[int, str], float],
) -> MutableMapping[int, float]:
    """Recursive search for NTL threshold that maximizes kappa."""

    left = max(1, st_num - radius)
    right = st_num + radius

    for key in (st_num, left, right):
        if key not in cache:
            cache[key] = score_fn(key, city)

    if cache[st_num] >= cache[left] and cache[st_num] >= cache[right]:
        if sep == 1 and radius > 1:
            return threshold_cal(st_num, radius // 2, 0, cache, city, score_fn)
        return cache

    if cache[left] > cache[right] and sep == 1:
        return threshold_cal(left, radius, 1, cache, city, score_fn)

    if cache[right] > cache[left] and sep == 1:
        return threshold_cal(right, radius, 1, cache, city, score_fn)

    return cache


def extract_fua(
    region_dir: Iterable[str],
    d_city: Mapping[str, Sequence[str]],
    d_epsg: Mapping[str, int],
    cfg: FUAPathConfig,
):
    """Run the full FUA extraction process for all target cities."""

    threshold_list: Dict[str, int] = {}

    for region in region_dir:
        if region not in d_city:
            continue

        for city in d_city[region]:
            city_slug = city.replace(" ", "_")
            temp = _ensure_city_dirs(cfg.city_data_dir, city)
            land_use = _land_use_path(city, cfg)

            # Step1) Initial extent from 60 km squared city boundary buffer
            mask_layer = cfg.boundary_buffer_dir / f"{city_slug}_boundary.shp"
            extent = temp / "extent.shp"
            Qgis_Extract_Layer(_to_qgis_path(mask_layer), _to_qgis_path(extent))

            # Step2) Clip + projection for NTL and LandUse
            ntl_clip = temp / "ntl_rec.tif"
            Qgis_Raster_Clip(_to_qgis_path(cfg.ntl), _to_qgis_path(extent), _to_qgis_path(ntl_clip))

            ntl_city = cfg.city_data_dir / city / f"ntl_{city}.tif"
            Qgis_Raster_Projection(_to_qgis_path(ntl_clip), _to_qgis_path(ntl_city), f"EPSG:{d_epsg[city]}")

            extent_pr = temp / "extent_pr.shp"
            Qgis_Projection(_to_qgis_path(extent), _to_qgis_path(extent_pr), "ESRI:53008")

            land_clip = temp / "land_clip.tif"
            Qgis_Raster_Clip(_to_qgis_path(land_use), _to_qgis_path(extent_pr), _to_qgis_path(land_clip))

            land_city = cfg.city_data_dir / city / f"land_use_{city}.tif"
            Qgis_Raster_Projection(_to_qgis_path(land_clip), _to_qgis_path(land_city), f"EPSG:{d_epsg[city]}")

            # Step3) Search optimal NTL threshold by Kappa
            cache: Dict[int, float] = {}
            score_fn = lambda th, c: calculate_area(th, c, cfg, Qgis_Raster_Calculator, Qgis_Raster_to_Polygon)
            result = threshold_cal(10, 2, 1, cache, city, score_fn)
            threshold = max(result, key=result.get)
            threshold_list[city] = threshold

            # Step4) Preprocess NTL for region growing
            ntl_cal = temp / "ntl_cal.tif"
            query = '("{}@1" >= {}) * {} + ("{}@1" < {}) * "{}@1"'.format(
                _to_qgis_path(ntl_city), threshold, threshold, _to_qgis_path(ntl_city), threshold, _to_qgis_path(ntl_city)
            )
            Qgis_Raster_Calculator([_to_qgis_path(ntl_city)], _to_qgis_path(ntl_cal), query)

            # Step5) Region Growing + vectorization
            ntl_rga = temp / "ntl_rga.tif"
            Qgis_Region_Growing(_to_qgis_path(ntl_cal), _to_qgis_path(ntl_rga), 10, 0.5)

            ntl_rga_dis = temp / "ntl_rga_dis.shp"
            Qgis_Raster_to_Vector(_to_qgis_path(ntl_rga), _to_qgis_path(ntl_rga_dis))

            # Step6) Identify main segment overlapped with city boundary
            boundary_pr = temp / "boundary_pr.shp"
            Qgis_Projection(
                _to_qgis_path(cfg.boundary_dir / f"{city_slug}.shp"),
                _to_qgis_path(boundary_pr),
                f"EPSG:{d_epsg[city]}",
            )

            ntl_rga_dis_re = temp / "ntl_rga_dis_re.shp"
            Qgis_Fix_Geometries(_to_qgis_path(ntl_rga_dis), _to_qgis_path(ntl_rga_dis_re))

            union = temp / "union.shp"
            Qgis_Union(_to_qgis_path(boundary_pr), _to_qgis_path(ntl_rga_dis_re), _to_qgis_path(union))

            union_clip = temp / "union_clip.shp"
            Qgis_clip(_to_qgis_path(union), _to_qgis_path(boundary_pr), _to_qgis_path(union_clip))

            boundary_area = temp / "boundary_area.shp"
            Qgis_Field_Calculator(_to_qgis_path(union_clip), _to_qgis_path(boundary_area), "Area", "$area")
            df = Dbf5(str(temp / "boundary_area.dbf")).to_dataframe()
            max_area = df["Area"].max()

            boundary_select = temp / "boundary_select.shp"
            Qgis_Extract_by_Attr(_to_qgis_path(boundary_area), _to_qgis_path(boundary_select), "Area", 0, max_area)

            boundary_info = temp / "boundary_info.shp"
            Qgis_Field_Calculator(_to_qgis_path(boundary_select), _to_qgis_path(boundary_info), "Check", 1)

            spatial_join = temp / "spatal_join.shp"
            Qgis_Spatial_Join(
                _to_qgis_path(ntl_rga_dis_re),
                _to_qgis_path(boundary_info),
                ["Check"],
                0,
                0,
                _to_qgis_path(spatial_join),
            )

            # Step7) Include adjacent zones + finalize boundary
            ntl_area = temp / "ntl_area.shp"
            Qgis_Field_Calculator(_to_qgis_path(spatial_join), _to_qgis_path(ntl_area), "Area", "$area")

            ntl_df = Dbf5(str(temp / "ntl_area.dbf")).to_dataframe()
            candidate = ntl_df[ntl_df["Check"] == 1]
            max_area = candidate["Area"].max()

            ntl_select = temp / "ntl_select.shp"
            Qgis_Extract_by_Attr(_to_qgis_path(ntl_area), _to_qgis_path(ntl_select), "Area", 0, max_area)

            boundary_fin = temp / "boundary_fin.shp"
            Qgis_Delete_Holes(_to_qgis_path(ntl_select), _to_qgis_path(boundary_fin))

            Qgis_Zonal_Stat(_to_qgis_path(ntl_city), _to_qgis_path(ntl_rga_dis_re), [2])

            zonal_select = temp / "zonal_select.shp"
            Qgis_Extract_by_Attr(_to_qgis_path(ntl_rga_dis_re), _to_qgis_path(zonal_select), "_mean", 3, threshold)

            boundary_buffer_1m = temp / "boundary_buffer_1m.shp"
            Qgis_Buffer(_to_qgis_path(boundary_fin), _to_qgis_path(boundary_buffer_1m), 1)

            zonal_spjoin_1k = temp / "zonal_spjoin_1k.shp"
            Qgis_Spatial_Join(
                _to_qgis_path(zonal_select),
                _to_qgis_path(boundary_buffer_1m),
                ["Check"],
                0,
                0,
                _to_qgis_path(zonal_spjoin_1k),
            )

            semi_extent = temp / "semi_extent.shp"
            Qgis_Extract_by_Attr(_to_qgis_path(zonal_spjoin_1k), _to_qgis_path(semi_extent), "Check", 0, 1)

            boundary_buffer_1k = temp / "boundary_buffer_1k.shp"
            Qgis_Buffer(_to_qgis_path(semi_extent), _to_qgis_path(boundary_buffer_1k), 1000)

            zonal_spjoin = temp / "zonal_spjoin.shp"
            Qgis_Spatial_Join(
                _to_qgis_path(zonal_select),
                _to_qgis_path(boundary_buffer_1k),
                ["Check"],
                0,
                0,
                _to_qgis_path(zonal_spjoin),
            )

            final_extent = temp / "final_extent.shp"
            Qgis_Extract_by_Attr(_to_qgis_path(zonal_spjoin), _to_qgis_path(final_extent), "Check", 0, 1)

            final_extent2 = temp / "final_extent2.shp"
            Qgis_Delete_Holes(_to_qgis_path(final_extent), _to_qgis_path(final_extent2))

            final_boundary = cfg.city_data_dir / city / f"{city}_boundary.shp"
            Qgis_Dissolve(_to_qgis_path(final_extent2), _to_qgis_path(final_boundary), "")

    return threshold_list
