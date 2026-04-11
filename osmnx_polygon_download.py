"""Download OSM data within a polygon boundary using osmnx.

Supports:
- Road network (all types or filtered by network type)
- Buildings
- Points of Interest (POI / amenities)
- Land use polygons
- Greenspace (parks, forests, etc.)

The polygon can be supplied as:
- A Shapefile / GeoPackage path
- A place name resolved by the Nominatim geocoder
- A list of (lat, lon) coordinate tuples

Outputs are saved as GeoPackage layers under a user-defined output directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Sequence

import geopandas as gpd
import osmnx as ox
from shapely.geometry import MultiPolygon, Polygon, mapping

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

NetworkType = Literal["all", "all_public", "bike", "drive", "drive_service", "walk"]


# ---------------------------------------------------------------------------
# Polygon loading utilities
# ---------------------------------------------------------------------------


def polygon_from_file(path: str | Path, layer: int | str = 0) -> Polygon | MultiPolygon:
    """Load the first (or specified) geometry from a vector file and return its union.

    Parameters
    ----------
    path:
        Path to a Shapefile, GeoPackage, GeoJSON, etc.
    layer:
        Layer name or index (used for multi-layer formats such as GeoPackage).
    """
    gdf = gpd.read_file(path, layer=layer)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf.geometry.unary_union


def polygon_from_place(place_name: str) -> Polygon | MultiPolygon:
    """Geocode a place name with Nominatim and return its boundary polygon.

    Parameters
    ----------
    place_name:
        Human-readable place name, e.g. ``"Seoul, South Korea"``.
    """
    gdf = ox.geocode_to_gdf(place_name)
    return gdf.geometry.unary_union


def polygon_from_coords(coords: Sequence[tuple[float, float]]) -> Polygon:
    """Build a Shapely polygon from a sequence of ``(lat, lon)`` tuples.

    Parameters
    ----------
    coords:
        Ring of coordinates in **(latitude, longitude)** order.
    """
    # shapely uses (x, y) = (lon, lat)
    return Polygon([(lon, lat) for lat, lon in coords])


# ---------------------------------------------------------------------------
# Download functions
# ---------------------------------------------------------------------------


def download_road_network(
    polygon: Polygon | MultiPolygon,
    network_type: NetworkType = "all",
    output_path: str | Path | None = None,
) -> gpd.GeoDataFrame:
    """Download the road network inside *polygon* and optionally save it.

    Parameters
    ----------
    polygon:
        Boundary polygon (EPSG:4326).
    network_type:
        OSMnx network type: ``"all"``, ``"drive"``, ``"walk"``, ``"bike"``, etc.
    output_path:
        If given, the edges GeoDataFrame is saved to this path (GeoPackage or
        Shapefile inferred from extension).

    Returns
    -------
    GeoDataFrame of road edges.
    """
    logger.info("Downloading road network (network_type=%s) ...", network_type)
    G = ox.graph_from_polygon(polygon, network_type=network_type)
    _, edges = ox.graph_to_gdfs(G)
    edges = edges.reset_index()

    if output_path is not None:
        _save_gdf(edges, output_path, layer="roads")
        logger.info("Road network saved → %s", output_path)

    logger.info("  %d road segments downloaded.", len(edges))
    return edges


def download_buildings(
    polygon: Polygon | MultiPolygon,
    output_path: str | Path | None = None,
) -> gpd.GeoDataFrame:
    """Download building footprints inside *polygon*.

    Parameters
    ----------
    polygon:
        Boundary polygon (EPSG:4326).
    output_path:
        Optional save path.

    Returns
    -------
    GeoDataFrame of building polygons.
    """
    logger.info("Downloading building footprints ...")
    tags = {"building": True}
    gdf = ox.features_from_polygon(polygon, tags=tags)
    gdf = _keep_polygon_geom(gdf)

    if output_path is not None:
        _save_gdf(gdf, output_path, layer="buildings")
        logger.info("Buildings saved → %s", output_path)

    logger.info("  %d building footprints downloaded.", len(gdf))
    return gdf


def download_pois(
    polygon: Polygon | MultiPolygon,
    amenity_filter: list[str] | None = None,
    output_path: str | Path | None = None,
) -> gpd.GeoDataFrame:
    """Download Points of Interest (POI / amenities) inside *polygon*.

    Parameters
    ----------
    polygon:
        Boundary polygon (EPSG:4326).
    amenity_filter:
        Subset of amenity values to keep, e.g. ``["school", "hospital"]``.
        Pass ``None`` to download all amenities.
    output_path:
        Optional save path.

    Returns
    -------
    GeoDataFrame of POI features.
    """
    logger.info("Downloading POIs ...")
    tags = {"amenity": amenity_filter if amenity_filter else True}
    gdf = ox.features_from_polygon(polygon, tags=tags)

    if output_path is not None:
        _save_gdf(gdf, output_path, layer="pois")
        logger.info("POIs saved → %s", output_path)

    logger.info("  %d POI features downloaded.", len(gdf))
    return gdf


def download_landuse(
    polygon: Polygon | MultiPolygon,
    output_path: str | Path | None = None,
) -> gpd.GeoDataFrame:
    """Download land-use polygons inside *polygon*.

    Parameters
    ----------
    polygon:
        Boundary polygon (EPSG:4326).
    output_path:
        Optional save path.

    Returns
    -------
    GeoDataFrame of land-use polygons.
    """
    logger.info("Downloading land-use polygons ...")
    tags = {"landuse": True}
    gdf = ox.features_from_polygon(polygon, tags=tags)
    gdf = _keep_polygon_geom(gdf)

    if output_path is not None:
        _save_gdf(gdf, output_path, layer="landuse")
        logger.info("Land use saved → %s", output_path)

    logger.info("  %d land-use polygons downloaded.", len(gdf))
    return gdf


def download_greenspace(
    polygon: Polygon | MultiPolygon,
    output_path: str | Path | None = None,
) -> gpd.GeoDataFrame:
    """Download parks, forests, and other green spaces inside *polygon*.

    Parameters
    ----------
    polygon:
        Boundary polygon (EPSG:4326).
    output_path:
        Optional save path.

    Returns
    -------
    GeoDataFrame of green-space polygons.
    """
    logger.info("Downloading green spaces ...")
    tags = {
        "leisure": ["park", "garden", "nature_reserve", "recreation_ground"],
        "landuse": ["forest", "grass", "meadow", "orchard", "village_green"],
        "natural": ["wood", "scrub", "heath", "grassland"],
    }
    gdf = ox.features_from_polygon(polygon, tags=tags)
    gdf = _keep_polygon_geom(gdf)

    if output_path is not None:
        _save_gdf(gdf, output_path, layer="greenspace")
        logger.info("Green space saved → %s", output_path)

    logger.info("  %d green-space polygons downloaded.", len(gdf))
    return gdf


# ---------------------------------------------------------------------------
# Convenience: download everything at once
# ---------------------------------------------------------------------------


def download_all(
    polygon: Polygon | MultiPolygon,
    output_dir: str | Path,
    city_name: str = "city",
    network_type: NetworkType = "all",
) -> dict[str, gpd.GeoDataFrame]:
    """Download roads, buildings, POIs, land use, and green spaces in one call.

    All layers are saved to a single GeoPackage named
    ``{city_name}_osm.gpkg`` inside *output_dir*.

    Parameters
    ----------
    polygon:
        Boundary polygon (EPSG:4326).
    output_dir:
        Directory where outputs are written (created if it doesn't exist).
    city_name:
        Prefix used for the output file name.
    network_type:
        OSMnx network type for road download.

    Returns
    -------
    Dictionary mapping layer names to their GeoDataFrames.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gpkg = output_dir / f"{city_name}_osm.gpkg"

    results: dict[str, gpd.GeoDataFrame] = {}

    results["roads"] = download_road_network(polygon, network_type, gpkg)
    results["buildings"] = download_buildings(polygon, gpkg)
    results["pois"] = download_pois(polygon, output_path=gpkg)
    results["landuse"] = download_landuse(polygon, gpkg)
    results["greenspace"] = download_greenspace(polygon, gpkg)

    logger.info("All layers saved to %s", gpkg)
    return results


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _keep_polygon_geom(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop point / line geometries, keeping only (Multi)Polygons."""
    mask = gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    return gdf[mask].copy()


def _save_gdf(gdf: gpd.GeoDataFrame, path: str | Path, layer: str) -> None:
    """Save a GeoDataFrame; format is inferred from the file extension."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in {".gpkg"}:
        # Append mode so multiple layers can coexist in one GeoPackage
        gdf.to_file(path, layer=layer, driver="GPKG")
    elif ext in {".shp"}:
        gdf.to_file(path)
    elif ext in {".geojson", ".json"}:
        gdf.to_file(path, driver="GeoJSON")
    else:
        # Default to GeoPackage
        gdf.to_file(path, layer=layer, driver="GPKG")


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    # Option 1: polygon from a place name (Nominatim geocoding)
    # ------------------------------------------------------------------ #
    polygon_place = polygon_from_place("Gangnam-gu, Seoul, South Korea")

    # ------------------------------------------------------------------ #
    # Option 2: polygon loaded from an existing Shapefile / GeoPackage
    # ------------------------------------------------------------------ #
    # polygon_file = polygon_from_file("path/to/boundary.shp")

    # ------------------------------------------------------------------ #
    # Option 3: polygon defined manually as (lat, lon) coordinates
    # ------------------------------------------------------------------ #
    # coords = [
    #     (37.514, 127.019),
    #     (37.514, 127.100),
    #     (37.557, 127.100),
    #     (37.557, 127.019),
    #     (37.514, 127.019),
    # ]
    # polygon_manual = polygon_from_coords(coords)

    # ------------------------------------------------------------------ #
    # Download individual layers
    # ------------------------------------------------------------------ #
    out_dir = Path("osm_output") / "Gangnam"

    roads = download_road_network(
        polygon_place,
        network_type="drive",
        output_path=out_dir / "roads.gpkg",
    )

    buildings = download_buildings(
        polygon_place,
        output_path=out_dir / "buildings.gpkg",
    )

    pois = download_pois(
        polygon_place,
        amenity_filter=["school", "hospital", "restaurant", "cafe", "bank"],
        output_path=out_dir / "pois.gpkg",
    )

    # ------------------------------------------------------------------ #
    # OR download everything into a single GeoPackage
    # ------------------------------------------------------------------ #
    # results = download_all(
    #     polygon_place,
    #     output_dir="osm_output",
    #     city_name="Gangnam",
    #     network_type="drive",
    # )
