import os
import glob
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import box
from rasterio.features import rasterize
from tqdm import tqdm
from joblib import Parallel, delayed
import json

def filter_fri_shapefile(fri_shapefile_path, output_shapefile_path, pids_to_keep):
    fri_gdf = gpd.read_file(fri_shapefile_path)
    fri_gdf['POLYID'] = fri_gdf['POLYID'].astype(int)
    filtered_gdf = fri_gdf[fri_gdf['POLYID'].isin(pids_to_keep)]
    filtered_gdf.to_file(output_shapefile_path, driver='ESRI Shapefile')
    print(f"Filtered shapefile saved to {output_shapefile_path}")

def safe_parse_specs(x):
    if isinstance(x, str):
        return np.array(json.loads(x), dtype=np.float32)
    elif isinstance(x, (list, np.ndarray)):
        return np.array(x, dtype=np.float32)
    else:
        raise ValueError("Invalid perc_specs format")

def generate_superpixels(tile_path, season, size_threshold, shapefile, output_dir):
    tile_name = os.path.splitext(os.path.basename(tile_path))[0]
    output_file = os.path.join(output_dir, f"{tile_name}.npz")

    with rasterio.open(tile_path) as src:
        tile_image = src.read()
        tile_transform = src.transform
        tile_crs = src.crs
        tile_bounds = src.bounds
        tile_height, tile_width = src.height, src.width
        nodata_value = src.nodata

    nodata_mask = np.any(tile_image == nodata_value, axis=0) if nodata_value is not None else np.zeros((tile_height, tile_width), dtype=bool)

    polygons_gdf = gpd.read_file(shapefile)
    if polygons_gdf.crs != tile_crs:
        polygons_gdf = polygons_gdf.to_crs(tile_crs)

    if 'POLYID' not in polygons_gdf.columns:
        raise ValueError("Missing 'POLYID' column.")
    if 'perc_specs' not in polygons_gdf.columns:
        raise ValueError("Missing 'perc_specs' column.")

    polygons_gdf['POLYID'] = polygons_gdf['POLYID'].astype(int)
    tile_bbox = box(*tile_bounds)
    intersecting_polygons = polygons_gdf[polygons_gdf.intersects(tile_bbox)].copy()

    if intersecting_polygons.empty:
        print(f"No polygons intersect tile {tile_name}. Skipping.")
        return []

    intersecting_polygons['geometry'] = intersecting_polygons.geometry.map(lambda geom: geom.intersection(tile_bbox))
    shapes = zip(intersecting_polygons.geometry, intersecting_polygons['POLYID'])

    superpixel_mask = rasterize(
        shapes=shapes,
        out_shape=(tile_height, tile_width),
        transform=tile_transform,
        fill=0,
        all_touched=True,
        dtype='int32'
    )

    superpixel_mask[nodata_mask] = 0
    num_classes = 11
    label_array = np.zeros((num_classes, tile_height, tile_width), dtype=np.float32)

    intersecting_polygons['perc_specs'] = intersecting_polygons['perc_specs'].apply(safe_parse_specs)
    polygon_id_to_label = dict(zip(intersecting_polygons['POLYID'], intersecting_polygons['perc_specs']))

    # Create mask before modifying sp_ids
    sp_ids, counts = np.unique(superpixel_mask, return_counts=True)
    nonzero_mask = sp_ids != 0

    # Apply mask once to both
    sp_ids = sp_ids[nonzero_mask]
    counts = counts[nonzero_mask]

    small_sp_ids = [sp_id for sp_id, count in zip(sp_ids, counts) if count < size_threshold]
    for sp_id in small_sp_ids:
        superpixel_mask[superpixel_mask == sp_id] = 0

    sp_ids = np.unique(superpixel_mask)
    sp_ids = sp_ids[sp_ids != 0]

    for sp_id in sp_ids:
        indices = np.where(superpixel_mask == sp_id)
        label_vector = polygon_id_to_label.get(sp_id)
        if label_vector is None or label_vector.shape[0] != num_classes:
            raise ValueError(f"Label for polyid {sp_id} has invalid shape.")
        for band in range(num_classes):
            label_array[band, indices[0], indices[1]] = label_vector[band]

    label_array[:, nodata_mask] = 0.0
    tile_image[:, nodata_mask] = 0.0

    np.savez_compressed(
        output_file,
        tile_image=tile_image,
        label_array=label_array,
        superpixel_mask=superpixel_mask,
        nodata_mask=nodata_mask
    )

    return sp_ids.tolist()

if __name__ == "__main__":
    shapefile = "/mnt/d/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/ovf_fri/ovf_fri_superpixel_updated.gpkg"
    out_shapefile = "/mnt/d/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/ovf_fri/superpixel.shp"
    size_threshold = 25

    all_pids = set()
    seasons = ["spring", "summer", "fall", "winter"]
    for season in seasons:
        tile_dir = f"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/ovf_s2/{season}/tiles_128"
        output_dir = f"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/ovf_s2/{season}/superpixel"
        os.makedirs(output_dir, exist_ok=True)

        tile_paths = glob.glob(os.path.join(tile_dir, "*.tif"))
        print(f"Processing {len(tile_paths)} tiles for season: {season}")

        results = Parallel(n_jobs=4)(
            delayed(generate_superpixels)(tile_path, season, size_threshold, shapefile, output_dir)
            for tile_path in tqdm(tile_paths)
        )

        for pids in results:
            all_pids.update(pids)

    # Save filtered shapefile once
    filter_fri_shapefile(shapefile, out_shapefile, all_pids)