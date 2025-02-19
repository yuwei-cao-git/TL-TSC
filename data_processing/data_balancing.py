import geopandas as gpd
import pandas as pd
import numpy as np
import random
import re
from tqdm import tqdm
from shapely.geometry import Point
from shapely.affinity import rotate, scale
from sklearn.model_selection import train_test_split

# ------------------- CONSTANTS ------------------- #
AUGMENTATION_FACTOR = 3  # Number of augmented copies per plot
BUFFER_DISTANCE = 50  # Meters from polygon boundary
PLOT_RADIUS = 11.28  # Meters (≈400m² area)
SPECIES_ORDER = [
    "BF",
    "BW",
    "MR",
    "OR",
    "PO",
    "PW",
    "CE",
    "SW",
    "MH",
    "PR",
    "AB",
    "BE",
    "IW",
    "LA",
    "SB",
    "BY",
    "HE",
    "BD",
    "PJ",
    "AW",
    "PS",
    "SR",
]


# ------------------- CORE FUNCTIONS ------------------- #
def create_points(polygon, number, distance=50):
    """Generate spatially constrained random points within a polygon"""
    points = []
    min_x, min_y, max_x, max_y = polygon.bounds
    attempts = 0
    while len(points) < number and attempts < 100:
        point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        buffered = point.buffer(11.28)  # 11.28m radius ≈ 0.04ha plot
        if (
            polygon.contains(buffered)
            and polygon.boundary.distance(buffered) >= distance
        ):
            # Check if the point is within the polygon and not too close to others
            if polygon.contains(point) and all(
                point.distance(existing) >= distance for existing in points
            ):
                points.append(point)
                attempts = 0  # Reset attempts after successful placement
        attempts += 1
    return points


def random_plots(boundary, target_samples, distance=50):
    """Generate balanced samples for a composition group"""
    if len(boundary) == 0:
        return None

    crs = boundary.crs
    all_points = []
    max_per_poly = 10  # Prevent over-sampling single polygons

    # Calculate samples per polygon
    base_samples = max(1, target_samples // len(boundary))
    remainder = target_samples % len(boundary)

    for idx in range(len(boundary)):
        poly_data = boundary.iloc[idx]
        n = min(base_samples + 1 if idx < remainder else base_samples, max_per_poly)
        points = create_points(poly_data.geometry, n)
        if points:
            points_gdf = gpd.GeoDataFrame(
                {
                    "geometry": points,
                    "POLYID": poly_data["POLYID"],  # Inherit POLYID
                    "OSPCOMP": poly_data["OSPCOMP"],  # Inherit composition
                },
                crs=crs,
            )
            all_points.append(points_gdf)

    return (
        gpd.GeoDataFrame(pd.concat(all_points, ignore_index=True))
        if all_points
        else None
    )


# ------------------- DATA AUGMENTATION ------------------- #
def augment_plot(plot_geom, parent_polygon):
    """Generate valid augmented geometries"""
    valid_augs = []
    centroid = parent_polygon.centroid

    # Original plot
    valid_augs.append(plot_geom)

    # Flip horizontally/vertically
    for xfact, yfact in [(-1, 1), (1, -1)]:
        flipped = scale(plot_geom, xfact=xfact, yfact=yfact, origin=centroid)
        if validate_plot(flipped, parent_polygon):
            valid_augs.append(flipped)

    # Rotate (90°, 180°, 270°)
    for angle in [90, 180, 270]:
        rotated = rotate(plot_geom, angle, origin=centroid)
        if validate_plot(rotated, parent_polygon):
            valid_augs.append(rotated)

    return valid_augs[:AUGMENTATION_FACTOR]  # Return max AUGMENTATION_FACTOR copies


def validate_plot(geom, parent_polygon):
    """Check spatial constraints"""
    return (
        parent_polygon.contains(geom)
        and parent_polygon.boundary.distance(geom) >= BUFFER_DISTANCE
    )


def augment_dataset(gdf):
    """Augment plots while retaining attributes"""
    augmented_rows = []

    for comp in tqdm(gdf["OSPCOMP"].unique(), desc="Augmenting"):
        comp_subset = gdf[gdf["OSPCOMP"] == comp]
        parent_poly = comp_subset.union_all()

        for _, row in comp_subset.iterrows():
            aug_geoms = augment_plot(row.geometry, parent_poly)
            for geom in aug_geoms:
                new_row = row.copy()
                new_row.geometry = geom
                augmented_rows.append(new_row)

    return gpd.GeoDataFrame(pd.DataFrame(augmented_rows), crs=gdf.crs)


# ------------------- DATA PROCESSING ------------------- #
def parse_ospcomp(ospcomp):
    """Parse composition to species proportions"""
    species_dict = {}
    matches = re.findall(r"([A-Z]+)\s*(\d+)", ospcomp)
    for species, perc in matches:
        species_dict[species] = species_dict.get(species, 0) + int(perc)
    return species_dict


def add_perc_specs(gdf):
    """Add fixed-dimension species proportion vector"""
    perc_specs = []
    # SPECIES_ORDER = (gdf["OSPCOMP"].apply(parse_ospcomp).apply(pd.Series).fillna(0).columns.tolist())
    for _, row in gdf.iterrows():
        species_props = parse_ospcomp(row["OSPCOMP"])
        vector = {s: 0.0 for s in SPECIES_ORDER}
        for s, p in species_props.items():
            if s in vector:
                vector[s] = p / 100
        perc_specs.append([vector[s] for s in SPECIES_ORDER])

    gdf = gdf.copy()
    gdf["perc_specs"] = perc_specs
    return gdf


# ------------------- BALANCING ALGORITHM ------------------- #
def balance_compositions(gdf):
    """Tiered composition balancing"""
    comp_counts = gdf.groupby("OSPCOMP").size().reset_index(name="count")
    balanced_dfs = []

    for _, row in tqdm(
        comp_counts.iterrows(),
        total=len(comp_counts),
        colour="green",
        desc="training dataset composition balancing",
    ):
        comp, count = row["OSPCOMP"], row["count"]

        # Tiered target formula
        if count >= 1000:
            target = 100
        elif 100 <= count < 1000:
            target = 150
        else:
            target = 200

        subset = gdf[gdf["OSPCOMP"] == comp]
        plots = random_plots(subset, target, 25)
        if plots is not None:
            plots["OSPCOMP"] = comp
            balanced_dfs.append(plots)

    return gpd.GeoDataFrame(pd.concat(balanced_dfs, ignore_index=True))


def gen_plots_nontrain(gdf):
    comp_counts = gdf.groupby("OSPCOMP").size().reset_index(name="count")
    plot_dfs = []

    for _, row in tqdm(
        comp_counts.iterrows(),
        total=len(comp_counts),
        colour="green",
        desc="val/test plot generation",
    ):
        comp, count = row["OSPCOMP"], row["count"]

        subset = gdf[gdf["OSPCOMP"] == comp]
        plots = random_plots(subset, 2, 50)
        if plots is not None:
            plots["OSPCOMP"] = comp
            plot_dfs.append(plots)

    return gpd.GeoDataFrame(pd.concat(plot_dfs, ignore_index=True))


def calculate_species_targets(gdf, percentile=75, min_target=100):
    """
    Calculate species targets based on their frequency distribution.

    Args:
        gdf: GeoDataFrame after composition balancing.
        percentile: Target percentile for capping (0-100).
        min_target: Minimum samples for rare species.
    """
    # Step 1: Calculate total species counts
    species_counts = {}
    for _, row in gdf.iterrows():
        species_props = parse_ospcomp(row["OSPCOMP"])
        for species in species_props:
            species_counts[species] = species_counts.get(species, 0) + 1

    # Step 2: Compute percentile-based target
    counts = np.array(list(species_counts.values()))
    target = np.percentile(counts, percentile)

    # Step 3: Assign targets
    targets = {}
    for species, count in species_counts.items():
        if count > target:
            targets[species] = int(target)
        else:
            targets[species] = max(min_target, count)  # Protect rares

    return targets


"""
def balance_species_proportions(gdf, species_targets):
    # Species-aware proportional balancing
    species_props = []
    for _, row in gdf.iterrows():
        props = parse_ospcomp(row["OSPCOMP"])
        species_props.append(props)

    # Calculate overrepresentation
    sp_counts = {}
    for props in species_props:
        for s, p in props.items():
            key = f"{s}_{p}%"
            sp_counts[key] = sp_counts.get(key, 0) + 1

    # Downsample overrepresented species-proportions
    gdf = gdf.copy()
    indices_to_keep = []
    for sp_key, count in tqdm(
        sp_counts.items(),
        total=len(sp_counts),
        colour="red",
        desc="proportions balancing",
    ):
        species = sp_key.split("_")[0]
        target = species_targets.get(species, 100)
        if count <= target:
            indices_to_keep.extend(gdf.index.tolist())
            continue

        # Get contributing indices
        contributing = [
            i
            for i, props in enumerate(species_props)
            if species in props
            and props[species] == int(sp_key.split("_")[1].replace("%", ""))
        ]

        # Downsample
        keep_rate = target / count
        np.random.seed(42)
        keep_indices = np.random.choice(contributing, int(target), replace=False)
        indices_to_keep.extend(keep_indices)

    # Apply filtering
    return gdf.loc[list(set(indices_to_keep))]
"""


def balance_species_proportions(gdf, species_targets):
    # Downsample plots to meet species-level targets.
    # Track remaining quota for each species
    quota = {s: t for s, t in species_targets.items()}

    # Prioritize plots with rare species
    plot_scores = []
    for idx, row in gdf.iterrows():
        species = list(parse_ospcomp(row["OSPCOMP"]).keys())
        score = sum(1 / quota[s] for s in species)  # Lower score = more rare
        plot_scores.append((idx, score))

    # Sort plots by rarity (ascending score = rare-first)
    plot_scores.sort(key=lambda x: x[1])

    # Select plots until quotas are filled
    selected_indices = []
    for idx, score in plot_scores:
        species = list(parse_ospcomp(gdf.iloc[idx]["OSPCOMP"]).keys())
        if all(quota[s] > 0 for s in species):
            selected_indices.append(idx)
            for s in species:
                quota[s] -= 1

    return gdf.iloc[selected_indices]


def balance_training_set(train):
    """
    Balance the training set using tiered composition balancing.
    """
    # Tiered composition balancing
    comp_balanced = balance_compositions(train)

    # Species-proportion balancing
    species_targets = calculate_species_targets(
        comp_balanced, percentile=75, min_target=100
    )
    sp_balanced = balance_species_proportions(comp_balanced, species_targets)

    return sp_balanced


def ensure_min_samples(gdf, min_samples):
    """
    Ensure each class has at least `min_samples` in the validation set.
    """
    balanced_samples = []
    for comp, group in gdf.groupby("OSPCOMP"):
        if len(group) < min_samples:
            # Oversample rare classes
            balanced_samples.append(
                group.sample(min_samples, replace=True, random_state=42)
            )
        else:
            balanced_samples.append(group)
    return gpd.GeoDataFrame(pd.concat(balanced_samples, ignore_index=True))


def split_dataset(gdf, test_size=0.15, val_size=0.15):
    # Stratify by composition group to preserve balance
    train_val, test = train_test_split(
        gdf, test_size=test_size, stratify=gdf["OSPCOMP"], random_state=42
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        stratify=train_val["OSPCOMP"],
        random_state=42,
    )
    val = ensure_min_samples(val, min_samples=2)
    return train, val, test


# ------------------- MAIN WORKFLOW ------------------- #
if __name__ == "__main__":
    # Load data
    ovf_fri_clean = gpd.read_file(
        r"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/ovf_fri/ovf_fri_prominate10.gpkg"
    )

    # 1. Split dataset: as don't want to balance the validation and test dataset
    _, val_polygons, test_polygons = split_dataset(
        ovf_fri_clean, test_size=0.3, val_size=0.3
    )
    val = add_perc_specs(gen_plots_nontrain(val_polygons))
    print(f"validate plots: {len(val)}")
    test = add_perc_specs(gen_plots_nontrain(test_polygons))
    print(f"test plots: {len(test)}")

    val.to_file(
        r"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/ovf_fri/plot_val_prominant10.gpkg"
    )
    test.to_file(
        r"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/ovf_fri/plot_test_prominant10.gpkg"
    )
"""
    # 2. Balance training set
    train_polygons, _, _ = split_dataset(
        ovf_fri_clean, test_size=0.15, val_size=0.15
    )
    train_balanced = balance_training_set(train_polygons)

    # 4. Add fixed-dimension vectors
    sp_balanced = add_perc_specs(train_balanced)

    # 5. Augment training data
    train_augmented = augment_dataset(sp_balanced)
    full_train = gpd.GeoDataFrame(
        pd.concat([sp_balanced, train_augmented], ignore_index=True),
        crs=train_balanced.crs,
    )

    # 6. Save datasets
    full_train.to_file(
        r"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/ovf_fri/plot_train_prominant10.gpkg"
    )
"""
