import laspy
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import math
import pyvista as pv  # New import

import datetime


def save_snapshot(plotter, dpi=600, scale_factor=4):
    """
    Saves a high-resolution snapshot (screenshot) of the current plotter view.

    Parameters:
        plotter (pv.Plotter): The active PyVista plotter object.
        dpi (int): The target Dots Per Inch (for naming convention).
        scale_factor (int): Multiplier for the rendering resolution.
                            A value of 4 is usually excellent for high-DPI output.
    """
    # Define the output file path based on the current time
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"/mnt/d/Sync/research/tree_species_estimation/paper/TL-TSC-paper/images/snapshot_{timestamp}_{dpi}dpi.png"

    # Save the screenshot using the scale factor
    # This renders the current view at (window_size * scale_factor) resolution
    plotter.screenshot(
        output_filename,
        scale=scale_factor,
        transparent_background=True,  # Optional: useful if placing image on colored background
    )

    print(f"\n📸 Snapshot saved (Scaled by {scale_factor}) to: {output_filename}")


# ================================
# 1. READ LAS/LAZ FILE
# ================================
las = laspy.read(
    "/mnt/d/Sync/research/tree_species_estimation/paper/TL-TSC-paper/images/1kmZ165890537902021L.laz"
)

x = las.x
y = las.y
z = las.z
points = np.vstack([x, y, z]).T


# ================================
# 2. READ POLYGON FROM SHP
# ================================
gdf = gpd.read_file(
    "/mnt/d/Sync/research/tree_species_estimation/paper/TL-TSC-paper/images/superpixel_plot_41825.shp"
)
polygon = gdf.geometry.iloc[0]

center_x, center_y = polygon.centroid.x, polygon.centroid.y
radius = math.sqrt(polygon.area / math.pi)

print("Center:", center_x, center_y)
print("Radius:", radius)


# ================================
# 3. CLIP POINTS USING POLYGON
# ================================
mask = np.array(
    [polygon.contains(Point(px, py)) for px, py in zip(points[:, 0], points[:, 1])]
)

clipped_points = points[mask]


# ============================================
# 4. CONVERT TO PYVISTA DATA STRUCTURES (NEW)
# ============================================

# Full cloud: PyVista PolyData from numpy array
pcd_full = pv.PolyData(points)

# Clipped cloud: PyVista PolyData from numpy array
pcd_clip = pv.PolyData(clipped_points)

output_pc_file = "/mnt/d/Sync/research/tree_species_estimation/paper/TL-TSC-paper/images/clipped_point_cloud.ply"

# Use the .save() method
try:
    pcd_clip.save(output_pc_file)
    print(f"\n✅ Clipped point cloud successfully saved to: {output_pc_file}")
except Exception as e:
    print(f"\n❌ Error saving point cloud: {e}")
# ============================================
# 5. CREATE CYLINDER GEOMETRY (PYVISTA)
# ============================================
z_min = z.min()
z_max = z.max()
height = z_max - z_min

# Create the cylinder mesh
cyl_mesh = pv.Cylinder(
    center=(center_x, center_y, z_min),
    direction=(0, 0, 1),
    radius=radius,
    height=height,
)

output_file = "/mnt/d/Sync/research/tree_species_estimation/paper/TL-TSC-paper/images/clipped_cylinder.stl"

# Use the .save() method
try:
    cyl_mesh.save(output_file)
    print(f"\n✅ Cylinder mesh successfully saved to: {output_file}")
except Exception as e:
    print(f"\n❌ Error saving mesh: {e}")
# ================================
# 6. VISUALIZATION USING PYVISTA
# ================================

# Initialize a PyVista Plotter (the main viewer window)
plotter = pv.Plotter(window_size=(1024, 768))

# Add the Full Point Cloud (Background)
# We use 'render_points_as_spheres=True' for better visibility
plotter.add_mesh(
    pcd_full,
    color="lightgrey",
    point_size=2.0,
    label="Full Cloud (1)",
    render_points_as_spheres=True,
    name="full",
)

# Add the Clipped Point Cloud (Foreground/Target)
plotter.add_mesh(
    pcd_clip,
    color="green",
    point_size=5.0,
    label="Clipped Cloud (2)",
    render_points_as_spheres=True,
    name="clip",
)

# Add the Cylinder as a wireframe
plotter.add_mesh(
    cyl_mesh,  # .extract_all_edges(),  # Only plot the edges for a wireframe look
    color="grey",
    opacity=0.3,
    show_edges=True,
    label="Cylinder",
    name="cyl",
)

# --- PyVista Interaction: Add Key Bindings (Toggles) ---
def toggle_visibility(name):
    """Toggles the visibility of a mesh actor by name."""
    actor = plotter.actors.get(name)
    if actor:
        is_visible = actor.GetVisibility()
        actor.SetVisibility(not is_visible)
        plotter.render()


# Register key callbacks (1, 2, 3)
plotter.add_key_event("1", lambda: toggle_visibility("full"))
plotter.add_key_event("2", lambda: toggle_visibility("clip"))
plotter.add_key_event("3", lambda: toggle_visibility("cyl"))
plotter.add_key_event("s", lambda: save_snapshot(plotter))

# Add a simple Legend
plotter.add_legend()

# Set an initial camera position for a good view (optional but recommended)
plotter.camera_position = "iso"

# Show the plot
plotter.show(full_screen=False)
