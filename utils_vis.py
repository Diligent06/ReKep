import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def visualize_sdf(sdf, title="SDF Visualization", cmap="coolwarm"):
    """
    Visualize central slices of a 3D SDF volume along each axis.

    Args:
        sdf (np.ndarray): 3D SDF array.
        title (str): Plot title.
        cmap (str): Colormap for visualization.
    """
    assert sdf.ndim == 3, "SDF must be a 3D array"
    cx, cy, cz = [s // 2 for s in sdf.shape]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(sdf[cx, :, :], cmap=cmap, origin="lower")
    axes[0].set_title(f"Axial slice (x={cx})")
    axes[1].imshow(sdf[:, cy, :], cmap=cmap, origin="lower")
    axes[1].set_title(f"Coronal slice (y={cy})")
    axes[2].imshow(sdf[:, :, cz], cmap=cmap, origin="lower")
    axes[2].set_title(f"Sagittal slice (z={cz})")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def visualize_sdf_open3d(sdf, voxel_size=1.0, level=0.0):
    """
    Visualize the zero level set (isosurface) of a 3D SDF using Open3D and marching cubes.
    Args:
        sdf (np.ndarray): 3D SDF array.
        voxel_size (float): Size of each voxel in world units.
        level (float): The isosurface value to extract (default 0.0 for zero level set).
    """
    import open3d as o3d
    from skimage import measure

    assert sdf.ndim == 3, "SDF must be a 3D array"
    # Marching cubes expects (z, y, x)
    verts, faces, normals, values = measure.marching_cubes(sdf, level=level)
    # Scale verts to world coordinates
    verts = verts * voxel_size
    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.7, 1.0])
    o3d.visualization.draw_geometries([mesh], window_name="SDF Isosurface (Open3D)")


# Usage example (in your main code):
# from utils_vis import visualize_sdf_open3d
# visualize_sdf_open3d(sdf_voxels, voxel_size=0.01)


def visualize_points_open3d(points, color=[1, 0, 0]):
    """
    Visualize 3D points as a point cloud using Open3D.
    Args:
        points (np.ndarray): Nx3 array of 3D points.
        color (list): RGB color for the points.
    """
    import open3d as o3d

    if isinstance(points, list):
        points = np.concatenate(points, axis=0)
    assert points.shape[1] == 3, "Points must be Nx3"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)
    o3d.visualization.draw_geometries([pcd], window_name="3D Collision Points")


# Usage example:
# from utils_vis import visualize_points_open3d
# visualize_points_open3d(collision_points)
#
def draw_pc(pc_arr):
    import os

    disable = os.environ.get("DISABLE_PC", "").lower()
    if disable in ("1", "true", "yes"):  # If disabled, just return
        return
    import open3d as o3d

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.01,  # 坐标轴的长度
        origin=[0, 0, 0],  # 坐标系的原点
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_arr)
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd, coordinate_frame])


def load_and_visualize_link_mesh(link_name, debug_dir="debug_link_meshes"):
    """
    Load a saved link mesh and pose from an .npz file and visualize it using Open3D.

    Args:
        link_name (str): Name of the link to load (used as filename).
        debug_dir (str): Directory where the .npz files are saved.
    """
    import os
    import numpy as np
    import open3d as o3d

    mesh_filename = os.path.join(debug_dir, f"{link_name}.npz")
    if not os.path.exists(mesh_filename):
        print(f"[load_and_visualize_link_mesh] File not found: {mesh_filename}")
        return

    try:
        data = np.load(mesh_filename, allow_pickle=True)
        vertices = data["vertices"]
        faces = data["faces"]
        pose_p = data["pose_p"]
        pose_q = data["pose_q"]

        # breakpoint()
        if vertices is not None and len(vertices) > 0:
            # Create Open3D mesh
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            if faces is not None and len(faces) > 0:
                mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.paint_uniform_color([0.7, 0.7, 1.0])  # Light blue color

            # Apply pose transformation if available
            if pose_p is not None and pose_q is not None:
                from scipy.spatial.transform import Rotation as R

                R_mat = R.from_quat(pose_q).as_matrix()
                transformation = np.eye(4)
                transformation[:3, :3] = R_mat
                transformation[:3, 3] = pose_p
                mesh.transform(transformation)

            # Create coordinate frame for reference
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.05, origin=[0, 0, 0]
            )

            # Visualize
            o3d.visualization.draw_geometries(
                [mesh, coordinate_frame], window_name=f"Link Mesh: {link_name}"
            )
        else:
            print(
                f"[load_and_visualize_link_mesh] No valid vertices found for {link_name}"
            )
    except Exception as e:
        print(
            f"[load_and_visualize_link_mesh] Failed to load/visualize mesh for {link_name}: {e}"
        )


if __name__ == "__main__":
    load_and_visualize_link_mesh("panda_hand")
