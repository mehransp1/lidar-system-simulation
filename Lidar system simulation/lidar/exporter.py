import open3d as o3d

def export_point_cloud(points, filename="outputs/scan.ply"):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved point cloud to {filename}")
    return pcd
