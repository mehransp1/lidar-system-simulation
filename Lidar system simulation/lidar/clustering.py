import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def cluster_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    labels = DBSCAN(eps=0.5, min_samples=10).fit(points).labels_
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = [0, 0, 0, 1]
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    geometries = [pcd]
    for i in range(max_label + 1):
        cluster = points[labels == i]
        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster)
        aabb = cluster_pcd.get_axis_aligned_bounding_box()
        aabb.color = (1, 0, 0)
        geometries.append(aabb)

    o3d.visualization.draw_geometries(geometries)
