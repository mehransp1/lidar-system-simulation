import numpy as np

def generate_lidar_directions(vertical_res=16, horizontal_res=360, vertical_fov=(-15, 15)):
    elev = np.radians(np.linspace(*vertical_fov, vertical_res))
    azim = np.radians(np.linspace(0, 360, horizontal_res, endpoint=False))
    dirs = [[np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)] for e in elev for a in azim]
    return np.array(dirs)

def simulate_lidar_scan(mesh, origin=np.array([0, 0, 1.0]), vertical_res=16, horizontal_res=360):
    dirs = generate_lidar_directions(vertical_res, horizontal_res)
    origins = np.tile(origin, (dirs.shape[0], 1))
    locations, index_ray, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
    return origins[index_ray], locations
