"""
Ray tracing module for LiDAR simulation.
Handles efficient ray-mesh intersection calculations.
"""

import numpy as np
import trimesh
from typing import Tuple, Optional

class RayTracer:
    """Efficient ray tracer for LiDAR simulation."""
    
    def __init__(self, mesh: trimesh.Trimesh):
        """
        Initialize the ray tracer with a mesh.
        
        Args:
            mesh: Trimesh object representing the scene
        """
        self.mesh = mesh
        self._setup_acceleration_structure()
    
    def _setup_acceleration_structure(self) -> None:
        """Setup acceleration structure for faster ray tracing."""
        # Use trimesh's built-in acceleration structure
        self.mesh.vertices = np.asarray(self.mesh.vertices, dtype=np.float32)
        self.mesh.faces = np.asarray(self.mesh.faces, dtype=np.int32)
        self.mesh.face_normals = np.asarray(self.mesh.face_normals, dtype=np.float32)
    
    def trace_rays(self, origin: np.ndarray, directions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trace rays from origin in given directions.
        
        Args:
            origin: 3D position of the sensor
            directions: Nx3 array of ray directions
            
        Returns:
            Tuple of (hit points, surface normals)
        """
        # Create ray origins array
        origins = np.tile(origin, (directions.shape[0], 1))
        
        # Perform ray-mesh intersection
        locations, index_ray, index_tri = self.mesh.ray.intersects_location(
            origins,
            directions,
            multiple_hits=False
        )
        
        # Get surface normals at hit points
        normals = self.mesh.face_normals[index_tri]
        
        return locations, normals
    
    def get_occlusion(self, points: np.ndarray, sensor_position: np.ndarray) -> np.ndarray:
        """
        Calculate occlusion for a set of points.
        
        Args:
            points: Nx3 array of points
            sensor_position: 3D position of the sensor
            
        Returns:
            Boolean array indicating which points are occluded
        """
        # Calculate directions from sensor to points
        directions = points - sensor_position
        directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
        
        # Trace rays from sensor to points
        origins = np.tile(sensor_position, (points.shape[0], 1))
        locations, index_ray, _ = self.mesh.ray.intersects_location(
            origins,
            directions,
            multiple_hits=False
        )
        
        # Points are occluded if their ray intersection is not at the point itself
        distances = np.linalg.norm(points - sensor_position, axis=1)
        hit_distances = np.linalg.norm(locations - sensor_position, axis=1)
        
        return np.abs(distances - hit_distances) > 1e-6
    
    def get_incidence_angles(self, points: np.ndarray, normals: np.ndarray,
                           sensor_position: np.ndarray) -> np.ndarray:
        """
        Calculate incidence angles for a set of points.
        
        Args:
            points: Nx3 array of points
            normals: Nx3 array of surface normals
            sensor_position: 3D position of the sensor
            
        Returns:
            Array of incidence angles in radians
        """
        # Calculate directions from sensor to points
        directions = points - sensor_position
        directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
        
        # Calculate incidence angles
        incidence_angles = np.arccos(
            np.abs(np.sum(normals * directions, axis=1))
        )
        
        return incidence_angles 