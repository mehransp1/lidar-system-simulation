"""
LiDAR sensor model implementation.
Handles sensor specifications, beam pattern, and noise characteristics.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class SensorSpecs:
    """Hardware specifications for a LiDAR sensor."""
    type: str
    vertical_resolution: int
    horizontal_resolution: int
    vertical_fov: Tuple[float, float]
    horizontal_fov: float
    scan_rate: float
    max_range: float
    min_range: float
    noise_std: float

class SensorModel:
    """LiDAR sensor model with beam pattern and noise characteristics."""
    
    def __init__(self, specs: SensorSpecs):
        """
        Initialize the sensor model with given specifications.
        
        Args:
            specs: SensorSpecs object containing hardware parameters
        """
        self.specs = specs
        self._generate_beam_pattern()
    
    def _generate_beam_pattern(self) -> None:
        """Generate the beam pattern based on sensor specifications."""
        # Generate vertical angles
        v_min, v_max = self.specs.vertical_fov
        self.vertical_angles = np.linspace(
            np.radians(v_min),
            np.radians(v_max),
            self.specs.vertical_resolution
        )
        
        # Generate horizontal angles
        self.horizontal_angles = np.linspace(
            0,
            np.radians(self.specs.horizontal_fov),
            self.specs.horizontal_resolution,
            endpoint=False
        )
        
        # Generate beam directions
        self.beam_directions = np.array([
            [
                np.cos(v) * np.cos(h),
                np.cos(v) * np.sin(h),
                np.sin(v)
            ]
            for v in self.vertical_angles
            for h in self.horizontal_angles
        ])
    
    def add_noise(self, points: np.ndarray) -> np.ndarray:
        """
        Add realistic noise to point cloud data.
        
        Args:
            points: Nx3 array of 3D points
            
        Returns:
            Noisy point cloud
        """
        noise = np.random.normal(
            scale=self.specs.noise_std,
            size=points.shape
        )
        return points + noise
    
    def get_scan_pattern(self) -> np.ndarray:
        """
        Get the complete scan pattern for the sensor.
        
        Returns:
            Array of beam directions
        """
        return self.beam_directions
    
    def get_scan_points(self, origin: np.ndarray) -> np.ndarray:
        """
        Generate scan points from a given origin.
        
        Args:
            origin: 3D position of the sensor
            
        Returns:
            Array of scan points
        """
        return origin + self.beam_directions * self.specs.max_range
    
    def get_intensity(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """
        Calculate intensity returns based on incidence angle and distance.
        
        Args:
            points: Nx3 array of 3D points
            normals: Nx3 array of surface normals
            
        Returns:
            Array of intensity values
        """
        # Calculate incidence angles
        incidence_angles = np.arccos(
            np.abs(np.sum(normals * self.beam_directions, axis=1))
        )
        
        # Calculate distances
        distances = np.linalg.norm(points, axis=1)
        
        # Simple intensity model (can be made more sophisticated)
        intensity = np.exp(-distances / self.specs.max_range) * np.cos(incidence_angles)
        intensity = np.clip(intensity, 0, 1)
        
        return intensity 