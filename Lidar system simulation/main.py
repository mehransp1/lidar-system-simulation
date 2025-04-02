"""
Main script for the enhanced LiDAR system simulation.
"""

import os
import yaml
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import open3d as o3d

from lidar.sensor_model import SensorModel, SensorSpecs
from lidar.environment import EnvironmentModel, WeatherParams
from lidar.scene_loader import load_scene
from lidar.raytracer import RayTracer
from lidar.processor import PointCloudProcessor
from lidar.visualizer import LiDARVisualizer

@dataclass
class SimulationConfig:
    """Configuration for the LiDAR simulation."""
    sensor: Dict[str, Any]
    environment: Dict[str, Any]
    processing: Dict[str, Any]
    visualization: Dict[str, Any]

def load_config(config_path: str) -> SimulationConfig:
    """
    Load simulation configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        SimulationConfig object
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return SimulationConfig(**config_dict)

def main():
    """Main simulation function."""
    # Create necessary directories
    os.makedirs("scene", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    try:
        # Load configuration
        config = load_config("configs/lidar_config.yaml")
        
        # Initialize sensor model
        sensor_specs = SensorSpecs(**config.sensor)
        sensor = SensorModel(sensor_specs)
        
        # Initialize environment model
        weather_params = WeatherParams(**config.environment)
        environment = EnvironmentModel(weather_params)
        
        # Load scene
        print("Loading scene...")
        mesh = load_scene("scene/urban_scene.stl")
        
        # Initialize ray tracer
        raytracer = RayTracer(mesh)
        
        # Initialize point cloud processor
        processor = PointCloudProcessor(**config.processing)
        
        # Initialize visualizer
        visualizer = LiDARVisualizer(**config.visualization)
        
        # Simulate LiDAR scan
        print("Simulating LiDAR scan...")
        origin = np.array([0, 0, 1.0])  # Sensor position
        beam_directions = sensor.get_scan_pattern()
        
        # Perform ray tracing
        points, normals = raytracer.trace_rays(origin, beam_directions)
        
        # Add sensor noise
        points = sensor.add_noise(points)
        
        # Apply environmental effects
        intensities = sensor.get_intensity(points, normals)
        intensities = environment.apply_attenuation(points, intensities)
        points = environment.add_weather_noise(points)
        
        # Process point cloud
        print("Processing point cloud...")
        processed_points = processor.process(points, intensities)
        
        # Export results
        print("Exporting results...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(processed_points)
        o3d.io.write_point_cloud("outputs/scan_output.ply", pcd)
        
        # Visualize results
        print("Visualizing results...")
        visualizer.visualize(processed_points, intensities)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
