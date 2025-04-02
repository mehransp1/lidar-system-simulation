"""
Visualization module for LiDAR simulation.
Handles real-time and static visualization of point clouds.
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

class LiDARVisualizer:
    """Visualizer for LiDAR point cloud data."""
    
    def __init__(self, **kwargs):
        """
        Initialize the visualizer.
        
        Args:
            **kwargs: Visualization parameters
        """
        self.live_view = kwargs.get('live_view', True)
        self.save_frames = kwargs.get('save_frames', False)
        self.frame_rate = kwargs.get('frame_rate', 10)
        self.color_scheme = kwargs.get('color_scheme', 'intensity')
        
        if self.live_view:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()
    
    def visualize(self, points: np.ndarray, intensities: Optional[np.ndarray] = None) -> None:
        """
        Visualize point cloud data.
        
        Args:
            points: Nx3 array of 3D points
            intensities: Optional N array of intensity values
        """
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Set colors based on color scheme
        if self.color_scheme == 'intensity' and intensities is not None:
            colors = self._intensity_to_color(intensities)
        elif self.color_scheme == 'height':
            colors = self._height_to_color(points)
        else:
            colors = self._default_color(points.shape[0])
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        if self.live_view:
            self._live_visualization(pcd)
        else:
            self._static_visualization(pcd)
    
    def _live_visualization(self, pcd: o3d.geometry.PointCloud) -> None:
        """
        Perform live visualization of point cloud.
        
        Args:
            pcd: Open3D point cloud object
        """
        # Add geometry to visualizer
        self.vis.add_geometry(pcd)
        
        # Set up camera view
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.8)
        
        # Run visualization
        while True:
            self.vis.poll_events()
            self.vis.update_renderer()
            
            if self.save_frames:
                self._save_frame()
            
            if not self.vis.poll_events():
                break
    
    def _static_visualization(self, pcd: o3d.geometry.PointCloud) -> None:
        """
        Perform static visualization of point cloud.
        
        Args:
            pcd: Open3D point cloud object
        """
        o3d.visualization.draw_geometries([pcd])
    
    def _intensity_to_color(self, intensities: np.ndarray) -> np.ndarray:
        """
        Convert intensity values to RGB colors.
        
        Args:
            intensities: N array of intensity values
            
        Returns:
            Nx3 array of RGB colors
        """
        # Normalize intensities
        intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
        
        # Create colormap
        cmap = plt.get_cmap('viridis')
        colors = cmap(intensities)[:, :3]
        
        return colors
    
    def _height_to_color(self, points: np.ndarray) -> np.ndarray:
        """
        Convert height values to RGB colors.
        
        Args:
            points: Nx3 array of 3D points
            
        Returns:
            Nx3 array of RGB colors
        """
        # Get height values (z-coordinates)
        heights = points[:, 2]
        
        # Normalize heights
        heights = (heights - np.min(heights)) / (np.max(heights) - np.min(heights))
        
        # Create colormap
        cmap = plt.get_cmap('terrain')
        colors = cmap(heights)[:, :3]
        
        return colors
    
    def _default_color(self, n_points: int) -> np.ndarray:
        """
        Generate default color for points.
        
        Args:
            n_points: Number of points
            
        Returns:
            Nx3 array of RGB colors
        """
        return np.tile([0.5, 0.5, 0.5], (n_points, 1))
    
    def _save_frame(self) -> None:
        """Save current frame of visualization."""
        image = self.vis.capture_screen_float_buffer()
        plt.imsave(f"outputs/frame_{self.frame_count}.png", np.asarray(image))
        self.frame_count += 1
    
    def close(self) -> None:
        """Close visualization window."""
        if self.live_view:
            self.vis.destroy_window() 