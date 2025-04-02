"""
Environmental effects model for LiDAR simulation.
Handles weather conditions like rain, snow, and fog.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class WeatherParams:
    """Weather condition parameters."""
    condition: str  # clear, rain, snow, fog
    rain_intensity: float  # mm/h
    snow_intensity: float  # mm/h
    fog_density: float  # g/m³
    temperature: float  # °C
    humidity: float  # %

class EnvironmentModel:
    """Model for environmental effects on LiDAR performance."""
    
    def __init__(self, params: WeatherParams):
        """
        Initialize the environment model with weather parameters.
        
        Args:
            params: WeatherParams object containing weather conditions
        """
        self.params = params
        self._setup_weather_effects()
    
    def _setup_weather_effects(self) -> None:
        """Setup weather-specific parameters and models."""
        if self.params.condition == "rain":
            self._setup_rain_model()
        elif self.params.condition == "snow":
            self._setup_snow_model()
        elif self.params.condition == "fog":
            self._setup_fog_model()
    
    def _setup_rain_model(self) -> None:
        """Setup rain-specific attenuation model."""
        # Rain drop size distribution (Marshall-Palmer)
        self.rain_drop_diameter = 1.0  # mm
        self.rain_drop_density = self.params.rain_intensity * 1000  # drops/m³
        
        # Rain attenuation coefficient (dB/km)
        self.rain_attenuation = 1.076 * (self.params.rain_intensity ** 0.67)
    
    def _setup_snow_model(self) -> None:
        """Setup snow-specific attenuation model."""
        # Snowflake size distribution
        self.snow_diameter = 2.0  # mm
        self.snow_density = self.params.snow_intensity * 1000  # flakes/m³
        
        # Snow attenuation coefficient (dB/km)
        self.snow_attenuation = 0.5 * (self.params.snow_intensity ** 0.7)
    
    def _setup_fog_model(self) -> None:
        """Setup fog-specific attenuation model."""
        # Fog droplet size distribution
        self.fog_droplet_diameter = 0.01  # mm
        self.fog_droplet_density = self.params.fog_density * 1e6  # droplets/m³
        
        # Fog attenuation coefficient (dB/km)
        self.fog_attenuation = 3.912 / (self.params.fog_density + 0.01)
    
    def apply_attenuation(self, points: np.ndarray, intensities: np.ndarray) -> np.ndarray:
        """
        Apply weather-based attenuation to point cloud intensities.
        
        Args:
            points: Nx3 array of 3D points
            intensities: N array of intensity values
            
        Returns:
            Attenuated intensity values
        """
        distances = np.linalg.norm(points, axis=1)
        
        if self.params.condition == "rain":
            attenuation = self.rain_attenuation
        elif self.params.condition == "snow":
            attenuation = self.snow_attenuation
        elif self.params.condition == "fog":
            attenuation = self.fog_attenuation
        else:
            return intensities
        
        # Apply exponential attenuation
        attenuated = intensities * np.exp(-attenuation * distances / 1000)
        return np.clip(attenuated, 0, 1)
    
    def add_weather_noise(self, points: np.ndarray) -> np.ndarray:
        """
        Add weather-dependent noise to point cloud.
        
        Args:
            points: Nx3 array of 3D points
            
        Returns:
            Noisy point cloud
        """
        if self.params.condition == "clear":
            return points
        
        # Weather-dependent noise standard deviation
        if self.params.condition == "rain":
            noise_std = 0.02 * (1 + self.params.rain_intensity / 10)
        elif self.params.condition == "snow":
            noise_std = 0.03 * (1 + self.params.snow_intensity / 5)
        else:  # fog
            noise_std = 0.01 * (1 + self.params.fog_density / 0.1)
        
        noise = np.random.normal(scale=noise_std, size=points.shape)
        return points + noise
    
    def get_visibility(self) -> float:
        """
        Calculate visibility range based on weather conditions.
        
        Returns:
            Visibility range in meters
        """
        if self.params.condition == "clear":
            return float('inf')
        elif self.params.condition == "rain":
            return 3000 / (self.params.rain_intensity + 1)
        elif self.params.condition == "snow":
            return 2000 / (self.params.snow_intensity + 1)
        else:  # fog
            return 3000 / (self.params.fog_density + 0.01) 