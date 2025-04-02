# LiDAR System Simulation

A Python-based LiDAR system simulator that generates point clouds from 3D meshes, adds realistic noise, and performs clustering analysis.

## Features

- 3D mesh loading and processing
- LiDAR scan simulation with configurable parameters
- Realistic noise modeling
- Point cloud clustering and visualization
- Export to PLY format

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/lidar-system-simulation.git
cd lidar-system-simulation
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your STL file in the `scene` directory with the name `urban_scene.stl`
2. Run the simulation:
```bash
python main.py
```

The script will:
- Load your 3D scene
- Simulate a LiDAR scan
- Add noise to the point cloud
- Save the results in the `outputs` directory
- Show a visualization of the clustered point cloud

## Project Structure

```
lidar-system-simulation/
├── scene/                  # Place your STL files here
├── outputs/               # Generated point clouds
├── lidar/                 # Core LiDAR simulation modules
│   ├── scene_loader.py    # 3D mesh loading
│   ├── lidar_simulator.py # LiDAR scan simulation
│   ├── noise_model.py     # Noise modeling
│   ├── exporter.py        # Point cloud export
│   └── clustering.py      # Point cloud clustering
├── main.py                # Main script
└── requirements.txt       # Python dependencies
```

## License

MIT License # lidar-system-simulation
