# Reinforcement Layout Optimization Using AI + 3D Modeling

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

**AI-powered facility layout optimization using Reinforcement Learning and 3D visualization**

</div>

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

This project uses **reinforcement learning** to optimize layouts for facilities or buildings, taking into account factors like efficiency, space utilization, and safety regulations. The system dynamically adjusts the layout to optimize these factors and provides an interactive platform for users to visualize the optimized designs in 3D.

### Objectives

- Optimize layouts for facilities, buildings, or industrial plants using RL
- Ensure efficient space utilization while adhering to safety regulations
- Provide real-time feedback and simulations for layout adjustments
- Visualize optimized layouts using interactive 3D modeling tools

---

## Features

### Core Capabilities

- **Flexible Facility Configuration**: Define custom facility dimensions and constraints
- **Multiple Element Types**: Support for desks, machinery, workstations, storage, and more
- **RL-Based Optimization**: Uses PPO, SAC, or A2C algorithms from Stable-Baselines3
- **Safety Compliance**: Automatic validation against safety regulations
- **Performance Metrics**: Real-time tracking of space utilization, workflow efficiency
- **Interactive UI**: User-friendly Streamlit interface
- **3D/2D Visualization**: Interactive Plotly visualizations from multiple angles
- **Save/Load Configurations**: Export and import layout configurations
- **Workflow Optimization**: Considers material flow and personnel movement

### Optimization Criteria

The RL agent optimizes layouts based on:

1. **Space Utilization** (30%): Maximize efficient use of available space
2. **Safety Compliance** (40%): Adhere to safety regulations and clearances
3. **Workflow Efficiency** (20%): Minimize material handling distances
4. **Accessibility** (10%): Ensure proper aisle widths and accessibility

---

## Technologies

### Core Stack

- **Language**: Python 3.8+
- **RL Framework**: Stable-Baselines3, Gymnasium
- **Visualization**: Plotly, Matplotlib
- **UI Framework**: Streamlit
- **Geometry**: Shapely, NumPy
- **Data Processing**: Pandas, NumPy

### Algorithms

- **PPO** (Proximal Policy Optimization) - Recommended
- **SAC** (Soft Actor-Critic) - Alternative
- **A2C** (Advantage Actor-Critic) - Faster training

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step-by-Step Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd layout_optim
```

2. **Create a virtual environment**

```bash
python -m venv venv

# On Linux/Mac
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import stable_baselines3; import streamlit; print('✓ Installation successful!')"
```

---

## Usage

### Quick Start

1. **Run the Streamlit Application**

```bash
streamlit run app.py
```

2. **Access the Web Interface**

Open your browser and navigate to `http://localhost:8501`

### Using the Application

#### 1. Configuration

- Define your facility dimensions (length, width, height)
- Set safety regulations (aisle widths, exit clearances)
- Add layout elements (desks, machinery, etc.)
- Or use pre-built templates for quick start

#### 2. Optimization

- Choose RL algorithm (PPO recommended)
- Adjust reward weights based on priorities
- Run optimization to find optimal layout
- Monitor progress in real-time

#### 3. Visualization

- View layout in interactive 3D
- Switch to 2D top-down view
- Toggle workflow connections
- Rotate, zoom, and pan

#### 4. Analysis

- Review performance metrics
- Check safety compliance
- Validate layout constraints
- Export results

### Training a Custom Model

For advanced users who want to train their own RL model:

```bash
# Train with PPO (default)
python train.py --algorithm PPO --timesteps 100000

# Train with SAC
python train.py --algorithm SAC --timesteps 50000

# Evaluate a trained model
python train.py --eval models/best_model/best_model.zip
```

**Monitor training with TensorBoard:**

```bash
tensorboard --logdir ./logs
```

### Testing the Environment

```bash
# Test the custom Gym environment
python environments/layout_env.py

# Run unit tests
python -m pytest tests/

# Test data schema validation
python utils/data_schema.py

# Test visualization components
python visualization/layout_viz.py
```

### Data Formats

The system uses JSON format for layout configurations:

```json
{
  "facility": {
    "name": "Office Layout",
    "length": 20.0,
    "width": 15.0,
    "height": 3.0
  },
  "elements": [
    {
      "element_id": "desk_1",
      "element_type": "DESK",
      "length": 1.6,
      "width": 0.8,
      "height": 0.75,
      "x": 5.0,
      "y": 3.0,
      "rotation": 0.0,
      "min_clearance": 0.5
    }
  ],
  "safety_regulations": {
    "min_aisle_width": 1.2,
    "emergency_exit_clearance": 2.0,
    "min_exit_count": 2,
    "max_travel_distance_to_exit": 30.0
  }
}
```

**Element Types Supported:**
- `DESK`: Office desks and workstations
- `MEETING_ROOM`: Conference rooms
- `STORAGE`: Storage units and cabinets
- `MACHINERY`: Industrial equipment
- `WORKSTATION`: Assembly or processing stations
- `AISLE`: Passageways (automatically generated)
- `EMERGENCY_EXIT`: Exit points
- `EQUIPMENT`: Miscellaneous equipment

---

## Project Structure

```
layout_optim/
├── app.py                      # Main Streamlit application
├── train.py                    # RL model training script
├── requirements.txt            # Python dependencies
│
├── data/                       # Data storage
│   ├── create_samples.py       # Script to generate sample datasets
│   └── (sample datasets: small_office.json, medium_office.json, warehouse.json)
│
├── models/                     # Trained models
│   ├── checkpoints/           # Training checkpoints
│   └── best_model/            # Best performing model
│
├── environments/              # RL environment
│   ├── __init__.py
│   └── layout_env.py          # Custom Gymnasium environment
│
├── utils/                     # Utility modules
│   ├── __init__.py
│   ├── data_schema.py         # Data models and validation
│   ├── geometry.py            # Geometry utilities
│   └── constraints.py         # Constraint checking
│
├── visualization/             # Visualization tools
│   ├── __init__.py
│   ├── layout_viz.py          # Plotly 3D/2D visualization
│   └── threejs_viz.py         # Three.js WebGL visualization
│
├── tests/                     # Unit tests
│   └── (test files)
│
└── logs/                      # Training logs
    └── (TensorBoard logs)
```

---

## How It Works

### System Workflow

```
1. Data Ingestion
   ↓
2. Configuration & Validation
   ↓
3. RL Environment Creation
   ↓
4. Optimization Loop
   ├─ State: Element positions, rotations, metrics
   ├─ Action: Move/rotate elements
   ├─ Reward: Multi-objective score
   └─ Update: Adjust layout
   ↓
5. Visualization & Analysis
```

### Reinforcement Learning Environment

**State Space:**
- Element positions (x, y) and rotations
- Occupancy grid representation
- Constraint violation metrics
- Workflow efficiency scores

**Action Space:**
- Continuous actions: [element_index, dx, dy, d_rotation]
- Element selection and movement
- Rotation adjustments

**Reward Function:**

```python
R = w₁ × space_util + w₂ × safety + w₃ × workflow + w₄ × accessibility
```

Where:
- `space_util`: Space utilization efficiency (0-1)
- `safety`: Safety compliance score (0-1)
- `workflow`: Workflow efficiency based on distances (0-1)
- `accessibility`: Aisle width and accessibility (0-1)

### Constraint Validation

The system validates:
- No element collisions
- All elements within facility boundaries
- Minimum clearance requirements met
- Adequate aisle widths maintained
- Emergency exit clearances preserved

### Research Methodology

This project implements a **multi-objective reinforcement learning approach** to facility layout optimization, combining:

**Algorithm Selection:**
- **PPO (Proximal Policy Optimization)**: Default choice for stable training and good performance
- **SAC (Soft Actor-Critic)**: Better for continuous control with entropy regularization
- **A2C (Advantage Actor-Critic)**: Faster training, suitable for simpler layouts

**State Representation:**
- **Continuous positions**: Normalized element coordinates (x, y)
- **Rotation angles**: Element orientations in degrees
- **Occupancy grid**: 2D spatial representation for collision detection
- **Constraint metrics**: Real-time validation scores

**Reward Engineering:**
The reward function balances competing objectives using weighted sum:

```
R = w₁×space_util + w₂×safety + w₃×workflow + w₄×accessibility
```

Where weights are user-configurable based on facility priorities.

**Constraint Handling:**
- **Hard constraints**: Enforced through penalty terms in reward
- **Soft constraints**: Incorporated as optimization objectives
- **Validation**: Real-time constraint checking during training

---

## Results & Performance

### Expected Performance

- **Space Utilization**: >80% efficient use of available space
- **Safety Compliance**: Zero violations in optimized layouts
- **Workflow Efficiency**: >30% improvement over random layouts
- **Rendering Speed**: <5 seconds for 3D visualization
- **Optimization Time**: Handles up to 50 elements efficiently

### Sample Results

```
Layout Optimization Results:
├─ Space Utilization: 78.5%
├─ Safety Score: 0.95/1.00
├─ Workflow Efficiency: 0.82/1.00
├─ Valid Layout: ✅
└─ Constraint Violations: 0
```

---

## Troubleshooting

### Common Issues

**Streamlit App Won't Start**
```bash
# Make sure you're in the virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install missing dependencies
pip install -r requirements.txt

# Run with specific port if 8501 is busy
streamlit run app.py --server.port 8502
```

**RL Training Fails**
```bash
# Check if stable-baselines3 is properly installed
python -c "import stable_baselines3; print('✓ RL framework available')"

# For GPU training (if CUDA available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Visualization Issues**
```bash
# For Three.js WebGL issues, fall back to Plotly
# The app automatically handles this, but you can force Plotly in the UI

# Install additional visualization dependencies if needed
pip install pythreejs ipywidgets
```

**Memory Issues with Large Facilities**
```bash
# Reduce grid resolution in layout_env.py
self.grid_resolution = 5  # Instead of 10

# Use smaller batch sizes in training
python train.py --batch-size 32
```

**Import Errors**
```bash
# Make sure you're running from the project root
cd /path/to/layout_optim

# Check Python path
python -c "import sys; print(sys.path)"
```

### Performance Tips

- **For faster training**: Use A2C instead of PPO for simpler environments
- **For better results**: Increase total_timesteps to 500,000+ for complex layouts
- **For visualization**: Use Plotly 3D for better performance than Three.js
- **For large facilities**: Break down into smaller zones and optimize separately

---

## Future Improvements

### Planned Features

- [ ] **Multi-agent RL** for larger facilities
- [ ] **Energy efficiency** constraints
- [ ] **Environmental impact** calculations
- [ ] **Personnel comfort** metrics
- [ ] **Cost optimization** integration
- [ ] **Real-time data** integration via APIs
- [ ] **AR/VR** support for immersive visualization
- [ ] **Collaborative editing** features
- [ ] **Export to CAD** formats (DXF, STEP)
- [ ] **Mobile app** support

### Enhancements

- Parallel environment training for faster convergence
- Curriculum learning for complex scenarios
- Transfer learning across facility types
- Integration with BIM software
- Cloud deployment options
