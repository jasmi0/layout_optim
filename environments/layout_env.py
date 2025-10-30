import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_schema import LayoutConfiguration, LayoutElement
from utils.constraints import ConstraintChecker, calculate_space_utilization
from utils.geometry import calculate_centroid_distance, create_occupancy_grid


class LayoutOptimizationEnv(gym.Env):
    """
    Custom Gymnasium Environment for layout optimization.
    
    State Space:
        - Positions (x, y) of all elements
        - Rotations of all elements
        - Occupancy grid representation
        - Constraint violation metrics
    
    Action Space:
        - Continuous: [element_index, dx, dy, d_rotation]
        - element_index: which element to move (normalized 0-1)
        - dx, dy: position change (normalized -1 to 1)
        - d_rotation: rotation change in degrees (normalized -1 to 1, maps to -45 to 45)
    
    Reward:
        Multi-objective reward based on:
        - Space utilization
        - Safety compliance
        - Workflow efficiency
        - Accessibility
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, 
                 config: LayoutConfiguration,
                 max_steps: int = 500,
                 reward_weights: Dict[str, float] = None):
        """
        Initialize the environment
        
        Args:
            config: Layout configuration
            max_steps: Maximum steps per episode
            reward_weights: Weights for reward components
        """
        super(LayoutOptimizationEnv, self).__init__()
        
        self.config = config
        self.facility = config.facility
        self.elements = config.elements
        self.safety_regulations = config.safety_regulations
        self.workflow_requirements = config.workflow_requirements
        
        self.max_steps = max_steps
        self.current_step = 0
        
        # Reward weights
        self.reward_weights = reward_weights or {
            'space_utilization': 0.3,
            'safety': 0.4,
            'workflow': 0.2,
            'accessibility': 0.1
        }
        
        # Constraint checker
        self.constraint_checker = ConstraintChecker(self.facility, self.safety_regulations)
        
        # Define action and observation spaces
        self.n_elements = len(self.elements)
        
        # Action: [element_index (0-1), dx (-1,1), dy (-1,1), d_rotation (-1,1)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Calculate observation size
        # Grid resolution for occupancy grid
        self.grid_resolution = 10
        grid_size = self.grid_resolution * self.grid_resolution
        
        # Observation: positions, rotations, and additional metrics
        # Each element: [x, y, rotation] + occupancy grid + metrics
        obs_size = self.n_elements * 3 + grid_size + 10  # grid + 10 metrics
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        # Movement parameters
        self.max_position_change = 1.0  # meters per step
        self.max_rotation_change = 45.0  # degrees per step
        
        # Initialize state
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        
        # Random initial positions for movable elements
        for elem in self.elements:
            if elem.is_movable:
                elem.x = np.random.uniform(elem.length/2, self.facility.length - elem.length/2)
                elem.y = np.random.uniform(elem.width/2, self.facility.width - elem.width/2)
                elem.rotation = np.random.uniform(0, 360)
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Action to take [element_idx, dx, dy, d_rotation]
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1
        
        # Check if we have elements
        if self.n_elements == 0 or len(self.elements) == 0:
            # No elements to move, return neutral reward
            observation = self._get_observation()
            info = self._get_info()
            return observation, 0.0, False, self.current_step >= self.max_steps, info
        
        # Parse action
        # Convert action[0] from [-1, 1] to [0, n_elements-1]
        # Use floor to ensure we never exceed n_elements-1
        normalized_idx = (action[0] + 1.0) / 2.0  # Convert to [0, 1]
        element_idx = int(np.floor(normalized_idx * self.n_elements))
        # Extra safety: ensure index is valid
        element_idx = max(0, min(element_idx, len(self.elements) - 1))
        
        dx = action[1] * self.max_position_change
        dy = action[2] * self.max_position_change
        d_rotation = action[3] * self.max_rotation_change
        
        # Apply action to element
        element = self.elements[element_idx]
        if element.is_movable:
            element.x = np.clip(element.x + dx, 0, self.facility.length)
            element.y = np.clip(element.y + dy, 0, self.facility.width)
            element.rotation = (element.rotation + d_rotation) % 360
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination conditions
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # Get new observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation
        
        Returns:
            Flattened observation vector
        """
        obs_parts = []
        
        # Element positions and rotations (normalized)
        for elem in self.elements:
            obs_parts.extend([
                elem.x / self.facility.length,
                elem.y / self.facility.width,
                elem.rotation / 360.0
            ])
        
        # Occupancy grid (use grid_resolution x grid_resolution)
        corners_list = [elem.get_corners() for elem in self.elements]
        grid = create_occupancy_grid(
            self.facility.length, 
            self.facility.width,
            corners_list,
            resolution=max(self.facility.length, self.facility.width) / self.grid_resolution
        )
        # Ensure grid is exactly grid_resolution x grid_resolution
        grid_flat = grid.flatten()
        # Pad or truncate to exact size if needed
        expected_size = self.grid_resolution * self.grid_resolution
        if len(grid_flat) < expected_size:
            grid_flat = np.pad(grid_flat, (0, expected_size - len(grid_flat)))
        elif len(grid_flat) > expected_size:
            grid_flat = grid_flat[:expected_size]
        obs_parts.extend(grid_flat.tolist())
        
        # Additional metrics
        validation = self.constraint_checker.validate_layout(self.elements)
        space_util = calculate_space_utilization(self.elements, self.facility)
        
        metrics = [
            validation['collision_count'] / max(1, self.n_elements),
            validation['boundary_violations'] / max(1, self.n_elements),
            validation['clearance_violations'] / max(1, self.n_elements),
            float(validation['aisle_width_ok']),
            validation['safety_score'],
            space_util / 100.0,
            self._calculate_workflow_efficiency(),
            0.0,  # placeholder
            0.0,  # placeholder
            0.0   # placeholder
        ]
        obs_parts.extend(metrics)
        
        return np.array(obs_parts, dtype=np.float32)
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on multiple objectives
        
        Returns:
            Scalar reward value
        """
        # Validate layout
        validation = self.constraint_checker.validate_layout(self.elements)
        
        # Space utilization component (0-1)
        space_util = calculate_space_utilization(self.elements, self.facility) / 100.0
        space_util_reward = np.clip(space_util, 0, 1)
        
        # Safety compliance component (0-1)
        safety_reward = validation['safety_score']
        
        # Penalties for violations
        collision_penalty = validation['collision_count'] * 0.1
        boundary_penalty = validation['boundary_violations'] * 0.1
        clearance_penalty = validation['clearance_violations'] * 0.05
        
        safety_reward -= (collision_penalty + boundary_penalty + clearance_penalty)
        safety_reward = np.clip(safety_reward, 0, 1)
        
        # Workflow efficiency component (0-1)
        workflow_reward = self._calculate_workflow_efficiency()
        
        # Accessibility component (simplified)
        accessibility_reward = 1.0 if validation['aisle_width_ok'] else 0.5
        
        # Combined reward
        reward = (
            self.reward_weights['space_utilization'] * space_util_reward +
            self.reward_weights['safety'] * safety_reward +
            self.reward_weights['workflow'] * workflow_reward +
            self.reward_weights['accessibility'] * accessibility_reward
        )
        
        return reward
    
    def _calculate_workflow_efficiency(self) -> float:
        """
        Calculate workflow efficiency based on element distances
        
        Returns:
            Efficiency score (0-1, higher is better)
        """
        if not self.workflow_requirements.connections:
            return 0.5  # neutral if no workflow defined
        
        total_weighted_distance = 0.0
        total_weight = 0.0
        max_distance = np.sqrt(self.facility.length**2 + self.facility.width**2)
        
        # Build element lookup
        elem_dict = {e.element_id: e for e in self.elements}
        
        for conn in self.workflow_requirements.connections:
            if conn.from_element_id in elem_dict and conn.to_element_id in elem_dict:
                elem1 = elem_dict[conn.from_element_id]
                elem2 = elem_dict[conn.to_element_id]
                
                corners1 = elem1.get_corners()
                corners2 = elem2.get_corners()
                distance = calculate_centroid_distance(corners1, corners2)
                
                # Normalize distance
                normalized_distance = distance / max_distance
                
                # Weight by frequency and priority
                weight = conn.frequency * conn.priority
                total_weighted_distance += normalized_distance * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        # Average weighted distance (lower is better)
        avg_distance = total_weighted_distance / total_weight
        
        # Convert to reward (invert so closer is better)
        efficiency = 1.0 - avg_distance
        
        return np.clip(efficiency, 0, 1)
    
    def _get_info(self) -> Dict:
        """Get additional information about current state"""
        validation = self.constraint_checker.validate_layout(self.elements)
        
        return {
            'step': self.current_step,
            'valid_layout': validation['valid'],
            'collision_count': validation['collision_count'],
            'boundary_violations': validation['boundary_violations'],
            'clearance_violations': validation['clearance_violations'],
            'safety_score': validation['safety_score'],
            'space_utilization': calculate_space_utilization(self.elements, self.facility),
            'workflow_efficiency': self._calculate_workflow_efficiency()
        }
    
    def render(self, mode='human'):
        """Render the environment (optional)"""
        if mode == 'human':
            print(f"\n=== Step {self.current_step} ===")
            info = self._get_info()
            print(f"Valid Layout: {info['valid_layout']}")
            print(f"Space Utilization: {info['space_utilization']:.2f}%")
            print(f"Safety Score: {info['safety_score']:.2f}")
            print(f"Workflow Efficiency: {info['workflow_efficiency']:.2f}")
            print(f"Collisions: {info['collision_count']}")


if __name__ == "__main__":
    # Test the environment
    from utils.data_schema import create_sample_layout
    
    print("Creating sample layout configuration...")
    config = create_sample_layout()
    
    print("Initializing environment...")
    env = LayoutOptimizationEnv(config)
    
    print("Testing random actions...")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(f"Reward: {reward:.4f}")
        
        if terminated or truncated:
            break
    
    print("\nEnvironment test completed!")
