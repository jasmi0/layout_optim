"""
Data schema and validation for layout optimization system.
Defines data models for facilities, layout elements, safety regulations, and workflows.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import numpy as np


class ElementType(Enum):
    """Types of layout elements"""
    DESK = "desk"
    MACHINERY = "machinery"
    WORKSTATION = "workstation"
    STORAGE = "storage"
    AISLE = "aisle"
    EMERGENCY_EXIT = "emergency_exit"
    MEETING_ROOM = "meeting_room"
    EQUIPMENT = "equipment"


@dataclass
class FacilitySpecification:
    """Facility dimensions and constraints"""
    length: float  # meters
    width: float  # meters
    height: float  # meters
    total_area: float = field(init=False)
    name: str = "Unnamed Facility"
    
    def __post_init__(self):
        self.total_area = self.length * self.width
        self.validate()
    
    def validate(self):
        """Validate facility specifications"""
        if self.length <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("All dimensions must be positive")
        if self.length > 1000 or self.width > 1000:
            raise ValueError("Facility dimensions exceed maximum (1000m)")


@dataclass
class LayoutElement:
    """Individual layout element (desk, machinery, etc.)"""
    element_id: str
    element_type: ElementType
    length: float  # meters
    width: float  # meters
    height: float  # meters
    weight: float = 0.0  # kg
    is_movable: bool = True
    min_clearance: float = 0.5  # minimum clearance in meters
    
    # Position and orientation (to be set during optimization)
    x: float = 0.0
    y: float = 0.0
    rotation: float = 0.0  # degrees
    
    def __post_init__(self):
        self.validate()
    
    def validate(self):
        """Validate element specifications"""
        if self.length <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError(f"Element {self.element_id}: All dimensions must be positive")
        if self.min_clearance < 0:
            raise ValueError(f"Element {self.element_id}: Clearance cannot be negative")
        if self.rotation < 0 or self.rotation >= 360:
            raise ValueError(f"Element {self.element_id}: Rotation must be [0, 360)")
    
    def get_footprint_area(self) -> float:
        """Calculate the footprint area of the element"""
        return self.length * self.width
    
    def get_corners(self) -> np.ndarray:
        """Get the four corners of the element based on position and rotation"""
        # Original corners (centered at origin)
        corners = np.array([
            [-self.length/2, -self.width/2],
            [self.length/2, -self.width/2],
            [self.length/2, self.width/2],
            [-self.length/2, self.width/2]
        ])
        
        # Rotate
        theta = np.radians(self.rotation)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        rotated_corners = corners @ rotation_matrix.T
        
        # Translate to position
        translated_corners = rotated_corners + np.array([self.x, self.y])
        
        return translated_corners


@dataclass
class SafetyRegulation:
    """Safety regulations and constraints"""
    min_aisle_width: float = 1.2  # meters
    emergency_exit_clearance: float = 2.0  # meters
    max_occupancy_per_sqm: float = 0.1  # persons per square meter
    fire_safety_zone_width: float = 1.5  # meters
    min_exit_count: int = 2
    max_travel_distance_to_exit: float = 30.0  # meters
    
    def validate(self):
        """Validate safety regulations"""
        if self.min_aisle_width < 0.8:
            raise ValueError("Minimum aisle width must be at least 0.8m")
        if self.emergency_exit_clearance < 1.0:
            raise ValueError("Emergency exit clearance must be at least 1.0m")
        if self.min_exit_count < 1:
            raise ValueError("At least one exit is required")


@dataclass
class WorkflowConnection:
    """Connection between two elements representing workflow"""
    from_element_id: str
    to_element_id: str
    frequency: float = 1.0  # trips per hour
    priority: int = 1  # 1 (low) to 5 (high)
    
    def validate(self):
        """Validate workflow connection"""
        if self.frequency < 0:
            raise ValueError("Frequency cannot be negative")
        if self.priority < 1 or self.priority > 5:
            raise ValueError("Priority must be between 1 and 5")


@dataclass
class WorkflowRequirements:
    """Workflow requirements for the layout"""
    connections: List[WorkflowConnection] = field(default_factory=list)
    adjacency_preferences: Dict[str, List[str]] = field(default_factory=dict)
    
    def add_connection(self, from_id: str, to_id: str, frequency: float = 1.0, priority: int = 1):
        """Add a workflow connection"""
        connection = WorkflowConnection(from_id, to_id, frequency, priority)
        connection.validate()
        self.connections.append(connection)
    
    def add_adjacency_preference(self, element_id: str, preferred_neighbors: List[str]):
        """Add adjacency preference for an element"""
        self.adjacency_preferences[element_id] = preferred_neighbors
    
    def get_connection_matrix(self, element_ids: List[str]) -> np.ndarray:
        """Get workflow connection matrix"""
        n = len(element_ids)
        matrix = np.zeros((n, n))
        
        id_to_idx = {eid: i for i, eid in enumerate(element_ids)}
        
        for conn in self.connections:
            if conn.from_element_id in id_to_idx and conn.to_element_id in id_to_idx:
                i = id_to_idx[conn.from_element_id]
                j = id_to_idx[conn.to_element_id]
                matrix[i, j] = conn.frequency * conn.priority
        
        return matrix


@dataclass
class LayoutConfiguration:
    """Complete layout configuration"""
    facility: FacilitySpecification
    elements: List[LayoutElement]
    safety_regulations: SafetyRegulation
    workflow_requirements: WorkflowRequirements
    
    def validate(self):
        """Validate complete layout configuration"""
        self.facility.validate()
        self.safety_regulations.validate()
        
        # Check for duplicate element IDs
        element_ids = [e.element_id for e in self.elements]
        if len(element_ids) != len(set(element_ids)):
            raise ValueError("Duplicate element IDs found")
        
        # Validate each element
        for element in self.elements:
            element.validate()
        
        # Validate workflow connections reference existing elements
        for conn in self.workflow_requirements.connections:
            if conn.from_element_id not in element_ids:
                raise ValueError(f"Workflow connection references unknown element: {conn.from_element_id}")
            if conn.to_element_id not in element_ids:
                raise ValueError(f"Workflow connection references unknown element: {conn.to_element_id}")
    
    def get_total_element_area(self) -> float:
        """Calculate total area of all elements"""
        return sum(e.get_footprint_area() for e in self.elements)
    
    def get_space_utilization(self) -> float:
        """Calculate current space utilization percentage"""
        return (self.get_total_element_area() / self.facility.total_area) * 100


def create_sample_layout() -> LayoutConfiguration:
    """Create a sample layout configuration for testing"""
    # Create facility
    facility = FacilitySpecification(length=20.0, width=15.0, height=3.0, name="Sample Office")
    
    # Create elements
    elements = [
        LayoutElement("desk_1", ElementType.DESK, 1.6, 0.8, 0.75),
        LayoutElement("desk_2", ElementType.DESK, 1.6, 0.8, 0.75),
        LayoutElement("desk_3", ElementType.DESK, 1.6, 0.8, 0.75),
        LayoutElement("desk_4", ElementType.DESK, 1.6, 0.8, 0.75),
        LayoutElement("meeting_1", ElementType.MEETING_ROOM, 4.0, 3.0, 2.5, min_clearance=1.0),
        LayoutElement("storage_1", ElementType.STORAGE, 2.0, 1.5, 2.0),
        LayoutElement("workstation_1", ElementType.WORKSTATION, 2.0, 1.0, 1.0),
        LayoutElement("workstation_2", ElementType.WORKSTATION, 2.0, 1.0, 1.0),
    ]
    
    # Safety regulations
    safety = SafetyRegulation(
        min_aisle_width=1.2,
        emergency_exit_clearance=2.0,
        min_exit_count=2
    )
    
    # Workflow requirements
    workflow = WorkflowRequirements()
    workflow.add_connection("desk_1", "workstation_1", frequency=5.0, priority=3)
    workflow.add_connection("desk_2", "workstation_1", frequency=3.0, priority=2)
    workflow.add_connection("workstation_1", "storage_1", frequency=2.0, priority=4)
    workflow.add_adjacency_preference("desk_1", ["desk_2", "desk_3"])
    
    # Create configuration
    config = LayoutConfiguration(facility, elements, safety, workflow)
    config.validate()
    
    return config


if __name__ == "__main__":
    # Test the data schema
    config = create_sample_layout()
    print(f"Facility: {config.facility.name}")
    print(f"Dimensions: {config.facility.length}m x {config.facility.width}m")
    print(f"Total area: {config.facility.total_area} sqm")
    print(f"Number of elements: {len(config.elements)}")
    print(f"Total element area: {config.get_total_element_area():.2f} sqm")
    print(f"Space utilization: {config.get_space_utilization():.2f}%")
    print(f"Workflow connections: {len(config.workflow_requirements.connections)}")
