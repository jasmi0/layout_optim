"""
Sample datasets for testing the layout optimization system.
"""

import json
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_schema import (
    FacilitySpecification,
    LayoutElement,
    ElementType,
    SafetyRegulation,
    WorkflowRequirements,
    LayoutConfiguration
)


def create_small_office_dataset():
    """Create a small office layout configuration"""
    facility = FacilitySpecification(
        length=15.0,
        width=10.0,
        height=3.0,
        name="Small Office"
    )
    
    elements = [
        LayoutElement("desk_1", ElementType.DESK, 1.6, 0.8, 0.75),
        LayoutElement("desk_2", ElementType.DESK, 1.6, 0.8, 0.75),
        LayoutElement("desk_3", ElementType.DESK, 1.6, 0.8, 0.75),
        LayoutElement("desk_4", ElementType.DESK, 1.6, 0.8, 0.75),
        LayoutElement("meeting_room", ElementType.MEETING_ROOM, 4.0, 3.0, 2.5, min_clearance=1.0),
        LayoutElement("storage", ElementType.STORAGE, 2.0, 1.5, 2.0),
    ]
    
    safety = SafetyRegulation(
        min_aisle_width=1.2,
        emergency_exit_clearance=2.0,
        min_exit_count=2
    )
    
    workflow = WorkflowRequirements()
    workflow.add_connection("desk_1", "storage", frequency=3.0, priority=2)
    workflow.add_connection("desk_2", "storage", frequency=2.0, priority=2)
    workflow.add_connection("desk_3", "meeting_room", frequency=1.0, priority=1)
    
    return LayoutConfiguration(facility, elements, safety, workflow)


def create_medium_office_dataset():
    """Create a medium office layout configuration"""
    facility = FacilitySpecification(
        length=25.0,
        width=18.0,
        height=3.5,
        name="Medium Office"
    )
    
    elements = []
    
    # Add 12 desks
    for i in range(12):
        elements.append(
            LayoutElement(f"desk_{i+1}", ElementType.DESK, 1.6, 0.8, 0.75)
        )
    
    # Add workstations
    elements.extend([
        LayoutElement("workstation_1", ElementType.WORKSTATION, 2.0, 1.0, 1.0),
        LayoutElement("workstation_2", ElementType.WORKSTATION, 2.0, 1.0, 1.0),
        LayoutElement("meeting_room_1", ElementType.MEETING_ROOM, 5.0, 4.0, 2.5, min_clearance=1.0),
        LayoutElement("meeting_room_2", ElementType.MEETING_ROOM, 3.0, 3.0, 2.5, min_clearance=1.0),
        LayoutElement("storage_1", ElementType.STORAGE, 3.0, 2.0, 2.5),
        LayoutElement("storage_2", ElementType.STORAGE, 2.0, 1.5, 2.0),
    ])
    
    safety = SafetyRegulation(
        min_aisle_width=1.5,
        emergency_exit_clearance=2.5,
        min_exit_count=3
    )
    
    workflow = WorkflowRequirements()
    # Add workflow connections
    workflow.add_connection("desk_1", "workstation_1", frequency=5.0, priority=4)
    workflow.add_connection("desk_2", "workstation_1", frequency=4.0, priority=3)
    workflow.add_connection("workstation_1", "storage_1", frequency=2.0, priority=5)
    workflow.add_connection("desk_5", "meeting_room_1", frequency=1.5, priority=2)
    
    return LayoutConfiguration(facility, elements, safety, workflow)


def create_warehouse_dataset():
    """Create a warehouse layout configuration"""
    facility = FacilitySpecification(
        length=40.0,
        width=30.0,
        height=6.0,
        name="Warehouse"
    )
    
    elements = []
    
    # Add storage units
    for i in range(15):
        elements.append(
            LayoutElement(
                f"storage_{i+1}",
                ElementType.STORAGE,
                4.0, 2.5, 3.0,
                weight=1000.0
            )
        )
    
    # Add machinery
    elements.extend([
        LayoutElement("forklift_area_1", ElementType.EQUIPMENT, 3.0, 3.0, 1.0, is_movable=False),
        LayoutElement("forklift_area_2", ElementType.EQUIPMENT, 3.0, 3.0, 1.0, is_movable=False),
        LayoutElement("loading_dock", ElementType.EQUIPMENT, 8.0, 4.0, 3.0, is_movable=False, min_clearance=2.0),
        LayoutElement("workstation", ElementType.WORKSTATION, 3.0, 2.0, 1.5),
    ])
    
    safety = SafetyRegulation(
        min_aisle_width=2.5,
        emergency_exit_clearance=3.0,
        min_exit_count=4,
        fire_safety_zone_width=2.0
    )
    
    workflow = WorkflowRequirements()
    # Workflow from loading dock to storage
    for i in range(1, 6):
        workflow.add_connection("loading_dock", f"storage_{i}", frequency=3.0, priority=5)
    
    return LayoutConfiguration(facility, elements, safety, workflow)


def save_dataset(config: LayoutConfiguration, filename: str):
    """Save a layout configuration to JSON file"""
    data = {
        'facility': {
            'name': config.facility.name,
            'length': config.facility.length,
            'width': config.facility.width,
            'height': config.facility.height
        },
        'elements': [
            {
                'element_id': e.element_id,
                'element_type': e.element_type.value,
                'length': e.length,
                'width': e.width,
                'height': e.height,
                'weight': e.weight,
                'is_movable': e.is_movable,
                'min_clearance': e.min_clearance,
                'x': e.x,
                'y': e.y,
                'rotation': e.rotation
            }
            for e in config.elements
        ],
        'safety': {
            'min_aisle_width': config.safety_regulations.min_aisle_width,
            'emergency_exit_clearance': config.safety_regulations.emergency_exit_clearance,
            'min_exit_count': config.safety_regulations.min_exit_count,
            'max_travel_distance_to_exit': config.safety_regulations.max_travel_distance_to_exit
        },
        'workflow': {
            'connections': [
                {
                    'from': c.from_element_id,
                    'to': c.to_element_id,
                    'frequency': c.frequency,
                    'priority': c.priority
                }
                for c in config.workflow_requirements.connections
            ]
        }
    }
    
    os.makedirs('data', exist_ok=True)
    with open(os.path.join('data', filename), 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Saved dataset to data/{filename}")


if __name__ == "__main__":
    print("Generating sample datasets...\n")
    
    # Create and save datasets
    small_office = create_small_office_dataset()
    save_dataset(small_office, "small_office.json")
    
    medium_office = create_medium_office_dataset()
    save_dataset(medium_office, "medium_office.json")
    
    warehouse = create_warehouse_dataset()
    save_dataset(warehouse, "warehouse.json")
    
    print("\n✅ All sample datasets created successfully!")
    print("\nDatasets:")
    print("  - data/small_office.json")
    print("  - data/medium_office.json")
    print("  - data/warehouse.json")
