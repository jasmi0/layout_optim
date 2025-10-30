"""
Constraint checking utilities for safety regulations and layout validation.
"""

import numpy as np
from typing import List, Dict, Tuple
from .geometry import (
    check_collision,
    calculate_clearance,
    calculate_min_distance,
    check_boundary_violation,
    get_bounding_box
)
from .data_schema import LayoutElement, SafetyRegulation, FacilitySpecification


class ConstraintChecker:
    """Check various constraints for layout optimization"""
    
    def __init__(self, facility: FacilitySpecification, safety: SafetyRegulation):
        self.facility = facility
        self.safety = safety
        self.boundary_min = np.array([0.0, 0.0])
        self.boundary_max = np.array([facility.length, facility.width])
    
    def check_collisions(self, elements: List[LayoutElement]) -> Tuple[int, List[Tuple[str, str]]]:
        """
        Check for collisions between elements
        
        Returns:
            Tuple of (collision_count, list of colliding pairs)
        """
        collisions = []
        
        for i, elem1 in enumerate(elements):
            corners1 = elem1.get_corners()
            
            for elem2 in elements[i+1:]:
                corners2 = elem2.get_corners()
                
                if check_collision(corners1, corners2):
                    collisions.append((elem1.element_id, elem2.element_id))
        
        return len(collisions), collisions
    
    def check_boundary_constraints(self, elements: List[LayoutElement]) -> Tuple[int, List[str]]:
        """
        Check if any element violates facility boundaries
        
        Returns:
            Tuple of (violation_count, list of violating element IDs)
        """
        violations = []
        
        for elem in elements:
            corners = elem.get_corners()
            if check_boundary_violation(corners, self.boundary_min, self.boundary_max):
                violations.append(elem.element_id)
        
        return len(violations), violations
    
    def check_clearance_requirements(self, elements: List[LayoutElement]) -> Tuple[int, List[Dict]]:
        """
        Check if minimum clearance requirements are met
        
        Returns:
            Tuple of (violation_count, list of violation details)
        """
        violations = []
        
        for i, elem1 in enumerate(elements):
            corners1 = elem1.get_corners()
            required_clearance = elem1.min_clearance
            
            for elem2 in elements[i+1:]:
                corners2 = elem2.get_corners()
                clearance = calculate_clearance(corners1, corners2)
                
                if 0 <= clearance < required_clearance:
                    violations.append({
                        'element1': elem1.element_id,
                        'element2': elem2.element_id,
                        'required': required_clearance,
                        'actual': clearance
                    })
        
        return len(violations), violations
    
    def check_aisle_widths(self, elements: List[LayoutElement]) -> bool:
        """
        Check if minimum aisle widths are maintained
        
        This is a simplified check - a full implementation would require
        path analysis throughout the facility.
        """
        # Simplified: check if there's enough free space
        occupied_area = sum(e.get_footprint_area() for e in elements)
        free_area = self.facility.total_area - occupied_area
        
        # Rough estimate: free area should be at least 20% for aisles
        return free_area >= self.facility.total_area * 0.2
    
    def check_emergency_exits(self, elements: List[LayoutElement], 
                            exit_positions: List[np.ndarray]) -> Tuple[bool, List[str]]:
        """
        Check if emergency exits have required clearance
        
        Args:
            elements: List of layout elements
            exit_positions: List of exit coordinates
        
        Returns:
            Tuple of (all_clear, list of blocked exits)
        """
        blocked_exits = []
        
        for i, exit_pos in enumerate(exit_positions):
            for elem in elements:
                corners = elem.get_corners()
                centroid = np.mean(corners, axis=0)
                distance = np.linalg.norm(centroid - exit_pos)
                
                if distance < self.safety.emergency_exit_clearance:
                    blocked_exits.append(f"Exit_{i}")
                    break
        
        return len(blocked_exits) == 0, blocked_exits
    
    def validate_layout(self, elements: List[LayoutElement]) -> Dict:
        """
        Comprehensive layout validation
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'collision_count': 0,
            'collision_pairs': [],
            'boundary_violations': 0,
            'boundary_violating_elements': [],
            'clearance_violations': 0,
            'clearance_details': [],
            'aisle_width_ok': True,
            'safety_score': 1.0
        }
        
        # Check collisions
        collision_count, collision_pairs = self.check_collisions(elements)
        results['collision_count'] = collision_count
        results['collision_pairs'] = collision_pairs
        
        # Check boundaries
        boundary_count, boundary_elements = self.check_boundary_constraints(elements)
        results['boundary_violations'] = boundary_count
        results['boundary_violating_elements'] = boundary_elements
        
        # Check clearances
        clearance_count, clearance_details = self.check_clearance_requirements(elements)
        results['clearance_violations'] = clearance_count
        results['clearance_details'] = clearance_details
        
        # Check aisle widths
        results['aisle_width_ok'] = self.check_aisle_widths(elements)
        
        # Determine overall validity
        results['valid'] = (
            collision_count == 0 and
            boundary_count == 0 and
            clearance_count == 0 and
            results['aisle_width_ok']
        )
        
        # Calculate safety score (0-1)
        total_violations = collision_count + boundary_count + clearance_count
        if not results['aisle_width_ok']:
            total_violations += 1
        
        results['safety_score'] = max(0.0, 1.0 - (total_violations * 0.1))
        
        return results


def calculate_space_utilization(elements: List[LayoutElement], 
                               facility: FacilitySpecification) -> float:
    """
    Calculate space utilization percentage
    
    Args:
        elements: List of layout elements
        facility: Facility specifications
    
    Returns:
        Space utilization percentage (0-100)
    """
    total_element_area = sum(e.get_footprint_area() for e in elements)
    return (total_element_area / facility.total_area) * 100


def check_load_distribution(elements: List[LayoutElement], 
                           facility: FacilitySpecification,
                           max_load_per_sqm: float = 500.0) -> bool:
    """
    Check if weight distribution is within limits
    
    Args:
        elements: List of layout elements
        facility: Facility specifications
        max_load_per_sqm: Maximum load in kg per square meter
    
    Returns:
        True if load distribution is acceptable
    """
    total_weight = sum(e.weight for e in elements)
    avg_load = total_weight / facility.total_area
    
    return avg_load <= max_load_per_sqm


if __name__ == "__main__":
    # Test constraint checking
    from .data_schema import create_sample_layout
    
    config = create_sample_layout()
    checker = ConstraintChecker(config.facility, config.safety_regulations)
    
    # Random placement for testing
    for elem in config.elements:
        elem.x = np.random.uniform(2, config.facility.length - 2)
        elem.y = np.random.uniform(2, config.facility.width - 2)
    
    results = checker.validate_layout(config.elements)
    print("Validation Results:")
    print(f"  Valid: {results['valid']}")
    print(f"  Collisions: {results['collision_count']}")
    print(f"  Boundary violations: {results['boundary_violations']}")
    print(f"  Clearance violations: {results['clearance_violations']}")
    print(f"  Safety score: {results['safety_score']:.2f}")
