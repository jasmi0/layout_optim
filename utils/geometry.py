"""
Geometry utilities for collision detection, distance calculation, and spatial operations.
"""

import numpy as np
from typing import List, Tuple, Optional
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union


def calculate_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points"""
    return np.linalg.norm(point1 - point2)


def rotate_point(point: np.ndarray, angle_degrees: float, center: np.ndarray = None) -> np.ndarray:
    """
    Rotate a point around a center by given angle in degrees
    
    Args:
        point: 2D point to rotate
        angle_degrees: Rotation angle in degrees
        center: Center of rotation (default: origin)
    
    Returns:
        Rotated point
    """
    if center is None:
        center = np.array([0.0, 0.0])
    
    theta = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    translated = point - center
    rotated = rotation_matrix @ translated
    return rotated + center


def get_rectangle_corners(x: float, y: float, length: float, width: float, rotation: float = 0.0) -> np.ndarray:
    """
    Get the four corners of a rectangle given its center, dimensions, and rotation
    
    Args:
        x, y: Center coordinates
        length, width: Rectangle dimensions
        rotation: Rotation angle in degrees
    
    Returns:
        Array of shape (4, 2) with corner coordinates
    """
    # Corners relative to center
    corners = np.array([
        [-length/2, -width/2],
        [length/2, -width/2],
        [length/2, width/2],
        [-length/2, width/2]
    ])
    
    # Rotate
    theta = np.radians(rotation)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    rotated_corners = corners @ rotation_matrix.T
    
    # Translate
    return rotated_corners + np.array([x, y])


def check_collision(corners1: np.ndarray, corners2: np.ndarray) -> bool:
    """
    Check if two rectangles (defined by corners) collide using SAT algorithm
    
    Args:
        corners1: First rectangle corners (4, 2)
        corners2: Second rectangle corners (4, 2)
    
    Returns:
        True if rectangles overlap, False otherwise
    """
    poly1 = Polygon(corners1)
    poly2 = Polygon(corners2)
    
    return poly1.intersects(poly2)


def calculate_overlap_area(corners1: np.ndarray, corners2: np.ndarray) -> float:
    """
    Calculate the area of overlap between two rectangles
    
    Args:
        corners1: First rectangle corners
        corners2: Second rectangle corners
    
    Returns:
        Overlap area in square units
    """
    poly1 = Polygon(corners1)
    poly2 = Polygon(corners2)
    
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    
    intersection = poly1.intersection(poly2)
    return intersection.area


def point_in_rectangle(point: np.ndarray, corners: np.ndarray) -> bool:
    """
    Check if a point is inside a rectangle
    
    Args:
        point: 2D point coordinates
        corners: Rectangle corners (4, 2)
    
    Returns:
        True if point is inside rectangle
    """
    poly = Polygon(corners)
    pt = Point(point)
    return poly.contains(pt)


def check_boundary_violation(corners: np.ndarray, boundary_min: np.ndarray, boundary_max: np.ndarray) -> bool:
    """
    Check if any corner of a rectangle is outside the boundary
    
    Args:
        corners: Rectangle corners (4, 2)
        boundary_min: Minimum boundary coordinates [x_min, y_min]
        boundary_max: Maximum boundary coordinates [x_max, y_max]
    
    Returns:
        True if any corner is outside the boundary
    """
    return np.any(corners < boundary_min) or np.any(corners > boundary_max)


def calculate_centroid_distance(corners1: np.ndarray, corners2: np.ndarray) -> float:
    """
    Calculate distance between centroids of two rectangles
    
    Args:
        corners1: First rectangle corners
        corners2: Second rectangle corners
    
    Returns:
        Distance between centroids
    """
    centroid1 = np.mean(corners1, axis=0)
    centroid2 = np.mean(corners2, axis=0)
    return calculate_distance(centroid1, centroid2)


def calculate_min_distance(corners1: np.ndarray, corners2: np.ndarray) -> float:
    """
    Calculate minimum distance between two rectangles
    
    Args:
        corners1: First rectangle corners
        corners2: Second rectangle corners
    
    Returns:
        Minimum distance (0 if overlapping)
    """
    poly1 = Polygon(corners1)
    poly2 = Polygon(corners2)
    
    return poly1.distance(poly2)


def calculate_clearance(corners1: np.ndarray, corners2: np.ndarray) -> float:
    """
    Calculate clearance (gap) between two rectangles
    
    Args:
        corners1: First rectangle corners
        corners2: Second rectangle corners
    
    Returns:
        Clearance distance (negative if overlapping)
    """
    if check_collision(corners1, corners2):
        # Overlapping - return negative overlap area
        return -calculate_overlap_area(corners1, corners2)
    else:
        # Not overlapping - return minimum distance
        return calculate_min_distance(corners1, corners2)


def get_bounding_box(corners_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get bounding box for a list of rectangles
    
    Args:
        corners_list: List of rectangle corners
    
    Returns:
        Tuple of (min_coords, max_coords)
    """
    all_corners = np.vstack(corners_list)
    min_coords = np.min(all_corners, axis=0)
    max_coords = np.max(all_corners, axis=0)
    return min_coords, max_coords


def calculate_occupied_area(corners_list: List[np.ndarray]) -> float:
    """
    Calculate total occupied area (accounting for overlaps)
    
    Args:
        corners_list: List of rectangle corners
    
    Returns:
        Total occupied area
    """
    polygons = [Polygon(corners) for corners in corners_list]
    union = unary_union(polygons)
    return union.area


def grid_to_continuous(grid_x: int, grid_y: int, cell_size: float) -> Tuple[float, float]:
    """Convert grid coordinates to continuous coordinates"""
    return (grid_x + 0.5) * cell_size, (grid_y + 0.5) * cell_size


def continuous_to_grid(x: float, y: float, cell_size: float) -> Tuple[int, int]:
    """Convert continuous coordinates to grid coordinates"""
    return int(x / cell_size), int(y / cell_size)


def create_occupancy_grid(facility_length: float, facility_width: float, 
                         corners_list: List[np.ndarray], 
                         resolution: float = 0.5) -> np.ndarray:
    """
    Create an occupancy grid for the facility
    
    Args:
        facility_length: Facility length
        facility_width: Facility width
        corners_list: List of element corners
        resolution: Grid cell size in meters
    
    Returns:
        2D binary occupancy grid (1 = occupied, 0 = free)
    """
    grid_height = int(np.ceil(facility_length / resolution))
    grid_width = int(np.ceil(facility_width / resolution))
    grid = np.zeros((grid_height, grid_width), dtype=np.int8)
    
    for corners in corners_list:
        poly = Polygon(corners)
        
        # Find grid cells that intersect with the polygon
        min_x, min_y = np.min(corners, axis=0)
        max_x, max_y = np.max(corners, axis=0)
        
        min_i = max(0, int(min_x / resolution))
        max_i = min(grid_width, int(np.ceil(max_x / resolution)))
        min_j = max(0, int(min_y / resolution))
        max_j = min(grid_height, int(np.ceil(max_y / resolution)))
        
        for i in range(min_i, max_i):
            for j in range(min_j, max_j):
                # Check if cell center is inside polygon
                cell_center = np.array([(i + 0.5) * resolution, (j + 0.5) * resolution])
                if point_in_rectangle(cell_center, corners):
                    grid[j, i] = 1
    
    return grid


if __name__ == "__main__":
    # Test geometry utilities
    corners1 = get_rectangle_corners(5, 5, 2, 1, 0)
    corners2 = get_rectangle_corners(6, 5, 2, 1, 45)
    
    print(f"Collision: {check_collision(corners1, corners2)}")
    print(f"Distance: {calculate_centroid_distance(corners1, corners2):.2f}")
    print(f"Clearance: {calculate_clearance(corners1, corners2):.2f}")
