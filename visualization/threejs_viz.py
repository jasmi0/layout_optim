import numpy as np
from typing import List, Dict, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_schema import LayoutElement, FacilitySpecification, ElementType, LayoutConfiguration

try:
    from pythreejs import *
    import ipywidgets as widgets
    from IPython.display import display
    THREEJS_AVAILABLE = True
except ImportError:
    THREEJS_AVAILABLE = False
    print("⚠️ PyThreeJS not available. Install with: pip install pythreejs ipywidgets")


# Color scheme for different element types (hex colors)
ELEMENT_COLORS_HEX = {
    ElementType.DESK: '#3498db',          # Blue
    ElementType.MACHINERY: '#e74c3c',     # Red
    ElementType.WORKSTATION: '#2ecc71',   # Green
    ElementType.STORAGE: '#f39c12',       # Orange
    ElementType.AISLE: '#95a5a6',         # Gray
    ElementType.EMERGENCY_EXIT: '#e67e22', # Dark Orange
    ElementType.MEETING_ROOM: '#9b59b6',  # Purple
    ElementType.EQUIPMENT: '#1abc9c'      # Turquoise
}


def create_box_geometry(length: float, width: float, height: float) -> BoxGeometry:
    """
    Create a box geometry for Three.js
    
    Three.js BoxGeometry parameters:
    - width: size along X axis
    - height: size along Y axis  
    - depth: size along Z axis
    
    Our coordinate system:
    - length: X axis (facility length)
    - width: Z axis (facility width)
    - height: Y axis (vertical)
    """
    if not THREEJS_AVAILABLE:
        return None
    return BoxGeometry(width=length, height=height, depth=width)


def create_element_mesh_threejs(element: LayoutElement) -> Mesh:
    """
    Create a Three.js mesh for a layout element
    
    Args:
        element: Layout element
    
    Returns:
        Three.js Mesh object
    """
    if not THREEJS_AVAILABLE:
        return None
    
    # Create geometry - BoxGeometry(width, height, depth) in Three.js coordinate system
    # Our system: length=X, width=Z, height=Y
    geometry = create_box_geometry(element.length, element.width, element.height)
    
    # Get color based on element type
    color = ELEMENT_COLORS_HEX.get(element.element_type, '#34495e')
    
    # Create material
    material = MeshStandardMaterial(
        color=color,
        metalness=0.2,
        roughness=0.6,
        transparent=False,
        opacity=1.0
    )
    
    # Create mesh
    mesh = Mesh(geometry=geometry, material=material, name=element.element_id)
    
    # Set position (Three.js uses center position)
    # Convert from our coordinate system where z=0 is floor
    mesh.position = (float(element.x), float(element.height / 2), float(element.y))
    
    # Set rotation (convert degrees to radians, rotate around Y axis)
    # PyThreeJS expects (x, y, z, order) where order is 'XYZ', 'YXZ', etc.
    rotation_y = float(np.radians(element.rotation))
    mesh.rotation = (0.0, rotation_y, 0.0, 'XYZ')
    
    return mesh


def create_floor_mesh_threejs(facility: FacilitySpecification) -> Mesh:
    """
    Create a floor mesh for the facility
    
    Args:
        facility: Facility specifications
    
    Returns:
        Three.js Mesh object for the floor
    """
    if not THREEJS_AVAILABLE:
        return None
    
    # Create floor geometry
    geometry = BoxGeometry(
        width=facility.length,
        height=0.1,
        depth=facility.width
    )
    
    # Create floor material
    material = MeshStandardMaterial(
        color='#ecf0f1',
        metalness=0.1,
        roughness=0.9,
        transparent=False,
        opacity=1.0
    )
    
    # Create mesh at ground level
    mesh = Mesh(geometry=geometry, material=material, name='floor')
    mesh.position = (float(facility.length / 2), -0.05, float(facility.width / 2))
    
    return mesh


def create_grid_helper_threejs(facility: FacilitySpecification) -> GridHelper:
    """
    Create a grid helper for the floor
    
    Args:
        facility: Facility specifications
    
    Returns:
        Three.js GridHelper object
    """
    if not THREEJS_AVAILABLE:
        return None
    
    size = float(max(facility.length, facility.width))
    divisions = int(size / 2)
    
    grid = GridHelper(
        size=size,
        divisions=divisions,
        color_center_line='#34495e',
        color_grid='#95a5a6'
    )
    
    # Position at center of facility
    grid.position = (float(facility.length / 2), 0.0, float(facility.width / 2))
    
    return grid


def create_boundary_walls_threejs(facility: FacilitySpecification) -> List:
    """
    Create solid boundary walls for the facility
    
    Args:
        facility: Facility specifications
    
    Returns:
        List of Three.js Mesh objects representing walls
    """
    if not THREEJS_AVAILABLE:
        return []
    
    walls = []
    wall_thickness = 0.2  # 20cm thick walls
    wall_height = float(facility.height)
    
    # Material for walls - semi-transparent glass-like
    material = MeshStandardMaterial(
        color='#ecf0f1',
        metalness=0.1,
        roughness=0.3,
        transparent=True,
        opacity=0.6,
        side='DoubleSide'
    )
    
    # Wall 1: Along length (front, at z=0)
    wall1_geom = BoxGeometry(
        width=float(facility.length),
        height=wall_height,
        depth=wall_thickness
    )
    wall1 = Mesh(geometry=wall1_geom, material=material, name='wall_front')
    wall1.position = (float(facility.length / 2), float(wall_height / 2), -wall_thickness / 2)
    walls.append(wall1)
    
    # Wall 2: Along length (back, at z=width)
    wall2_geom = BoxGeometry(
        width=float(facility.length),
        height=wall_height,
        depth=wall_thickness
    )
    wall2 = Mesh(geometry=wall2_geom, material=material, name='wall_back')
    wall2.position = (float(facility.length / 2), float(wall_height / 2), float(facility.width + wall_thickness / 2))
    walls.append(wall2)
    
    # Wall 3: Along width (left, at x=0)
    wall3_geom = BoxGeometry(
        width=wall_thickness,
        height=wall_height,
        depth=float(facility.width)
    )
    wall3 = Mesh(geometry=wall3_geom, material=material, name='wall_left')
    wall3.position = (-wall_thickness / 2, float(wall_height / 2), float(facility.width / 2))
    walls.append(wall3)
    
    # Wall 4: Along width (right, at x=length)
    wall4_geom = BoxGeometry(
        width=wall_thickness,
        height=wall_height,
        depth=float(facility.width)
    )
    wall4 = Mesh(geometry=wall4_geom, material=material, name='wall_right')
    wall4.position = (float(facility.length + wall_thickness / 2), float(wall_height / 2), float(facility.width / 2))
    walls.append(wall4)
    
    return walls


def create_workflow_arrows_threejs(elements: List[LayoutElement], 
                                   connections: List) -> List:
    """
    Create arrows showing workflow connections
    
    Args:
        elements: List of layout elements
        connections: List of workflow connections
    
    Returns:
        List of Three.js arrow objects
    """
    if not THREEJS_AVAILABLE:
        return []
    
    arrows = []
    elem_dict = {e.element_id: e for e in elements}
    
    for conn in connections:
        if conn.from_element_id in elem_dict and conn.to_element_id in elem_dict:
            elem1 = elem_dict[conn.from_element_id]
            elem2 = elem_dict[conn.to_element_id]
            
            # Arrow from elem1 to elem2 at height of 1.5m
            start = [elem1.x, 1.5, elem1.y]
            end = [elem2.x, 1.5, elem2.y]
            
            # Calculate direction
            direction = np.array(end) - np.array(start)
            length = np.linalg.norm(direction)
            
            if length > 0:
                direction = direction / length
                
                # Create arrow
                arrow = ArrowHelper(
                    dir=direction.tolist(),
                    origin=start,
                    length=length,
                    color='#e74c3c',
                    headLength=0.5,
                    headWidth=0.3
                )
                
                arrows.append(arrow)
    
    return arrows


def create_threejs_scene(config: LayoutConfiguration, 
                         show_workflow: bool = True,
                         show_grid: bool = True) -> Dict:
    """
    Create a complete Three.js scene for the layout
    
    Args:
        config: Layout configuration
        show_workflow: Whether to show workflow arrows
        show_grid: Whether to show grid
    
    Returns:
        Dictionary with scene components or None if PyThreeJS not available
    """
    if not THREEJS_AVAILABLE:
        print("⚠️ PyThreeJS not available. Install with: pip install pythreejs ipywidgets")
        return None
    
    # Create scene
    scene = Scene()
    
    # Add children to scene
    children = []
    
    # Add floor
    floor = create_floor_mesh_threejs(config.facility)
    if floor:
        children.append(floor)
    
    # Add grid if requested
    if show_grid:
        grid = create_grid_helper_threejs(config.facility)
        if grid:
            children.append(grid)
    
    # Add boundary walls
    walls = create_boundary_walls_threejs(config.facility)
    for wall in walls:
        children.append(wall)
    
    # Add elements
    for element in config.elements:
        mesh = create_element_mesh_threejs(element)
        if mesh:
            children.append(mesh)
    
    # Add workflow arrows if requested
    if show_workflow and config.workflow_requirements.connections:
        arrows = create_workflow_arrows_threejs(
            config.elements,
            config.workflow_requirements.connections
        )
        children.extend(arrows)
    
    scene.children = children
    
    # Create camera - position it to look at the center
    center_x = float(config.facility.length / 2)
    center_y = float(config.facility.height / 2)
    center_z = float(config.facility.width / 2)
    
    # Camera position - further back and higher up
    cam_distance = float(max(config.facility.length, config.facility.width) * 2)
    
    camera = PerspectiveCamera(
        position=[center_x + cam_distance, 
                 center_y + cam_distance, 
                 center_z + cam_distance],
        fov=60,
        aspect=16/9,
        near=0.1,
        far=10000
    )
    
    # Point camera at center of facility
    camera.up = [0, 1, 0]
    
    # Create orbit controls with target at center
    controls = OrbitControls(controlling=camera)
    controls.target = [center_x, center_y, center_z]
    
    # Add lights - much brighter
    ambient_light = AmbientLight(color='#ffffff', intensity=1.5)
    
    # Add multiple directional lights for better illumination
    directional_light1 = DirectionalLight(
        color='#ffffff',
        intensity=1.5,
        position=[float(config.facility.length * 2), 
                 float(config.facility.height * 3), 
                 float(config.facility.width * 2)]
    )
    
    directional_light2 = DirectionalLight(
        color='#ffffff',
        intensity=1.0,
        position=[float(-config.facility.length), 
                 float(config.facility.height * 2), 
                 float(-config.facility.width)]
    )
    
    # Add a point light at camera position
    point_light = PointLight(
        color='#ffffff',
        intensity=2.0,
        position=[center_x + cam_distance, center_y + cam_distance, center_z + cam_distance]
    )
    
    # Add lights to children list
    children.append(ambient_light)
    children.append(directional_light1)
    children.append(directional_light2)
    children.append(point_light)
    
    # Set all children at once
    scene.children = children
    
    # Add background color to scene
    scene.background = '#87ceeb'  # Sky blue background
    
    # Create renderer
    renderer = Renderer(
        camera=camera,
        scene=scene,
        controls=[controls],
        width=1000,
        height=600,
        antialias=True
    )
    
    return {
        'scene': scene,
        'camera': camera,
        'renderer': renderer,
        'lights': [ambient_light, directional_light1, directional_light2, point_light],
        'controls': controls
    }


def display_threejs_scene(config: LayoutConfiguration, 
                         show_workflow: bool = True,
                         show_grid: bool = True):
    """
    Display the Three.js scene (for Jupyter notebooks)
    
    Args:
        config: Layout configuration
        show_workflow: Whether to show workflow arrows
        show_grid: Whether to show grid
    """
    if not THREEJS_AVAILABLE:
        print("⚠️ PyThreeJS not available. Install with: pip install pythreejs ipywidgets")
        return None
    
    scene_dict = create_threejs_scene(config, show_workflow, show_grid)
    
    if scene_dict:
        # Display the renderer
        display(scene_dict['renderer'])
        
        # Add title
        title = widgets.HTML(f"<h3>Layout: {config.facility.name}</h3>")
        display(title)
        
        return scene_dict['renderer']
    
    return None


def export_threejs_html(config: LayoutConfiguration, 
                       filename: str = "layout_3d.html",
                       show_workflow: bool = True,
                       show_grid: bool = True):
    """
    Export the Three.js scene to an HTML file
    
    Args:
        config: Layout configuration
        filename: Output filename
        show_workflow: Whether to show workflow arrows
        show_grid: Whether to show grid
    """
    if not THREEJS_AVAILABLE:
        print("⚠️ PyThreeJS not available. Install with: pip install pythreejs ipywidgets")
        return
    
    scene_dict = create_threejs_scene(config, show_workflow, show_grid)
    
    if scene_dict:
        from ipywidgets.embed import embed_minimal_html
        
        embed_minimal_html(
            filename,
            views=[scene_dict['renderer']],
            title=f"Layout: {config.facility.name}"
        )
        
        print(f"✓ Exported to {filename}")


if __name__ == "__main__":
    # Test PyThreeJS visualization
    if THREEJS_AVAILABLE:
        from utils.data_schema import create_sample_layout
        
        print("Creating sample layout...")
        config = create_sample_layout()
        
        # Random positions for testing
        for elem in config.elements:
            elem.x = np.random.uniform(2, config.facility.length - 2)
            elem.y = np.random.uniform(2, config.facility.width - 2)
            elem.rotation = np.random.uniform(0, 360)
        
        print("Creating Three.js scene...")
        scene_dict = create_threejs_scene(config)
        
        if scene_dict:
            print("✓ Three.js scene created successfully!")
            print("Use display_threejs_scene() in Jupyter or export_threejs_html() to save")
    else:
        print("❌ PyThreeJS not installed. Install with: pip install pythreejs ipywidgets")
