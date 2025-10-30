import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import List, Dict, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_schema import LayoutElement, FacilitySpecification, ElementType, LayoutConfiguration


# Color scheme for different element types
ELEMENT_COLORS = {
    ElementType.DESK: '#3498db',          # Blue
    ElementType.MACHINERY: '#e74c3c',     # Red
    ElementType.WORKSTATION: '#2ecc71',   # Green
    ElementType.STORAGE: '#f39c12',       # Orange
    ElementType.AISLE: '#95a5a6',         # Gray
    ElementType.EMERGENCY_EXIT: '#e67e22', # Dark Orange
    ElementType.MEETING_ROOM: '#9b59b6',  # Purple
    ElementType.EQUIPMENT: '#1abc9c'      # Turquoise
}


def create_box_mesh(x: float, y: float, z: float, 
                   length: float, width: float, height: float,
                   rotation: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create vertices and faces for a 3D box
    
    Args:
        x, y, z: Center position
        length, width, height: Box dimensions
        rotation: Rotation angle in degrees (around z-axis)
    
    Returns:
        Tuple of (vertices, faces)
    """
    # Define vertices of the box (relative to center)
    vertices = np.array([
        [-length/2, -width/2, 0],
        [length/2, -width/2, 0],
        [length/2, width/2, 0],
        [-length/2, width/2, 0],
        [-length/2, -width/2, height],
        [length/2, -width/2, height],
        [length/2, width/2, height],
        [-length/2, width/2, height]
    ])
    
    # Apply rotation around z-axis
    theta = np.radians(rotation)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    vertices = vertices @ rotation_matrix.T
    
    # Translate to position
    vertices += np.array([x, y, z])
    
    # Define faces (triangles)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom
        [4, 5, 6], [4, 6, 7],  # Top
        [0, 1, 5], [0, 5, 4],  # Front
        [2, 3, 7], [2, 7, 6],  # Back
        [1, 2, 6], [1, 6, 5],  # Right
        [0, 3, 7], [0, 7, 4]   # Left
    ])
    
    return vertices, faces


def create_floor_plan(facility: FacilitySpecification) -> go.Mesh3d:
    """
    Create a 3D mesh for the facility floor
    
    Args:
        facility: Facility specifications
    
    Returns:
        Plotly Mesh3d object for the floor
    """
    # Floor vertices
    vertices = np.array([
        [0, 0, 0],
        [facility.length, 0, 0],
        [facility.length, facility.width, 0],
        [0, facility.width, 0]
    ])
    
    # Floor faces
    i = [0, 0]
    j = [1, 2]
    k = [2, 3]
    
    floor = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=i, j=j, k=k,
        color='lightgray',
        opacity=0.3,
        name='Floor',
        showlegend=True,
        hoverinfo='name'
    )
    
    return floor


def create_element_mesh(element: LayoutElement) -> go.Mesh3d:
    """
    Create a 3D mesh for a layout element
    
    Args:
        element: Layout element
    
    Returns:
        Plotly Mesh3d object
    """
    vertices, faces = create_box_mesh(
        element.x, element.y, 0,
        element.length, element.width, element.height,
        element.rotation
    )
    
    # Get color based on element type
    color = ELEMENT_COLORS.get(element.element_type, '#34495e')
    
    # Extract face indices
    i = faces[:, 0]
    j = faces[:, 1]
    k = faces[:, 2]
    
    mesh = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=i, j=j, k=k,
        color=color,
        opacity=0.8,
        name=f"{element.element_id} ({element.element_type.value})",
        showlegend=True,
        hovertemplate=f"<b>{element.element_id}</b><br>" +
                     f"Type: {element.element_type.value}<br>" +
                     f"Position: ({element.x:.2f}, {element.y:.2f})<br>" +
                     f"Dimensions: {element.length:.2f} x {element.width:.2f} x {element.height:.2f}<br>" +
                     f"Rotation: {element.rotation:.1f}°<extra></extra>"
    )
    
    return mesh


def create_workflow_arrows(elements: List[LayoutElement], 
                          connections: List) -> List[go.Scatter3d]:
    """
    Create arrows showing workflow connections
    
    Args:
        elements: List of layout elements
        connections: List of workflow connections
    
    Returns:
        List of Plotly Scatter3d objects for arrows
    """
    arrows = []
    elem_dict = {e.element_id: e for e in elements}
    
    for conn in connections:
        if conn.from_element_id in elem_dict and conn.to_element_id in elem_dict:
            elem1 = elem_dict[conn.from_element_id]
            elem2 = elem_dict[conn.to_element_id]
            
            # Arrow from elem1 center to elem2 center (at height of 1.5m)
            arrow = go.Scatter3d(
                x=[elem1.x, elem2.x],
                y=[elem1.y, elem2.y],
                z=[1.5, 1.5],
                mode='lines',
                line=dict(
                    color='rgba(255, 0, 0, 0.5)',
                    width=2 * conn.priority
                ),
                name=f"{conn.from_element_id} → {conn.to_element_id}",
                showlegend=False,
                hovertemplate=f"<b>Workflow</b><br>" +
                             f"{conn.from_element_id} → {conn.to_element_id}<br>" +
                             f"Frequency: {conn.frequency}/hr<br>" +
                             f"Priority: {conn.priority}<extra></extra>"
            )
            arrows.append(arrow)
    
    return arrows


def create_3d_layout_visualization(config: LayoutConfiguration, 
                                  show_workflow: bool = True,
                                  show_grid: bool = True) -> go.Figure:
    """
    Create complete 3D visualization of the layout
    
    Args:
        config: Layout configuration
        show_workflow: Whether to show workflow arrows
        show_grid: Whether to show grid lines
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Add floor
    floor = create_floor_plan(config.facility)
    fig.add_trace(floor)
    
    # Add elements
    for element in config.elements:
        mesh = create_element_mesh(element)
        fig.add_trace(mesh)
    
    # Add workflow arrows if requested
    if show_workflow and config.workflow_requirements.connections:
        arrows = create_workflow_arrows(
            config.elements,
            config.workflow_requirements.connections
        )
        for arrow in arrows:
            fig.add_trace(arrow)
    
    # Add boundary box
    boundary_x = [0, config.facility.length, config.facility.length, 0, 0]
    boundary_y = [0, 0, config.facility.width, config.facility.width, 0]
    boundary_z = [0, 0, 0, 0, 0]
    
    fig.add_trace(go.Scatter3d(
        x=boundary_x,
        y=boundary_y,
        z=boundary_z,
        mode='lines',
        line=dict(color='black', width=3),
        name='Facility Boundary',
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Layout: {config.facility.name}",
        scene=dict(
            xaxis=dict(
                title='Length (m)',
                range=[0, config.facility.length],
                showgrid=show_grid,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title='Width (m)',
                range=[0, config.facility.width],
                showgrid=show_grid,
                gridcolor='lightgray'
            ),
            zaxis=dict(
                title='Height (m)',
                range=[0, config.facility.height],
                showgrid=show_grid,
                gridcolor='lightgray'
            ),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode='closest',
        height=700
    )
    
    return fig


def create_2d_layout_visualization(config: LayoutConfiguration,
                                  show_workflow: bool = True) -> go.Figure:
    """
    Create 2D top-down view of the layout
    
    Args:
        config: Layout configuration
        show_workflow: Whether to show workflow connections
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Add facility boundary
    fig.add_shape(
        type="rect",
        x0=0, y0=0,
        x1=config.facility.length,
        y1=config.facility.width,
        line=dict(color="black", width=3),
        fillcolor="lightgray",
        opacity=0.1
    )
    
    # Add elements
    for element in config.elements:
        corners = element.get_corners()
        
        # Close the polygon
        x = np.append(corners[:, 0], corners[0, 0])
        y = np.append(corners[:, 1], corners[0, 1])
        
        color = ELEMENT_COLORS.get(element.element_type, '#34495e')
        
        fig.add_trace(go.Scatter(
            x=x, y=y,
            fill='toself',
            fillcolor=color,
            opacity=0.7,
            line=dict(color='black', width=1),
            name=f"{element.element_id}",
            hovertemplate=f"<b>{element.element_id}</b><br>" +
                         f"Type: {element.element_type.value}<br>" +
                         f"Position: ({element.x:.2f}, {element.y:.2f})<br>" +
                         f"Rotation: {element.rotation:.1f}°<extra></extra>"
        ))
    
    # Add workflow connections
    if show_workflow and config.workflow_requirements.connections:
        elem_dict = {e.element_id: e for e in config.elements}
        
        for conn in config.workflow_requirements.connections:
            if conn.from_element_id in elem_dict and conn.to_element_id in elem_dict:
                elem1 = elem_dict[conn.from_element_id]
                elem2 = elem_dict[conn.to_element_id]
                
                fig.add_trace(go.Scatter(
                    x=[elem1.x, elem2.x],
                    y=[elem1.y, elem2.y],
                    mode='lines+markers',
                    line=dict(color='red', width=2 * conn.priority, dash='dash'),
                    marker=dict(size=5, symbol='arrow', angleref='previous'),
                    name=f"{conn.from_element_id} → {conn.to_element_id}",
                    showlegend=False,
                    hovertemplate=f"Workflow: {conn.from_element_id} → {conn.to_element_id}<br>" +
                                 f"Frequency: {conn.frequency}/hr<extra></extra>"
                ))
    
    # Update layout
    fig.update_layout(
        title=f"Top View: {config.facility.name}",
        xaxis=dict(title='Length (m)', range=[0, config.facility.length]),
        yaxis=dict(title='Width (m)', range=[0, config.facility.width], scaleanchor='x'),
        showlegend=True,
        hovermode='closest',
        height=600
    )
    
    return fig


if __name__ == "__main__":
    # Test visualization
    from utils.data_schema import create_sample_layout
    
    print("Creating sample layout...")
    config = create_sample_layout()
    
    # Random positions for testing
    for elem in config.elements:
        elem.x = np.random.uniform(2, config.facility.length - 2)
        elem.y = np.random.uniform(2, config.facility.width - 2)
        elem.rotation = np.random.uniform(0, 360)
    
    print("Creating 3D visualization...")
    fig_3d = create_3d_layout_visualization(config)
    fig_3d.write_html("test_3d_layout.html")
    print("Saved to: test_3d_layout.html")
    
    print("Creating 2D visualization...")
    fig_2d = create_2d_layout_visualization(config)
    fig_2d.write_html("test_2d_layout.html")
    print("Saved to: test_2d_layout.html")
