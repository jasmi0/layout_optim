import streamlit as st
import numpy as np
import pandas as pd
from typing import List
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_schema import (
    FacilitySpecification,
    LayoutElement,
    ElementType,
    SafetyRegulation,
    WorkflowRequirements,
    LayoutConfiguration
)
from utils.constraints import ConstraintChecker, calculate_space_utilization
from visualization.layout_viz import create_3d_layout_visualization, create_2d_layout_visualization
from environments.layout_env import LayoutOptimizationEnv

# Try to import RL components (may not be available initially)
try:
    from stable_baselines3 import PPO
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    st.warning("Stable-Baselines3 not installed. RL optimization will not be available.")


# Page configuration
st.set_page_config(
    page_title="Layout Optimization System",
    page_icon="Layout Optimization System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if 'facility' not in st.session_state:
        st.session_state.facility = None
    if 'elements' not in st.session_state:
        st.session_state.elements = []
    if 'safety_regulations' not in st.session_state:
        st.session_state.safety_regulations = SafetyRegulation()
    if 'workflow_requirements' not in st.session_state:
        st.session_state.workflow_requirements = WorkflowRequirements()
    if 'config' not in st.session_state:
        st.session_state.config = None
    if 'optimized' not in st.session_state:
        st.session_state.optimized = False


def sidebar_navigation():
    """Sidebar navigation"""
    st.sidebar.title("Layout Optimizer")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Configuration", "Optimization", "Visualization", "Analysis"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**About**\n\n"
        "This system uses reinforcement learning to optimize facility layouts "
        "based on space utilization, safety regulations, and workflow efficiency."
    )
    
    return page


def home_page():
    """Home page with project overview"""
    st.markdown('<div class="main-header">Reinforcement Layout Optimization System</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ## Welcome to the AI-Powered Layout Optimization Platform
    
    This application uses **Reinforcement Learning** to optimize facility layouts by considering:
    
    - **Space Utilization**: Maximize efficient use of available space
    - **Safety Compliance**: Adhere to safety regulations and clearance requirements
    - **Workflow Efficiency**: Optimize material flow and personnel movement
    - **Accessibility**: Ensure proper aisle widths and emergency exits
    
    ### How to Use
    
    1. **Configuration**: Define your facility dimensions, layout elements, and constraints
    2. **Optimization**: Run the RL algorithm to find optimal layout
    3. **Visualization**: View results in interactive 2D and 3D views
    4. **Analysis**: Examine performance metrics and validation results
    
    ### Getting Started
    
    Use the sidebar to navigate through the application. Start by configuring your facility
    in the **Configuration** section.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Facility Setup**\n\nDefine dimensions and constraints")
    
    with col2:
        st.success("**AI Optimization**\n\nLet RL find the best layout")
    
    with col3:
        st.warning("**Results**\n\nVisualize and analyze")


def configuration_page():
    """Configuration page for defining layout parameters"""
    st.title("Layout Configuration")
    
    # Facility Configuration
    st.header("1. Facility Specifications")
    
    col1, col2 = st.columns(2)
    with col1:
        facility_name = st.text_input("Facility Name", value="My Facility")
        facility_length = st.number_input("Length (m)", min_value=5.0, max_value=200.0, value=20.0, step=1.0)
        facility_width = st.number_input("Width (m)", min_value=5.0, max_value=200.0, value=15.0, step=1.0)
    
    with col2:
        facility_height = st.number_input("Height (m)", min_value=2.0, max_value=20.0, value=3.0, step=0.5)
        st.metric("Total Area", f"{facility_length * facility_width:.2f} m²")
    
    # Safety Regulations
    st.header("2. Safety Regulations")
    
    col1, col2 = st.columns(2)
    with col1:
        min_aisle_width = st.number_input("Min Aisle Width (m)", min_value=0.8, max_value=3.0, value=1.2, step=0.1)
        emergency_clearance = st.number_input("Emergency Exit Clearance (m)", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
    
    with col2:
        min_exit_count = st.number_input("Minimum Exit Count", min_value=1, max_value=10, value=2)
        max_travel_distance = st.number_input("Max Travel Distance to Exit (m)", min_value=10.0, max_value=100.0, value=30.0, step=5.0)
    
    # Layout Elements
    st.header("3. Layout Elements")
    
    # Preset templates
    st.subheader("Quick Start Templates")
    template = st.selectbox(
        "Load Template",
        ["Custom", "Small Office (5-10 elements)", "Medium Office (10-20 elements)", "Warehouse"]
    )
    
    if template != "Custom" and st.button("Load Template"):
        st.session_state.elements = load_template(template, facility_length, facility_width)
        st.success(f"Loaded {template} template")
    
    # Element editor
    st.subheader("Add/Edit Elements")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        elem_id = st.text_input("Element ID", value=f"elem_{len(st.session_state.elements) + 1}")
        elem_type = st.selectbox("Type", [e.value for e in ElementType])
    
    with col2:
        elem_length = st.number_input("Length (m)", min_value=0.5, max_value=20.0, value=1.6, step=0.1, key="elem_length")
        elem_width = st.number_input("Width (m)", min_value=0.5, max_value=20.0, value=0.8, step=0.1, key="elem_width")
    
    with col3:
        elem_height = st.number_input("Height (m)", min_value=0.5, max_value=5.0, value=0.75, step=0.1, key="elem_height")
        elem_clearance = st.number_input("Min Clearance (m)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
    
    if st.button("Add Element"):
        element = LayoutElement(
            element_id=elem_id,
            element_type=ElementType(elem_type),
            length=elem_length,
            width=elem_width,
            height=elem_height,
            min_clearance=elem_clearance
        )
        st.session_state.elements.append(element)
        st.success(f"Added {elem_id}")
    
    # Display current elements
    if st.session_state.elements:
        st.subheader(f"Current Elements ({len(st.session_state.elements)})")
        
        elements_data = []
        for elem in st.session_state.elements:
            elements_data.append({
                'ID': elem.element_id,
                'Type': elem.element_type.value,
                'Dimensions': f"{elem.length}×{elem.width}×{elem.height}",
                'Area': f"{elem.get_footprint_area():.2f} m²"
            })
        
        df = pd.DataFrame(elements_data)
        st.dataframe(df, use_container_width=True)
        
        if st.button("Clear All Elements"):
            st.session_state.elements = []
            st.rerun()
    
    # Save configuration
    st.markdown("---")
    if st.button("Save Configuration", type="primary"):
        try:
            facility = FacilitySpecification(
                length=facility_length,
                width=facility_width,
                height=facility_height,
                name=facility_name
            )
            
            safety = SafetyRegulation(
                min_aisle_width=min_aisle_width,
                emergency_exit_clearance=emergency_clearance,
                min_exit_count=min_exit_count,
                max_travel_distance_to_exit=max_travel_distance
            )
            
            workflow = WorkflowRequirements()
            
            config = LayoutConfiguration(
                facility=facility,
                elements=st.session_state.elements,
                safety_regulations=safety,
                workflow_requirements=workflow
            )
            
            config.validate()
            st.session_state.config = config
            st.session_state.facility = facility
            st.session_state.safety_regulations = safety
            
            st.success("Configuration saved successfully!")
            st.info(f"Total element area: {config.get_total_element_area():.2f} m² " +
                   f"({config.get_space_utilization():.1f}% of facility)")
            
        except Exception as e:
            st.error(f"Configuration error: {e}")


def optimization_page():
    """Optimization page for running RL algorithm"""
    st.title("Layout Optimization")
    
    if st.session_state.config is None:
        st.warning("Please configure your layout first in the Configuration page.")
        return
    
    config = st.session_state.config
    
    # Display current configuration summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Facility Size", f"{config.facility.length}×{config.facility.width} m")
    with col2:
        st.metric("Elements", len(config.elements))
    with col3:
        st.metric("Initial Utilization", f"{config.get_space_utilization():.1f}%")
    
    st.markdown("---")
    
    # Optimization settings
    st.header("Optimization Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        algorithm = st.selectbox("Algorithm", ["PPO", "SAC", "A2C"], disabled=not RL_AVAILABLE)
        max_steps = st.slider("Max Steps per Episode", 100, 1000, 500)
    
    with col2:
        st.markdown("**Reward Weights**")
        w_space = st.slider("Space Utilization", 0.0, 1.0, 0.3, 0.05)
        w_safety = st.slider("Safety", 0.0, 1.0, 0.4, 0.05)
        w_workflow = st.slider("Workflow", 0.0, 1.0, 0.2, 0.05)
        w_access = st.slider("Accessibility", 0.0, 1.0, 0.1, 0.05)
    
    reward_weights = {
        'space_utilization': w_space,
        'safety': w_safety,
        'workflow': w_workflow,
        'accessibility': w_access
    }
    
    # Run optimization
    st.markdown("---")
    
    if st.button("Run Optimization", type="primary", disabled=not RL_AVAILABLE):
        # Check if configuration has elements
        if not config.elements or len(config.elements) == 0:
            st.error("No elements in configuration! Please add elements in the Configuration page first.")
            return
        
        with st.spinner("Optimizing layout..."):
            try:
                # Create environment
                env = LayoutOptimizationEnv(config, max_steps=max_steps, reward_weights=reward_weights)
                
                # For demo purposes, run a simple optimization loop
                # In production, this would use a trained model
                st.info("Running optimization (this may take a few minutes)...")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                obs, info = env.reset()
                best_reward = -np.inf
                best_elements = None
                
                for step in range(max_steps):
                    # Random action for demo (replace with model.predict() when trained model is available)
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    if reward > best_reward:
                        best_reward = reward
                        # Save best configuration
                        best_elements = [
                            LayoutElement(
                                element_id=e.element_id,
                                element_type=e.element_type,
                                length=e.length,
                                width=e.width,
                                height=e.height,
                                x=e.x,
                                y=e.y,
                                rotation=e.rotation,
                                min_clearance=e.min_clearance
                            ) for e in env.elements
                        ]
                    
                    # Update progress
                    progress_bar.progress((step + 1) / max_steps)
                    if step % 50 == 0:
                        status_text.text(f"Step {step}/{max_steps} - Best Reward: {best_reward:.4f}")
                    
                    if terminated or truncated:
                        break
                
                # Update configuration with best layout
                if best_elements:
                    st.session_state.config.elements = best_elements
                    st.session_state.optimized = True
                    
                    progress_bar.progress(1.0)
                    status_text.text(f"Optimization complete! Best Reward: {best_reward:.4f}")
                    
                    st.success("Optimization completed successfully!")
                    st.balloons()
                
            except Exception as e:
                st.error(f"Optimization error: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Quick test (no ML)
    if st.button("Random Layout (No ML)"):
        for elem in config.elements:
            elem.x = np.random.uniform(elem.length/2, config.facility.length - elem.length/2)
            elem.y = np.random.uniform(elem.width/2, config.facility.width - elem.width/2)
            elem.rotation = np.random.uniform(0, 360)
        st.session_state.optimized = True
        st.success("Random layout generated!")


def visualization_page():
    """Visualization page for viewing layouts"""
    st.title("Layout Visualization")
    
    if st.session_state.config is None:
        st.warning("Please configure your layout first.")
        return
    
    config = st.session_state.config
    
    # View options
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        view_mode = st.radio("View Mode", ["3D View (Plotly)", "2D Top View", "3D WebGL (Three.js)"], horizontal=False)
    with col2:
        show_workflow = st.checkbox("Show Workflow", value=True)
        show_grid = st.checkbox("Show Grid", value=True)
    with col3:
        if view_mode == "3D WebGL (Three.js)":
            if st.button("Export HTML"):
                try:
                    from visualization.threejs_viz import export_threejs_html, THREEJS_AVAILABLE
                    if THREEJS_AVAILABLE:
                        export_threejs_html(config, "layout_threejs.html", show_workflow, show_grid)
                        st.success("Exported to layout_threejs.html")
                    else:
                        st.warning("PyThreeJS not available")
                except Exception as e:
                    st.error(f"Export error: {e}")
    
    st.markdown("---")
    
    try:
        if view_mode == "3D View (Plotly)":
            fig = create_3d_layout_visualization(config, show_workflow=show_workflow)
            st.plotly_chart(fig, use_container_width=True)
        elif view_mode == "2D Top View":
            fig = create_2d_layout_visualization(config, show_workflow=show_workflow)
            st.plotly_chart(fig, use_container_width=True)
        elif view_mode == "3D WebGL (Three.js)":
            try:
                from visualization.threejs_viz import create_threejs_scene, THREEJS_AVAILABLE
                import tempfile
                import os
                
                if not THREEJS_AVAILABLE:
                    st.warning("PyThreeJS not installed. Install with: `pip install pythreejs ipywidgets`")
                    st.info("Falling back to Plotly 3D view...")
                    fig = create_3d_layout_visualization(config, show_workflow=show_workflow)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"Creating Three.js scene for {len(config.elements)} elements...")
                    scene_dict = create_threejs_scene(config, show_workflow, show_grid)
                    if scene_dict and scene_dict['renderer']:
                        st.success("Scene created successfully")
                        
                        # Export to temporary HTML file and embed
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                            from ipywidgets.embed import embed_minimal_html
                            embed_minimal_html(f.name, views=[scene_dict['renderer']], 
                                             title=f"Layout: {config.facility.name}")
                            temp_file = f.name
                        
                        # Read the HTML file
                        with open(temp_file, 'r') as f:
                            html_content = f.read()
                        
                        # Clean up temp file
                        try:
                            os.unlink(temp_file)
                        except:
                            pass
                        
                        # Display using Streamlit components
                        st.components.v1.html(html_content, height=650, scrolling=False)
                        st.info("Use mouse to rotate, zoom, and pan the 3D view")
                        st.warning("**Note:** If you see a black screen, PyThreeJS may have rendering issues in Streamlit. " +
                                 "Try:\n1. Click 'Export HTML' button above to view in a browser\n2. Use 'Plotly 3D View' instead (recommended)")
                        st.caption(f"Camera: {scene_dict['camera'].position} | Elements: {len(config.elements)} | Lights: {len(scene_dict['lights'])}")
                    else:
                        st.error("Failed to create Three.js scene")
            except Exception as e:
                st.error(f"Three.js error: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.info("Falling back to Plotly 3D view...")
                fig = create_3d_layout_visualization(config, show_workflow=show_workflow)
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Visualization error: {e}")
        import traceback
        st.code(traceback.format_exc())


def analysis_page():
    """Analysis page for performance metrics"""
    st.title("Layout Analysis")
    
    if st.session_state.config is None:
        st.warning("Please configure your layout first.")
        return
    
    config = st.session_state.config
    
    # Run validation
    checker = ConstraintChecker(config.facility, config.safety_regulations)
    validation = checker.validate_layout(config.elements)
    
    # Display metrics
    st.header("Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        util = calculate_space_utilization(config.elements, config.facility)
        st.metric("Space Utilization", f"{util:.1f}%", 
                 delta="Good" if util > 60 and util < 85 else "Review")
    
    with col2:
        st.metric("Safety Score", f"{validation['safety_score']:.2f}",
                 delta="Pass" if validation['safety_score'] > 0.8 else "Fail")
    
    with col3:
        st.metric("Valid Layout", "Valid" if validation['valid'] else "Invalid")
    
    with col4:
        st.metric("Total Elements", len(config.elements))
    
    # Detailed validation results
    st.markdown("---")
    st.header("Validation Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Compliance")
        if validation['collision_count'] == 0:
            st.success("No collisions detected")
        else:
            st.error(f"{validation['collision_count']} collisions detected")
        
        if validation['boundary_violations'] == 0:
            st.success("All elements within boundaries")
        else:
            st.error(f"{validation['boundary_violations']} boundary violations")
    
    with col2:
        st.subheader("Issues")
        if validation['clearance_violations'] == 0:
            st.success("All clearance requirements met")
        else:
            st.warning(f"{validation['clearance_violations']} clearance violations")
        
        if validation['aisle_width_ok']:
            st.success("Aisle width requirements met")
        else:
            st.warning("Insufficient aisle width")
    
    # Element details
    st.markdown("---")
    st.header("Element Details")
    
    elements_data = []
    for elem in config.elements:
        elements_data.append({
            'ID': elem.element_id,
            'Type': elem.element_type.value,
            'Position': f"({elem.x:.2f}, {elem.y:.2f})",
            'Rotation': f"{elem.rotation:.1f}°",
            'Area': f"{elem.get_footprint_area():.2f} m²"
        })
    
    df = pd.DataFrame(elements_data)
    st.dataframe(df, use_container_width=True)


def load_template(template_name: str, facility_length: float, facility_width: float) -> List[LayoutElement]:
    """Load predefined layout templates"""
    elements = []
    
    if "Small Office" in template_name:
        elements = [
            LayoutElement("desk_1", ElementType.DESK, 1.6, 0.8, 0.75),
            LayoutElement("desk_2", ElementType.DESK, 1.6, 0.8, 0.75),
            LayoutElement("desk_3", ElementType.DESK, 1.6, 0.8, 0.75),
            LayoutElement("meeting_1", ElementType.MEETING_ROOM, 4.0, 3.0, 2.5, min_clearance=1.0),
            LayoutElement("storage_1", ElementType.STORAGE, 2.0, 1.5, 2.0),
        ]
    elif "Medium Office" in template_name:
        # Add more desks and workstations
        for i in range(8):
            elements.append(LayoutElement(f"desk_{i+1}", ElementType.DESK, 1.6, 0.8, 0.75))
        elements.extend([
            LayoutElement("meeting_1", ElementType.MEETING_ROOM, 4.0, 3.0, 2.5, min_clearance=1.0),
            LayoutElement("meeting_2", ElementType.MEETING_ROOM, 3.0, 2.5, 2.5, min_clearance=1.0),
            LayoutElement("storage_1", ElementType.STORAGE, 2.0, 1.5, 2.0),
            LayoutElement("workstation_1", ElementType.WORKSTATION, 2.0, 1.0, 1.0),
        ])
    elif "Warehouse" in template_name:
        # Add storage and machinery
        for i in range(6):
            elements.append(LayoutElement(f"storage_{i+1}", ElementType.STORAGE, 3.0, 2.0, 3.0))
        elements.extend([
            LayoutElement("machinery_1", ElementType.MACHINERY, 4.0, 3.0, 2.5, weight=500.0, is_movable=False),
            LayoutElement("workstation_1", ElementType.WORKSTATION, 2.0, 1.5, 1.0),
        ])
    
    return elements


def main():
    """Main application"""
    init_session_state()
    
    page = sidebar_navigation()
    
    if page == "Home":
        home_page()
    elif page == "Configuration":
        configuration_page()
    elif page == "Optimization":
        optimization_page()
    elif page == "Visualization":
        visualization_page()
    elif page == "Analysis":
        analysis_page()


if __name__ == "__main__":
    main()
