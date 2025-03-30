import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io

# Set page configuration
st.set_page_config(
    page_title="Shankar's Data Plot & Delta Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve the appearance
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .plot-container {
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stSlider {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .stats-box {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(file):
    """Load and parse the CSV data, handling the specific format in the file."""
    content = file.read().decode('utf-8')
    lines = content.split('\n')
    
    # Look for the "Data" section
    data_start_index = -1
    for i, line in enumerate(lines):
        if line.strip() == 'Data':
            data_start_index = i + 1
            break
    
    if data_start_index == -1:
        st.error("Could not find 'Data' section in the file!")
        return None
    
    # Get the header and data rows
    header = lines[data_start_index].strip()
    column_names = header.split(',')[:2]  # Take only the first two columns
    
    # Extract and parse data rows
    data_rows = []
    for i in range(data_start_index + 1, len(lines)):
        line = lines[i].strip()
        if line:
            parts = line.split(',')
            if len(parts) >= 2 and all(isfloat(p) for p in parts[:2]):
                data_rows.append([float(parts[0]), float(parts[1])])
    
    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=['x', 'y'])
    return df

def isfloat(value):
    """Check if a string can be converted to float."""
    try:
        float(value)
        return True
    except ValueError:
        return False

def find_nearest_point(df, x_value):
    """Find the point in the DataFrame that is closest to the given x value."""
    idx = (df['x'] - x_value).abs().idxmin()
    return df.iloc[idx]

def level_data(df, point1, point2):
    """Level the data using linear fit between the two selected points."""
    # If points are the same, return the original data
    if point1['x'] == point2['x']:
        return df.copy()
    
    # Calculate the slope and intercept for the linear fit
    slope = (point2['y'] - point1['y']) / (point2['x'] - point1['x'])
    intercept = point1['y'] - slope * point1['x']
    
    # Create a copy of the dataframe for the leveled data
    leveled_df = df.copy()
    
    # Calculate the linear fit line at each x point
    fit_line = slope * df['x'] + intercept
    
    # Subtract the fit line from the original y values to level the data
    leveled_df['y'] = df['y'] - fit_line
    
    return leveled_df

def create_plot(df, point1=None, point2=None, delta_line=False, custom_range=None, show_leveled=False):
    """Create a plot with optional markers, delta line, and leveled data."""
    fig = go.Figure()
    
    # Add the main data trace
    fig.add_trace(go.Scatter(
        x=df['x'], 
        y=df['y'],
        mode='lines',
        name='Original Data',
        line=dict(color='royalblue', width=2)
    ))
    
    # Add leveled data if requested
    if show_leveled and point1 is not None and point2 is not None:
        leveled_df = level_data(df, point1, point2)
        fig.add_trace(go.Scatter(
            x=leveled_df['x'],
            y=leveled_df['y'],
            mode='lines',
            name='Leveled Data',
            line=dict(color='purple', width=2)
        ))
        
        # Find leveled points corresponding to the original selected points
        leveled_point1 = {'x': point1['x'], 'y': leveled_df.loc[df['x'] == point1['x'], 'y'].values[0]}
        leveled_point2 = {'x': point2['x'], 'y': leveled_df.loc[df['x'] == point2['x'], 'y'].values[0]}
        
        # Add leveled point markers
        fig.add_trace(go.Scatter(
            x=[leveled_point1['x'], leveled_point2['x']],
            y=[leveled_point1['y'], leveled_point2['y']],
            mode='markers',
            marker=dict(size=10, color=['magenta', 'purple']),
            name='Leveled Points'
        ))
    
    # Add original points and delta line if specified
    if point1 is not None and point2 is not None:
        # Add point markers
        fig.add_trace(go.Scatter(
            x=[point1['x'], point2['x']],
            y=[point1['y'], point2['y']],
            mode='markers',
            marker=dict(size=10, color=['#ff7f0e', '#2ca02c']),
            name='Selected Points'
        ))
        
        # Add labels for points
        fig.add_annotation(
            x=point1['x'],
            y=point1['y'],
            text=f"Point 1: ({point1['x']:.4f}, {point1['y']:.2f})",
            showarrow=True,
            arrowhead=1,
            ax=40,
            ay=-40
        )
        
        fig.add_annotation(
            x=point2['x'],
            y=point2['y'],
            text=f"Point 2: ({point2['x']:.4f}, {point2['y']:.2f})",
            showarrow=True,
            arrowhead=1,
            ax=-40,
            ay=40
        )
        
        # Draw delta line if requested
        if delta_line:
            fig.add_shape(
                type="line",
                x0=point1['x'],
                y0=point1['y'],
                x1=point2['x'],
                y1=point2['y'],
                line=dict(color="red", width=2, dash="dash"),
            )
            
            # Add delta annotation
            delta_x = point2['x'] - point1['x']
            delta_y = point2['y'] - point1['y']
            mid_x = (point1['x'] + point2['x']) / 2
            mid_y = (point1['y'] + point2['y']) / 2
            
            fig.add_annotation(
                x=mid_x,
                y=mid_y,
                text=f"Δx: {delta_x:.4f} mm<br>Δy: {delta_y:.2f} nm",
                showarrow=True,
                arrowhead=0,
                ax=0,
                ay=-60,
                bordercolor="#c7c7c7",
                borderwidth=2,
                borderpad=4,
                bgcolor="#ff7f0e",
                opacity=0.8,
                font=dict(color="white")
            )
    
    # Set layout
    fig.update_layout(
        title="Data Visualization",
        xaxis_title="Lateral Position (mm)",
        yaxis_title="Total Profile (nm)",
        legend_title="Legend",
        hovermode="closest",
        template="plotly_white",
        margin=dict(l=50, r=50, t=50, b=50),
    )
    
    # Apply custom range if provided
    if custom_range:
        fig.update_xaxes(range=custom_range['x'])
        fig.update_yaxes(range=custom_range['y'])
    
    return fig

def main():
    st.title("Data Plotting & Delta Analysis Tool")
    
    # Sidebar for controls only
    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            if df is not None and not df.empty:
                st.success(f"Successfully loaded {len(df)} data points!")
                
                # Controls in sidebar
                st.subheader("Visualization Options")
                show_delta = st.checkbox("Show delta between points", value=True)
                show_delta_line = st.checkbox("Show connecting line", value=True)
                show_leveled = st.checkbox("Show leveled data", value=False)
                
                enable_zoom = st.checkbox("Enable custom zoom", value=False)
                
                custom_range = None
                if enable_zoom:
                    x_min, x_max = st.slider("X-axis range (mm)", 
                                             float(df['x'].min()), 
                                             float(df['x'].max()), 
                                             (float(df['x'].min()), float(df['x'].max())))
                    
                    y_min, y_max = st.slider("Y-axis range (nm)", 
                                             float(df['y'].min()), 
                                             float(df['y'].max()), 
                                             (float(df['y'].min()), float(df['y'].max())))
                    
                    custom_range = {'x': [x_min, x_max], 'y': [y_min, y_max]}
                
                st.subheader("Point Selection")
                x_point1 = st.slider("X position for Point 1 (mm)", 
                                    float(df['x'].min()), 
                                    float(df['x'].max()), 
                                    float(df['x'].min()))
                
                x_point2 = st.slider("X position for Point 2 (mm)", 
                                    float(df['x'].min()), 
                                    float(df['x'].max()), 
                                    float(df['x'].max()))
            else:
                st.error("Failed to load data from the file. Make sure it's in the correct format.")
        else:
            st.info("Please upload a CSV file to begin analysis.")
    
    # Main content area
    if 'uploaded_file' in locals() and uploaded_file is not None:
        if 'df' in locals() and df is not None and not df.empty:
            # Main layout
            st.header("Data Visualization")
            
            # Find closest points in the dataset
            point1 = find_nearest_point(df, x_point1)
            point2 = find_nearest_point(df, x_point2)
            
            # Plot and info columns
            col1, col2 = st.columns([3, 1])
            
            with col1:
                fig = create_plot(df, point1, point2, show_delta_line, custom_range, show_leveled)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Data statistics
                st.subheader("Data Statistics")
                stats = pd.DataFrame({
                    'Min': [df['x'].min(), df['y'].min()],
                    'Max': [df['x'].max(), df['y'].max()],
                    'Mean': [df['x'].mean(), df['y'].mean()],
                    'Std Dev': [df['x'].std(), df['y'].std()]
                }, index=['X (mm)', 'Y (nm)'])
                
                st.dataframe(stats)
                
                if show_delta:
                    st.subheader("Selected Points")
                    
                    st.markdown(f"""
                    <div class="stats-box">
                        <strong>Point 1:</strong><br>
                        X: {point1['x']:.4f} mm<br>
                        Y: {point1['y']:.2f} nm
                    </div>
                    
                    <div class="stats-box">
                        <strong>Point 2:</strong><br>
                        X: {point2['x']:.4f} mm<br>
                        Y: {point2['y']:.2f} nm
                    </div>
                    
                    <div class="stats-box">
                        <strong>Delta Values:</strong><br>
                        ΔX: {point2['x'] - point1['x']:.4f} mm<br>
                        ΔY: {point2['y'] - point1['y']:.2f} nm<br>
                        Slope: {(point2['y'] - point1['y']) / (point2['x'] - point1['x']):.2f} nm/mm
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add leveled data stats if requested
                    if show_leveled:
                        leveled_df = level_data(df, point1, point2)
                        
                        # Find leveled points corresponding to the original selected points
                        leveled_point1 = {'x': point1['x'], 'y': leveled_df.loc[df['x'] == point1['x'], 'y'].values[0]}
                        leveled_point2 = {'x': point2['x'], 'y': leveled_df.loc[df['x'] == point2['x'], 'y'].values[0]}
                        
                        # Calculate delta values
                        delta_x = leveled_point2['x'] - leveled_point1['x']
                        delta_y = leveled_point2['y'] - leveled_point1['y']
                        
                        # Calculate slope (should be close to zero)
                        slope = 0
                        if delta_x != 0:  # Avoid division by zero
                            slope = delta_y / delta_x
                            
                        st.markdown(f"""
                        <div class="stats-box" style="background-color: #f0e6ff;">
                            <strong>Leveled Data:</strong><br>
                            Point 1: ({leveled_point1['x']:.4f}, {leveled_point1['y']:.2f}) nm<br>
                            Point 2: ({leveled_point2['x']:.4f}, {leveled_point2['y']:.2f}) nm<br>
                            ΔY (leveled): {delta_y:.2f} nm<br>
                            Slope (leveled): {slope:.2f} nm/mm<br>
                            <small>Note: The slope between points should be near zero after leveling</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add option to download leveled data
                        st.markdown("##### Download Leveled Data")
                        csv_buffer_leveled = io.StringIO()
                        leveled_df.to_csv(csv_buffer_leveled, index=False)
                        st.download_button(
                            label="Download leveled data as CSV",
                            data=csv_buffer_leveled.getvalue(),
                            file_name="leveled_data.csv",
                            mime="text/csv"
                        )
            
            # Data viewer (in an expander)
            with st.expander("View Data Table"):
                st.dataframe(df.head(100))
                
                # Download processed data as CSV
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download full dataset as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="processed_data.csv",
                    mime="text/csv"
                )
    else:
        # Instructions when no file is uploaded
        st.markdown("""
        ### How to use this tool:
        1. Upload your CSV data file using the sidebar
        2. Use the sliders to select two points on the graph
        3. View the delta values between the selected points
        4. Adjust zoom settings if needed for a closer look at specific regions
        """)

if __name__ == "__main__":
    main()
