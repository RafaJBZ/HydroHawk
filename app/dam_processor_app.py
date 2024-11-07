import streamlit as st
import os
import datetime
import pandas as pd
import geopandas as gpd
import altair as alt
import calendar
import pydeck as pdk
from sentinel1_processing import download_and_process_sentinel1_data

# Load the dams list from the CSV file
@st.cache_data
def load_dams(csv_path=None):
    if csv_path is None:
        # Get the directory where the current script is located
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the path to dams.csv
        csv_path = os.path.join(base_dir, '..', 'data', 'dams.csv')
    return pd.read_csv(csv_path)

def display_metrics_and_images(metrics_df, output_folder, folder_name, key_prefix=''):
    st.subheader("Dam Area Over Time")

    # Convert date columns to datetime
    metrics_df['start_date'] = pd.to_datetime(metrics_df['start_date'])
    metrics_df['start_date_str'] = metrics_df['start_date'].dt.strftime('%Y-%m-%d')

    # Handle missing area_m2 values
    metrics_df['area_m2'] = pd.to_numeric(metrics_df['area_m2'], errors='coerce')

    # Provide a date selection widget
    date_options = metrics_df['start_date_str'].tolist()
    selected_date_str = st.selectbox("Select Date", options=date_options, key=f"{key_prefix}_date_select")

    # Filter the metrics dataframe for the selected date
    selected_metrics = metrics_df[metrics_df['start_date_str'] == selected_date_str]

    # Create the timeseries chart
    base = alt.Chart(metrics_df).encode(
        x=alt.X('start_date:T', title='Start Date'),
        y=alt.Y('area_m2:Q', title='Dam Area (m²)')
    )

    # Line chart
    line = base.mark_line().encode(
        tooltip=[alt.Tooltip('start_date_str:N', title='Start Date'), alt.Tooltip('area_m2:Q', title='Dam Area (m²)')]
    )

    # Highlight the selected date
    highlight = alt.Chart(selected_metrics).mark_circle(size=100, color='red').encode(
        x='start_date:T',
        y='area_m2:Q'
    )

    # Combine the charts
    chart = (line + highlight).properties(width=700, height=400)

    st.altair_chart(chart, use_container_width=True)

    # Check if area_m2 is None or NaN
    if selected_metrics.empty or pd.isna(selected_metrics['area_m2'].iloc[0]):
        st.warning(f"No data available for {selected_date_str}.")
    else:
        # Display the images and metrics for the selected date
        display_images_and_metrics_for_date(selected_metrics.iloc[0], output_folder, folder_name)

def display_images_and_metrics_for_date(selected_metrics, output_folder, folder_name):
    start_date_str = selected_metrics['start_date'].strftime('%Y-%m-%d')
    end_date_str = selected_metrics['end_date'].strftime('%Y-%m-%d')

    st.subheader(f"Data for {start_date_str} to {end_date_str}")
    st.write(f"**Dam Area (m²):** {selected_metrics['area_m2']:,}")

    # Paths to images
    s1_image_path = os.path.join(
        output_folder,
        folder_name,
        "Sentinel1",
        f"data_{start_date_str}_{end_date_str}",
        "s1_plot.png"
    )
    cluster_image_path = os.path.join(
        output_folder,
        folder_name,
        "DBSCAN",
        f"data_{start_date_str}_{end_date_str}",
        "largest_cluster_plot.png"
    )

    # Check if images exist and display them
    if os.path.exists(s1_image_path) and os.path.exists(cluster_image_path):
        col1, col2 = st.columns(2)
        with col1:
            st.image(s1_image_path, caption="Sentinel-1 Backscatter", use_column_width=True)
        with col2:
            st.image(cluster_image_path, caption="Detected Dam Area", use_column_width=True)
    else:
        st.warning("Images not found for the selected date range.")

    # Display shapefile on a map using st.pydeck_chart()
    st.subheader("Dam Area Map")
    shapefile_path = os.path.join(
        output_folder,
        folder_name,
        "DBSCAN",
        f"data_{start_date_str}_{end_date_str}",
        "Shapefiles",
        "dam_shape.shp"
    )

    if os.path.exists(shapefile_path):
        try:
            dam_gdf = gpd.read_file(shapefile_path)
            # Ensure the GeoDataFrame is in WGS84 coordinate system
            dam_gdf = dam_gdf.to_crs(epsg=4326)

            # Convert the geometries to GeoJSON format
            geojson = dam_gdf.__geo_interface__

            # Create a PyDeck layer for the geometries
            layer = pdk.Layer(
                "GeoJsonLayer",
                geojson,
                get_fill_color=[255, 0, 0, 140],  # Semi-transparent red
                get_line_color=[255, 0, 0],       # Red outline
                pickable=True,
                auto_highlight=True,
            )

            # Compute the centroid of the geometries for the initial view
            centroid = dam_gdf.geometry.centroid
            view_state = pdk.ViewState(
                longitude=centroid.x.mean(),
                latitude=centroid.y.mean(),
                zoom=12,
                min_zoom=5,
                max_zoom=20,
            )

            # Create the deck.gl map
            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"text": "Dam Area"}
            )
            st.pydeck_chart(r)

        except Exception as e:
            st.error(f"Error loading shapefile: {e}")
    else:
        st.warning("Shapefile not found for the selected date range.")

def main():
    st.title("Dam Sentinel-1 Data Processor and Visualizer")

    # Load dams
    dams_df = load_dams()

    # Define default output folder accessible in both tabs
    default_output_folder = os.path.join(os.getcwd(), "output")

    # Create tabs
    tab1, tab2 = st.tabs(["Data Processing", "Data Visualization"])

    with tab1:
        st.header("Data Processing")

        # Dam selection
        dam_name = st.selectbox("Select a Dam", dams_df['dam_name'].unique(), key='processing_dam')

        # Get the selected dam's bounding box
        selected_dam = dams_df[dams_df['dam_name'] == dam_name].iloc[0]
        bbox = (
            selected_dam['min_lon'],
            selected_dam['min_lat'],
            selected_dam['max_lon'],
            selected_dam['max_lat']
        )

        # Output folder selection
        output_folder = st.text_input("Output Folder", value=default_output_folder, key='processing_output_folder')

        # Folder name
        folder_name = st.text_input("Folder Name", value=dam_name.replace(" ", "_"), key='processing_folder_name')

        # Store output_folder and folder_name in session_state for access later
        st.session_state['output_folder'] = output_folder
        st.session_state['folder_name'] = folder_name

        # Date range selection (Months and Years)
        st.subheader("Select Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_year = st.number_input("Start Year", min_value=2000, max_value=datetime.date.today().year, value=2022, key='start_year')
            start_month = st.selectbox("Start Month", list(range(1, 13)), format_func=lambda x: calendar.month_name[x], key='start_month')
        with col2:
            end_year = st.number_input("End Year", min_value=2000, max_value=datetime.date.today().year + 1, value=2022, key='end_year')
            end_month = st.selectbox("End Month", list(range(1, 13)), format_func=lambda x: calendar.month_name[x], key='end_month')

        # Construct start and end dates
        start_date = datetime.date(int(start_year), int(start_month), 1)
        last_day = calendar.monthrange(int(end_year), int(end_month))[1]
        end_date = datetime.date(int(end_year), int(end_month), last_day)

        # Ensure start date is before end date
        if start_date > end_date:
            st.error("Error: End date must fall after start date.")
            return

        # Time interval granularity
        interval_option = st.selectbox(
            "Time Interval Granularity",
            options=["Twice a Month", "Monthly", "Bi-monthly", "Custom"],
            key='interval_option'
        )

        # Set interval type and value based on selection
        interval_type = 'months'
        interval_value = 1  # Default to 1 month

        if interval_option == "Twice a Month":
            interval_type = 'days'
            interval_value = 15
        elif interval_option == "Monthly":
            interval_type = 'months'
            interval_value = 1
        elif interval_option == "Bi-monthly":
            interval_type = 'months'
            interval_value = 2
        elif interval_option == "Custom":
            interval_type = st.selectbox(
                "Interval Type",
                options=["Days", "Months"],
                key='custom_interval_type'
            ).lower()
            interval_value = st.number_input(
                f"Number of {interval_type} per interval",
                min_value=1,
                max_value=31 if interval_type == 'days' else 12,
                value=1,
                key='custom_interval_value'
            )

        # Run the processing when the button is clicked
        if st.button("Start Processing", key='start_processing'):
            st.write("Processing data...")
            with st.spinner('Processing...'):
                download_and_process_sentinel1_data(
                    bbox_wgs84=bbox,
                    output_folder=output_folder,
                    folder_name=folder_name,
                    interval_type=interval_type,
                    interval_value=interval_value,
                    start_date=start_date,
                    end_date=end_date
                )
            st.success("Processing completed.")

            # Set a flag to indicate processing completed
            st.session_state['processing_completed'] = True

        # Check if processing has been completed or metrics exist
        metrics_csv_path = os.path.join(
            output_folder, folder_name, "Metrics", "cluster_metrics.csv"
        )

        if os.path.exists(metrics_csv_path):
            # Read the metrics data
            metrics_df = pd.read_csv(metrics_csv_path)
            # Convert date columns to datetime
            metrics_df['start_date'] = pd.to_datetime(metrics_df['start_date'])
            metrics_df['end_date'] = pd.to_datetime(metrics_df['end_date'])

            # Display metrics and images with unique key prefix
            display_metrics_and_images(metrics_df, output_folder, folder_name, key_prefix='processing')
        else:
            st.info("No metrics found. Please run processing.")

    with tab2:
        st.header("Data Visualization")

        # Let user select the dam and date to view existing results
        existing_dams = dams_df['dam_name'].unique()
        existing_dam = st.selectbox("Select a Dam", existing_dams, key="visualization_dam")
        existing_folder_name = existing_dam.replace(" ", "_")
        existing_output_folder = st.text_input("Output Folder", value=default_output_folder, key="visualization_output_folder")

        existing_metrics_csv_path = os.path.join(
            existing_output_folder, existing_folder_name, "Metrics", "cluster_metrics.csv"
        )

        if os.path.exists(existing_metrics_csv_path):
            existing_metrics_df = pd.read_csv(existing_metrics_csv_path)
            # Convert date columns to datetime
            existing_metrics_df['start_date'] = pd.to_datetime(existing_metrics_df['start_date'])
            existing_metrics_df['end_date'] = pd.to_datetime(existing_metrics_df['end_date'])

            # Display metrics and images with unique key prefix
            display_metrics_and_images(existing_metrics_df, existing_output_folder, existing_folder_name, key_prefix='visualization')
        else:
            st.warning("Metrics file not found for existing results.")

if __name__ == "__main__":
    main()
