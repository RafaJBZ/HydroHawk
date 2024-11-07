import datetime
import numpy as np
import os
import calendar
import rasterio
from sklearn.cluster import DBSCAN
from skimage.morphology import closing, square, remove_small_objects
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    MimeType,
    MosaickingOrder,
    SentinelHubRequest,
    bbox_to_dimensions,
    SHConfig,
)
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from rasterio import features
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")


def download_and_process_sentinel1_data(
    bbox_wgs84,
    output_folder,
    folder_name,
    interval_type='months',
    interval_value=1,
    start_date=None,
    end_date=None,
    threshold_percentile=10,
    eps_meters=300,
    min_samples=5,
    structuring_element_size=3,
    min_object_area_m2=100000,
    polarization="VV",
    orbit_direction="ASCENDING",
):
    """
    Downloads and processes Sentinel-1 data for a specified bounding box and date range.

    Parameters:
    - bbox_wgs84 (tuple): Bounding box coordinates (min_lon, min_lat, max_lon, max_lat).
    - output_folder (str): Path to the main output folder.
    - folder_name (str): Name of the subfolder within the output folder.
    - interval_type (str): Type of interval ('days' or 'months').
    - interval_value (int): Number of days or months per processing interval.
    - start_date (datetime.date): Start date for processing.
    - end_date (datetime.date): End date for processing.
    """
    # Load Sentinel Hub configuration
    load_dotenv()
    config = SHConfig()
    config.instance_id = os.environ.get('SH_INSTANCE_ID')
    config.sh_client_id = os.environ.get('SH_CLIENT_ID')
    config.sh_client_secret = os.environ.get('SH_CLIENT_SECRET')

    # Validate polarization and orbit_direction
    if polarization not in ['VV', 'VH']:
        raise ValueError("Polarization must be 'VV' or 'VH'")
    if orbit_direction not in ['ASCENDING', 'DESCENDING']:
        raise ValueError("Orbit direction must be 'ASCENDING' or 'DESCENDING'")

    resolution = 10  # Resolution in meters
    bbox = BBox(bbox=bbox_wgs84, crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=resolution)

    # Generate time windows
    slots = get_time_windows(start_date, end_date, interval_type, interval_value)

    # Prepare data collection
    data_collection = DataCollection.SENTINEL1_IW
    if orbit_direction == 'ASCENDING':
        data_collection = DataCollection.SENTINEL1_IW_ASC
    elif orbit_direction == 'DESCENDING':
        data_collection = DataCollection.SENTINEL1_IW_DES

    # Create output folders
    s1_folder, dbscan_folder, metrics_folder = create_output_folders(output_folder, folder_name)

    # Initialize a list to store metrics for all time intervals
    metrics_list = []

    # Loop through each time interval and process data
    for idx, time_interval in enumerate(slots):
        print(f"\nProcessing slot: {time_interval}")
        try:
            # Download Sentinel-1 data
            s1_image = download_sentinel1_data(
                data_collection, bbox, size, time_interval, polarization, s1_folder, config
            )

            # Process the Sentinel-1 image
            water_mask = process_sentinel1_image(
                s1_image, threshold_percentile, structuring_element_size, min_object_area_m2,
                resolution
            )

            # Apply clustering to identify water bodies
            cluster_info = apply_clustering(
                water_mask, eps_meters, min_samples, resolution
            )

            if not cluster_info:
                print(f"No clusters detected for {time_interval}")
                # Append a metrics entry with zero area
                metrics_list.append({
                    'start_date': time_interval[0],
                    'end_date': time_interval[1],
                    'area_m2': 0
                })
                continue

            largest_cluster_mask, cluster_area_m2 = cluster_info

            # Save outputs and metrics
            save_outputs_and_metrics(
                s1_image, water_mask, largest_cluster_mask, cluster_area_m2, time_interval,
                bbox, s1_folder, dbscan_folder, metrics_folder, resolution, metrics_list
            )

        except ValueError as e:
            error_message = str(e)
            if "No valid Sentinel-1 data" in error_message or "No data available" in error_message:
                print(f"No data available for {time_interval}")
                # Append a metrics entry indicating no data
                metrics_list.append({
                    'start_date': time_interval[0],
                    'end_date': time_interval[1],
                    'area_m2': None  # Use None to indicate no data
                })
                continue
            else:
                # Handle other exceptions
                print(f"Error processing slot {time_interval}: {e}")
                continue

        except Exception as e:
            print(f"Unexpected error processing slot {time_interval}: {e}")
            continue

    # Save all metrics to CSV
    if metrics_list:
        save_metrics(metrics_list, metrics_folder)
    else:
        print("\nNo metrics to save.")


def get_time_windows(start_date, end_date, interval_type, interval_value):
    """
    Generates time windows based on the interval type and value.

    Parameters:
    - start_date (datetime.date): Start date for processing.
    - end_date (datetime.date): End date for processing.
    - interval_type (str): Type of interval ('days' or 'months').
    - interval_value (int): Number of days or months per processing interval.

    Returns:
    - List of tuples containing start and end dates in ISO format.
    """
    time_windows = []
    current_date = start_date

    while current_date <= end_date:
        if interval_type == 'months':
            # Calculate the start of the interval
            start_of_interval = current_date

            # Calculate the end date by adding interval_value months
            month = current_date.month - 1 + interval_value
            year = current_date.year + month // 12
            month = month % 12 + 1
            day = min(current_date.day, calendar.monthrange(year, month)[1])
            end_of_interval = datetime.date(year, month, day) - datetime.timedelta(days=1)

        elif interval_type == 'days':
            # Calculate the start and end dates by adding interval_value days
            start_of_interval = current_date
            end_of_interval = current_date + datetime.timedelta(days=interval_value - 1)

        else:
            raise ValueError("Invalid interval type. Must be 'days' or 'months'.")

        if end_of_interval > end_date:
            end_of_interval = end_date

        time_windows.append((start_of_interval.isoformat(), end_of_interval.isoformat()))

        # Move to the next interval
        current_date = end_of_interval + datetime.timedelta(days=1)

    return time_windows


def create_output_folders(output_folder, folder_name):
    """
    Creates necessary output folders and returns their paths.
    """
    os.makedirs(output_folder, exist_ok=True)
    s1_folder = os.path.join(output_folder, folder_name, "Sentinel1")
    dbscan_folder = os.path.join(output_folder, folder_name, "DBSCAN")
    metrics_folder = os.path.join(output_folder, folder_name, "Metrics")
    os.makedirs(s1_folder, exist_ok=True)
    os.makedirs(dbscan_folder, exist_ok=True)
    os.makedirs(metrics_folder, exist_ok=True)
    return s1_folder, dbscan_folder, metrics_folder


def download_sentinel1_data(
    data_collection, bbox, size, time_interval, polarization, s1_folder, config
):
    """
    Downloads Sentinel-1 data for a given time interval.

    Returns:
    - s1_image (numpy array): The downloaded Sentinel-1 image.
    """
    # Create a descriptive folder name using the time interval
    time_folder_name = f"data_{time_interval[0]}_{time_interval[1]}"
    s1_time_folder = os.path.join(s1_folder, time_folder_name)
    os.makedirs(s1_time_folder, exist_ok=True)

    # Evalscript to process Sentinel-1 data
    evalscript_sentinel1 = f"""
    //VERSION=3

    function setup() {{
        return {{
            input: ["{polarization}"],
            output: {{
                bands: 1,
                sampleType: "FLOAT32"
            }}
        }};
    }}

    function evaluatePixel(sample) {{
        return [10 * Math.log(sample.{polarization})];
    }}
    """

    request_s1 = SentinelHubRequest(
        data_folder=s1_time_folder,
        evalscript=evalscript_sentinel1,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=data_collection,
                time_interval=time_interval,
                mosaicking_order=MosaickingOrder.MOST_RECENT,
            )
        ],
        responses=[
            SentinelHubRequest.output_response("default", MimeType.TIFF)
        ],
        bbox=bbox,
        size=size,
        config=config,
    )

    # Download and save the Sentinel-1 data
    s1_data = request_s1.get_data(save_data=True)
    print(f"Sentinel-1 data for {time_interval} saved in {s1_time_folder}")

    if not s1_data or s1_data[0] is None:
        raise ValueError(f"No data available for {time_interval}")

    s1_image = s1_data[0].squeeze()  # Remove single-dimensional entries

    # Handle invalid values
    s1_image = np.ma.masked_invalid(s1_image)

    if s1_image.count() == 0:
        raise ValueError(f"No valid Sentinel-1 data for {time_interval}")

    return s1_image


def process_sentinel1_image(
    s1_image, threshold_percentile, structuring_element_size, min_object_area_m2,
    resolution
):
    """
    Processes the Sentinel-1 image to generate a water mask.

    Returns:
    - water_mask (numpy array): The processed binary water mask.
    """
    # Apply thresholding to create a binary water mask
    threshold = np.percentile(s1_image.compressed(), threshold_percentile)
    print(f"Using threshold at {threshold_percentile}th percentile: {threshold}")
    water_mask = s1_image < threshold

    # Apply morphological closing to remove small holes and connect regions
    water_mask = closing(water_mask, square(structuring_element_size))

    # Calculate pixel area
    pixel_area_m2 = resolution ** 2

    # Calculate min_size in pixels
    min_size = int(min_object_area_m2 / pixel_area_m2)

    # Remove small objects (noise)
    water_mask = remove_small_objects(water_mask, min_size=min_size)

    # Fill all holes within the water body
    water_mask = binary_fill_holes(water_mask)

    return water_mask


def apply_clustering(water_mask, eps_meters, min_samples, resolution):
    """
    Applies DBSCAN clustering to identify water bodies.

    Returns:
    - (largest_cluster_mask, cluster_area_m2): Tuple containing the mask of the largest cluster and its area in m2.
    """
    # Get coordinates of water body pixels
    points = np.column_stack(np.nonzero(water_mask))

    if points.size == 0:
        print("No water bodies detected.")
        return None

    # Adjust DBSCAN parameters based on resolution
    eps_pixels = eps_meters / resolution

    dbscan = DBSCAN(eps=eps_pixels, min_samples=min_samples).fit(points)
    labels = dbscan.labels_

    # Number of clusters detected (-1 is noise)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Number of clusters detected: {n_clusters}")

    if n_clusters == 0:
        return None

    # Create an array to store cluster labels for each pixel
    cluster_labels = np.full(water_mask.shape, -1, dtype=int)
    cluster_labels[points[:, 0], points[:, 1]] = labels

    # Compute area for each cluster
    cluster_areas = {}
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue  # Skip noise
        cluster_mask = cluster_labels == cluster_id
        cluster_area_pixels = np.sum(cluster_mask)
        cluster_areas[cluster_id] = cluster_area_pixels

    # Find the cluster with the maximum area
    largest_cluster_id = max(cluster_areas, key=cluster_areas.get)
    largest_cluster_mask = cluster_labels == largest_cluster_id
    cluster_area_pixels = cluster_areas[largest_cluster_id]
    pixel_area_m2 = resolution ** 2
    cluster_area_m2 = cluster_area_pixels * pixel_area_m2

    print(f"Largest cluster ID: {largest_cluster_id} with area {cluster_area_m2} m²")

    return largest_cluster_mask, cluster_area_m2


def save_outputs_and_metrics(
    s1_image, water_mask, largest_cluster_mask, cluster_area_m2, time_interval,
    bbox, s1_folder, dbscan_folder, metrics_folder, resolution, metrics_list
):
    """
    Saves the processed images, shapefiles, and updates the metrics list.
    """
    # Create a descriptive folder name using the time interval
    time_folder_name = f"data_{time_interval[0]}_{time_interval[1]}"
    s1_time_folder = os.path.join(s1_folder, time_folder_name)
    dbscan_time_folder = os.path.join(dbscan_folder, time_folder_name)
    os.makedirs(dbscan_time_folder, exist_ok=True)

    # Save water mask as a TIFF file
    water_mask_path = os.path.join(s1_time_folder, 'water_mask.tiff')
    save_raster(water_mask.astype('uint8'), water_mask_path, bbox)

    # Save largest cluster as a TIFF file
    cluster_labels_path = os.path.join(dbscan_time_folder, 'largest_cluster.tiff')
    save_raster(largest_cluster_mask.astype('uint8'), cluster_labels_path, bbox)

    # Extract and save shapefile
    extract_shapefile(largest_cluster_mask, bbox, dbscan_time_folder)

    # Plot and save the Sentinel-1 image
    s1_plot_path = os.path.join(s1_time_folder, 's1_plot.png')
    save_plot(s1_image, s1_plot_path, f'Sentinel-1 Backscatter - {time_interval[0]} to {time_interval[1]}')

    # Plot and save the largest cluster
    dbscan_plot_path = os.path.join(dbscan_time_folder, 'largest_cluster_plot.png')
    save_plot(
        largest_cluster_mask, dbscan_plot_path,
        f'Largest Water Body (Dam) - {time_interval[0]} to {time_interval[1]}', cmap='tab20'
    )

    # Update metrics
    metrics_list.append({
        'start_date': time_interval[0],
        'end_date': time_interval[1],
        'area_m2': cluster_area_m2
    })

    print(f"Processed data for {time_interval} saved in {dbscan_time_folder}")


def save_raster(array, path, bbox):
    """
    Saves a numpy array as a raster TIFF file.
    """
    height, width = array.shape
    transform = rasterio.transform.from_bounds(
        bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y,
        width, height
    )
    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': array.dtype,
        'crs': bbox.crs.pyproj_crs(),
        'transform': transform
    }
    with rasterio.Env():
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(array, 1)


def extract_shapefile(cluster_mask, bbox, dbscan_time_folder):
    """
    Extracts the shapefile from the cluster mask and saves it.
    """
    # Create Shapefiles subfolder
    shapefile_folder = os.path.join(dbscan_time_folder, 'Shapefiles')
    os.makedirs(shapefile_folder, exist_ok=True)

    # Transform the raster coordinates to geographic coordinates
    height, width = cluster_mask.shape
    transform = rasterio.transform.from_bounds(
        bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y,
        width, height
    )

    # Extract shapes
    shapes_generator = features.shapes(
        cluster_mask.astype('uint8'),
        mask=cluster_mask.astype(bool),
        transform=transform
    )

    # Convert shapes to geometries
    geometries = []
    for geom, value in shapes_generator:
        if value == 1:
            geometries.append(shape(geom))

    # Create a GeoDataFrame
    if geometries:
        dam_gdf = gpd.GeoDataFrame({'geometry': geometries}, crs=bbox.crs.pyproj_crs())

        # Define the path to save the shapefile
        dam_shapefile_path = os.path.join(shapefile_folder, 'dam_shape.shp')

        # Save the GeoDataFrame as a shapefile
        dam_gdf.to_file(dam_shapefile_path)
        print(f"Shapefile saved at {dam_shapefile_path}")
    else:
        print("No geometries extracted for shapefile.")


def save_plot(array, path, title, cmap='gray'):
    """
    Saves a plot of the given array.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(array, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    plt.savefig(path)
    plt.close()


def save_metrics(metrics_list, metrics_folder):
    """
    Saves the metrics list to a CSV file.
    """
    metrics_df = pd.DataFrame(metrics_list)
    metrics_csv_path = os.path.join(metrics_folder, 'cluster_metrics.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\nMetrics saved to {metrics_csv_path}")