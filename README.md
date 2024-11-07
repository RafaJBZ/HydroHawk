# HydroHawk

**HydroHawk** is an interactive application designed to keep a sharp eye on dam metrics like a hawk. Using Sentinel-1 satellite imagery, HydroHawk automates the process of downloading, processing, and visualizing satellite data to track changes in dam water levels over time—so you can spot those dam trends before they get out of hand.

---

## Table of Contents
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Setting Up Sentinel Hub Credentials](#setting-up-sentinel-hub-credentials)
- [Usage](#usage)
  - [Running the Application with Docker](#running-the-application-with-docker)
  - [Running the Application Locally](#running-the-application-locally)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Additional Resources](#additional-resources)

---

## Features
- **Automated Data Processing**: Download and process Sentinel-1 data for specified dams and date ranges—because nobody wants to manually crunch satellite data.
- **Flexible Time Intervals**: Set custom time intervals, including twice a month, monthly, bi-monthly, or whatever suits your hawk-like precision.
- **Interactive Visualization**: Visualize dam area changes over time with interactive charts and maps—see every drop, flap, and ripple.
- **Shapefile Generation**: Extract and display dam area shapefiles on interactive maps, because boundaries matter.
- **Dockerized Deployment**: Easily deploy the application using Docker, ensuring your environment is as reliable as a beaver’s dam.

## Getting Started
### Prerequisites
- **Docker**: Recommended for easy setup and deployment. [Download Docker](https://docs.docker.com/get-docker/)
- **Python 3.8+**: If you're planning to run the application locally without Docker—just in case you like the DIY approach.
- **Sentinel Hub Account**: Required to access Sentinel-1 satellite imagery. [Sign up for Sentinel Hub](https://www.sentinel-hub.com)

### Installation
#### Clone the Repository
```bash
git@github.com:RafaJBZ/HydroHawk.git
cd hydrohawk
```

#### (Optional) Create a Virtual Environment
If you plan to run the application locally without Docker:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Setting Up Sentinel Hub Credentials
The application requires Sentinel Hub API credentials to access satellite data. You need to provide your `SH_INSTANCE_ID`, `SH_CLIENT_ID`, and `SH_CLIENT_SECRET`.

#### Create a .env File
In the project root directory, create a file named `.env`:
```dotenv
SH_INSTANCE_ID=your_instance_id
SH_CLIENT_ID=your_client_id
SH_CLIENT_SECRET=your_client_secret
```
**Important**: Replace `your_instance_id`, `your_client_id`, and `your_client_secret` with your actual credentials.

**Security Note**:
- **Do Not Commit**: Ensure the `.env` file is not committed to version control.
- **.gitignore Entry**: The `.env` file is already listed in `.gitignore`.

## Usage
### Running the Application with Docker
**Step 1**: Build the Docker Image
```bash
docker build -t hydrohawk .
```

**Step 2**: Run the Docker Container
```bash
docker run -p 8501:8501 \
    -v .env \
    hydrohawk
```
- `-p 8501:8501`: Maps the container's port 8501 to your local machine.
- `-v $(pwd)/.env:/app/.env`: Mounts your `.env` file into the container.

**Step 3**: Access the Application
Open your browser and navigate to [http://localhost:8501](http://localhost:8501) to access HydroHawk.

### Running the Application Locally
**Step 1**: Install Dependencies
```bash
pip install -r requirements.txt
```

**Step 2**: Run the Application
```bash
streamlit run app/dam_processor_app.py
```

**Step 3**: Access the Application
Open your browser and navigate to [http://localhost:8501](http://localhost:8501).

## Project Structure
```
hydrohawk/
├── Dockerfile
├── app
│   ├── __init__.py
│   ├── dam_processor_app.py
│   └── sentinel1_processing.py
├── data
│   └── dams.csv
├── requirements.txt
└── .env
```
- **Dockerfile**: Defines the Docker image for the application.
- **app/**: Contains the application code.
  - **dam_processor_app.py**: The main Streamlit application script.
  - **sentinel1_processing.py**: Module for downloading and processing Sentinel-1 data.
- **data/**: Contains data files, including `dams.csv` with dam names and bounding box coordinates.
- **requirements.txt**: Lists all Python dependencies.
- **.env**: Contains Sentinel Hub API credentials (not committed to version control).

## Dependencies
Key Python libraries used in this project include:
- **Streamlit**: Web application framework for interactive apps.
- **Geopandas**: For geographic data processing.
- **PyDeck**: For interactive map visualizations.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical operations.
- **Rasterio**: Raster data processing.
- **Scikit-Learn**: Machine learning library for clustering algorithms.
- **Scikit-Image**: Image processing.
- **SentinelHub**: Access Sentinel Hub services and satellite data.
- **Matplotlib**: Plotting library—because we all love pretty pictures.
- **Python-Dotenv**: Reads key-value pairs from a `.env` file and adds them to environment variables.

All dependencies are listed in `requirements.txt`.

## Contributing
Contributions are welcome! To contribute:

1. **Fork the Repository**
   ```bash
   git fork git@github.com:RafaJBZ/HydroHawk.git
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**
   ```bash
   git commit -m 'Add your feature'
   ```

4. **Push to the Branch**
   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**
   Submit a pull request detailing your changes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
- **Author**: Rafael Juarez Badillo Chavez
- **Email**: [rafa.jbch123@gmail.com](mailto:rafa.jbch123@gmail.com)
- **GitHub**: [RafaJBZ](https://github.com/https://github.com/RafaJBZ)

Feel free to reach out for any questions, suggestions, or to just talk about dam cool things!

## Acknowledgements
- **Sentinel Hub**: For providing access to satellite imagery.
- **Streamlit Community**: For support and resources.
- **Open Source Libraries**: Thanks to all the contributors of the libraries used in this project.

## Troubleshooting
- **Docker Issues**: Ensure Docker is running and you have built the image correctly.
- **Missing Dependencies**: Install all required dependencies from `requirements.txt`.
- **Credentials Errors**: Verify your Sentinel Hub credentials are correct and set up in the `.env` file.
- **Port Conflicts**: Make sure port 8501 is not being used by another application.

## FAQ
1. **Can I add more dams to monitor?**
   - Absolutely! Just add the dam's name and bounding box coordinates to the `dams.csv` file in the `data` directory.

2. **How can I change the time intervals for data processing?**
   - The application lets you choose predefined intervals like twice a month, monthly, bi-monthly, or define your own within the app interface. Go wild!

3. **Where are the processed data and outputs stored?**
   - Processed data and outputs are stored in the output directory created by the application in the project root or a specified output folder.

## Additional Resources
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Sentinel Hub Documentation](https://www.sentinel-hub.com/develop/documentation/)
- [Geopandas Documentation](https://geopandas.org/)

---
Enjoy using HydroHawk for your dam monitoring and visualization needs—keep your eye on those water levels like a hawk!
