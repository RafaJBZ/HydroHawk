# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgdal-dev \
    gdal-bin \
    libspatialindex-dev \
    libgeos-dev \
    libproj-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install GDAL Python package dependencies
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Copy requirements.txt into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app ./app

# Copy the data directory (e.g., dams.csv)
COPY data ./data

# Copy the .env file if you prefer to include it in the image (not recommended for sensitive data)
# Alternatively, you can mount it at runtime

# Expose the port Streamlit uses (default is 8501)
EXPOSE 8501

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_PORT=8501

# Command to run when the container starts
CMD ["streamlit", "run", "app/dam_processor_app.py"]
