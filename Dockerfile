# Use Ubuntu 24.04 as the base image
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System dependencies:
# - python3, venv, pip
# - libegl1, libgl1, libglib2.0-0 for Open3D on Ubuntu 24.04
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a Python virtual environment
RUN python3 -m venv /opt/venv

# Use venv Python and pip
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
# rosbags (latest) with full ROS 2 support
RUN pip install --no-cache-dir \
    rosbags \
    open3d \
    numpy \
    scipy \
    tqdm

# Copy script
COPY bag_to_mesh.py .

RUN chmod +x bag_to_mesh.py

# Explicitly use venv python as entrypoint
ENTRYPOINT ["/opt/venv/bin/python", "/app/bag_to_mesh.py"]
