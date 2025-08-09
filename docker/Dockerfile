# Use the official Miniconda3 image
FROM continuumio/miniconda3:latest

# Avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for CompuCell3D
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libxrender1 \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libosmesa6 \
    libblas3 \
    liblapack3 \
    && apt-get clean

# Create the environment and install CompuCell3D
RUN conda update -n base -c defaults conda -y && \
    conda create -n cc3d_env python=3.10 -y && \
    conda install -n cc3d_env -c compucell3d -c conda-forge compucell3d -y && \
    conda clean --all --yes

# Add the environment to PATH
ENV PATH=/opt/conda/envs/cc3d_env/bin:$PATH

# Working directory inside the container
WORKDIR /sim

# Copy local files to the working directory
COPY . /sim

# Default command
CMD ["bash"]