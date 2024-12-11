FROM --platform=linux/arm64 ubuntu:22.04

# Set environment to non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    git \
    libsdl2-dev \
    libsdl2-image-dev \
    pkg-config \
    g++

# Set working directory
WORKDIR /app

# Copy your project files
COPY . .

# Compile your project
RUN mkdir -p build && \
    cd build && \
    cmake .. && \
    make