FROM ubuntu:22.04

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install OS-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    git \
    ca-certificates \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

# Install uv (latest via official installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"

# Install Python 3.13 using uv
RUN uv python install 3.13

# Install scientific python stack
RUN uv pip install cython numpy setuptools build scipy sympy matplotlib
