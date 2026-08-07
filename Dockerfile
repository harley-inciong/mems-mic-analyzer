# Use an official Python slim runtime as a parent image
FROM python:3.10-slim

# Set system environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (build-essential needed for some scipy packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for caching
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY config.yaml .
COPY dashboard/ ./dashboard/
COPY tests/ ./tests/

# Expose port 8501 for Streamlit access
EXPOSE 8501

# Healthcheck to verify container health
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch the Streamlit dashboard on startup
CMD ["streamlit", "run", "dashboard/app.py"]
