FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create data directory
RUN mkdir -p /app/data

# Copy the application code
COPY . .

# Make sure the data directory is writable
RUN chmod 777 /app/data

# Expose the port
EXPOSE 5000

# Set Python to run in unbuffered mode and limit memory usage
ENV PYTHONUNBUFFERED=1
ENV MALLOC_TRIM_THRESHOLD_=65536

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Command to run the application with memory limits
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "1", "--limit-concurrency", "10"]