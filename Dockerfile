# Use lightweight official Python image
FROM python:3.10-slim

# Install system dependencies needed for matplotlib / OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code
COPY . .

# Expose the port Hugging Face Spaces expects
EXPOSE 7860

# Run the Flask app on host 0.0.0.0 and port 7860
CMD ["python", "app.py"]
