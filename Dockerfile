# Use a lightweight Python base image
FROM python:3.10-slim

# Install system dependencies required for pdf2image (poppler)
RUN apt-get update && apt-get install -y \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port FastAPI runs on
EXPOSE 7860

# Command to run the application (assuming your main app file is app/main.py)
# Adjust "app.main:app" if your structure is different.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]