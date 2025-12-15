# Use an official lightweight Python image
FROM python:3.10-slim

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing pyc files to disc
# PYTHONUNBUFFERED: Ensures logs are flushed immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Point Hugging Face cache to a writable directory
    HF_HOME=/app/.cache/huggingface

# Set the working directory
WORKDIR /app

# Create a non-root user with a specific UID (1000) for security & HF compatibility
# and give them ownership of the /app directory
RUN useradd -m -u 1000 user && \
    chown -R user:user /app

# Switch to the non-root user
USER user

# Set up the PATH to include the user's local bin (where pip installs tools)
ENV PATH="/home/user/.local/bin:$PATH"

# Copy the requirements file first to leverage Docker cache
COPY --chown=user:user requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY --chown=user:user . .

# Expose port 7860 (Required for Hugging Face Spaces)
EXPOSE 7860

# Command to run the application
# Note: Ensure your main file is named 'main.py' and the app instance is 'app'
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]