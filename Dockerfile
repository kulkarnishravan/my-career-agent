# Use official lightweight Python runtime
FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr and configure Gradio
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME="0.0.0.0" \
    PORT=7860

WORKDIR /app

# Install system dependencies
# Note: libgomp1 is strictly required for faiss-cpu OpenMP multithreading
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies first to take advantage of Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files (includes app.py, data/, and faiss_index/)
COPY . .

# Expose the internal port
EXPOSE 7860

# Start Gradio application
CMD ["python", "app.py"]
