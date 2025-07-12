# Use Python 3.12 slim image
FROM python:3.13-alpine

# Set working directory
WORKDIR /app

# Install system dependencies for Alpine
RUN apk add --no-cache \
    gcc \
    musl-dev \
    g++ \
    libffi-dev \
    openssl-dev \
    curl

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit default port
EXPOSE 8501


# Run Streamlit app
CMD ["streamlit", "run", "streamlit.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
