# Base Python image
FROM python:3.10-slim

# Workdir inside container
WORKDIR /app

# System deps (optional but useful)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy dependency file and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your repo
COPY . .

# Streamlit will listen on port 8080
EXPOSE 8080

# Env vars for Streamlit + Vertex
ENV PORT=8080 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    PROJECT_ID=mlops-project-479512 \
    LOCATION=europe-west3 \
    ENDPOINT_ID=6025044444259024896

# Start Streamlit with your existing app
CMD ["sh", "-c", "streamlit run App.py --server.port=$PORT --server.address=0.0.0.0"]