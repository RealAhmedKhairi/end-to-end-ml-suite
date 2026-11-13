# 1. Start from a lightweight official Linux + Python 3.11 base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy requirements first (Docker caches this layer so rebuilds take 1 second!)
COPY requirements.txt .

# 4. Install Python dependencies and download NLTK stopwords
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m nltk.downloader stopwords

# 5. Copy your project folders (GUI and saved models) into the container
COPY . .

# 6. Expose Streamlit's default port
EXPOSE 8501

# 7. Default command to launch the Part D GUI when the container starts
CMD ["streamlit", "run", "gui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]