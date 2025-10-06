# Dockerfile for VizClean AI (Streamlit)
FROM python:3.11-slim

# set workdir
WORKDIR /app

# system deps for plotting / excel
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# copy requirements and app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# expose streamlit port
EXPOSE 8501

ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION=false

# run streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.headless=true"]
