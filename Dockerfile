# Use the official Apache Airflow image as the base
FROM apache/airflow:2.7.0

# Set environment variables
ENV AIRFLOW_HOME=/usr/local/airflow
ENV PYTHONUNBUFFERED=1

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libhnswlib-dev \
    cmake \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN mkdir -p ${AIRFLOW_HOME}/logs && \
    chown -R airflow: ${AIRFLOW_HOME} && \
    chmod -R 755 ${AIRFLOW_HOME}

USER airflow

COPY requirements.txt .
RUN pip3 install --cache-dir=/var/tmp/ torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your DAG file into the DAGs folder
COPY dag/rag_pipeline.py ${AIRFLOW_HOME}/dags/

# Set the working directory
WORKDIR ${AIRFLOW_HOME}

# Start Airflow web server and scheduler
CMD ["bash", "-c", "airflow db init && airflow webserver & airflow scheduler"]

