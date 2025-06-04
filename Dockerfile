FROM bitnami/spark:3.5

USER root

# Install Python3, pip, and OpenCV dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip libgl1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Optional: link python to python3 (if needed)
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY . .

# Set the default command to run your Spark job
CMD ["spark-submit", "--master", "local[*]", "hpc.py"]

