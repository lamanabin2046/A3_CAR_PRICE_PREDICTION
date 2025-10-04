FROM python:3.10.12-bookworm

WORKDIR /root/code

# Copy requirements first to leverage caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set PYTHONPATH
ENV PYTHONPATH="/root/code:${PYTHONPATH}"

# Copy source code
COPY ./code /root/code

# Expose Dash port
EXPOSE 8060

# Start Dash app
CMD ["python3", "/root/code/app.py"]
