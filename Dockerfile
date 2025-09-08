FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies required for psycopg2/numpy/pandas
RUN yum install -y gcc libpq-devel python3-devel \
    && yum clean all

# Upgrade pip/setuptools/wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY data/ ./data/

# Set the handler (make sure src/lambda_function.py exists with lambda_handler)
CMD ["src/lambda_function.lambda_handler"]
