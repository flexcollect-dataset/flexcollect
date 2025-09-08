FROM public.ecr.aws/lambda/python:3.11

# Copy your libraries
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy your Lambda function
COPY src/ ./src/

COPY data/ ./data/

# Set the handler
CMD ["lambda_function.lambda_handler"]