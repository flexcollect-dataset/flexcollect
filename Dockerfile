FROM public.ecr.aws/lambda/python:3.11

# Speed up installs and avoid building from source whenever possible
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ONLY_BINARY=:all:

RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

# Adjust to your actual handler module.path
CMD ["src.lambda_function.lambda_handler"]
