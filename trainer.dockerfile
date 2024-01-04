# Base image
FROM python:3.10.10

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY ML_OPS_Project/ ML_OPS_Project/
COPY data/ data/
COPY reports/ reports/


# upgrade pip
RUN pip install --upgrade pip

# Set working directory and install requirements
# The --no-chache-dir flag is important to reduce the image size
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

# Naming the training script as the entrypoint for the docker image
# The "-u" here makes sure that any output from our script e.g. any print(...) statements gets redirected to our terminal
ENTRYPOINT ["python", "-u", "ML_OPS_Project/train_model.py"]