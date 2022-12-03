# Use Python37
FROM python:3.8.6-buster

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy requirements.txt to the docker image and install packages
COPY brief_news /brief_news
COPY requirements.txt /requirements.txt
COPY setup.py /setup.py
RUN pip install --upgrade pip
RUN pip install .

# Set the WORKDIR to be the folder
WORKDIR /brief_news

# Use uvicorn
CMD uvicorn interface.brief_news_api:app --host 0.0.0.0 --port $PORT
