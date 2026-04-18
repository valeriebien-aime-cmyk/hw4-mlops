# hw4 flask api container
# uses python 3.11 to match training environment exactly

FROM python:3.11-slim

# set working directory inside the container
WORKDIR /app

# copy requirements first, install deps
# this layer gets cached - rebuilds are fast as long as requirements don't change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy application code and model files
COPY app.py .
COPY valid_categories.json .
COPY model/ ./model/

# the port the app listens on inside the container
EXPOSE 5000

# run with gunicorn (production grade) not flask's dev server
# 2 workers is enough for this assignment, bind to all interfaces on port 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]