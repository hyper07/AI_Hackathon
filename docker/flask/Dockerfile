FROM python:3.11-slim

WORKDIR /app

COPY ../../requirements/flask/requirements.txt requirements.txt

RUN pip install -r requirements.txt

# Set the FLASK_APP environment variable
ENV FLASK_APP=app.py

# No need to copy the application files, as they will be mounted as a volume

CMD ["flask", "run", "--host=0.0.0.0", "--port=5010"]
