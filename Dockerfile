FROM python:3.8-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgtk2.0-dev \
    libboost-all-dev

RUN pip install --no-cache-dir \
    numpy \
    opencv-python \
    face_recognition

COPY . /app
WORKDIR /app

CMD ["python", "detector.py"]
