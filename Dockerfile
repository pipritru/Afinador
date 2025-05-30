FROM python:3.13-slim

RUN apt-get update && apt-get install -y portaudio19-dev ffmpeg

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "afinador_digital.py", "--server.port=8501", "--server.address=0.0.0.0"]
