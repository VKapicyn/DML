FROM python:3.8-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
COPY . /
RUN pip3 install --no-cache-dir --user -r /requirements.txt
CMD [ "python", "server/http_server.py" ]
EXPOSE 8000