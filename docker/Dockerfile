FROM debian:stable-slim

RUN apt-get update \
 && apt-get install --no-install-recommends -y \
        python3-dev python3-pip python3-setuptools \
 && python3 -m pip install -U pip

COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt
