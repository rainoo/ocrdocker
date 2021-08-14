# Start FROM PyTorch image
FROM python:3.8

# Maintainer
MAINTAINER gaoyu<gaoyu@datatang.com>

# Create working directory
RUN mkdir -p /usr/src/app && mkdir /input && mkdir /result

# Copy requirements.txt
COPY requirements.txt /usr/src/app

WORKDIR /usr/src/app

RUN pip install --upgrade pip -i https://mirror.baidu.com/pypi/simple \
    && pip install paddlepaddle==2.1.0 -i https://mirror.baidu.com/pypi/simple \
    && pip install -r ./requirements.txt -i https://mirror.baidu.com/pypi/simple

# Copy contents
COPY . /usr/src/app

# exec detector
CMD ["python", "./tools/infer/predict_system.py"]

# docker build -t gaoyu/ocr_cpu:v1 .
# docker save -o /home/ocr_cpu-v1.tar gaoyu/ocr_cpu:v1
# docker run -d -v /pic/input:/input -v /pic/result:/result gaoyu/ocr_cpu:v1
