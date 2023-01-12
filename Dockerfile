# 
FROM python:3.9

# 
WORKDIR /code

# Install dblib
#RUN yum update -y && \
#   yum install build-essential cmake pkg-config -y
##RUN yum update -y && yum install -y gcc gcc-c++
#RUN pip3 install cmake --target "${LAMBDA_TASK_ROOT}"
#RUN yum install boost-devel -y
#RUN yum install make -y
#RUN yum install libXext libSM libXrender -y
#RUN pip3 install dlib --target "${LAMBDA_TASK_ROOT}"



RUN apt-get update -y && \
    apt-get install build-essential cmake pkg-config -y

RUN apt-get install -y ffmpeg

RUN pip install dlib==19.18.0

#RUN apt-get install -y cmake


COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install PyYAML==5.4.1 
#RUN pip install PyYAML==3.13
RUN pip install -r /code/requirements.txt

# 
COPY ./app /code/app

# 
ENV PORT 80
ENV HOST 0.0.0.0

EXPOSE 80:80
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

