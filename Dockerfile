FROM python:3.8
RUN apt-get -y update
RUN apt-get install -y pip build-essential
COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

WORKDIR "/"


#ENTRYPOINT ["flask run --host=0.0.0.0"]
#ENTRYPOINT ["python3"]
CMD ["python3","-u","code/app.py"]