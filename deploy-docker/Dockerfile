FROM python:3.9


# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app

CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0"]

#docker build -t mail-recsys .
#docker run -p 5000:5000  mail-recsys