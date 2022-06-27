FROM python:3.9-slim

RUN pip install -U pip
RUN pip install pipenv



WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]
COPY [ "data", "model.bin","test.py","starter.py", "./"  ]

RUN pipenv install --system --deploy

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind==0.0.0.0:9696 ","predict:app"]
