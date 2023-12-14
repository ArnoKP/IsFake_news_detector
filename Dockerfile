FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

CMD ["gunicorn", "-b", "0.0.0.0:7860", "main:app"]

RUN bash -c 'mkdir -p /code/{nltk}'

ENV NLTK_DATA /code/nltk

RUN python -m nltk.downloader -d /code/nltk stopwords punkt

