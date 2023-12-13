FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

CMD ["gunicorn", "-b", "0.0.0.0:7860", "main:app"]

RUN bash -c 'mkdir -p /code/{nltk}'

RUN nltk.download('stopwords', download_dir=/code/nltk)

RUN nltk.download('punkt', download_dir=/code/nltk)
