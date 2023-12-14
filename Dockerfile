FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

CMD ["gunicorn", "-b", "0.0.0.0:7860", "main:app"]

RUN bash -c 'mkdir -p /code/{nltk}'

#RUN python -m nltk.downloader -d /code/nltk stopwords punkt
RUN python -m nltk.download('punkt')
#RUN bash -c 'chmod o+w /code/nltk'
RUN bash -c 'chmod o+w /nltk_data'
#USER root
#RUN mkdir -p /nltk_data
#RUN chmod -R 777 /nltk_data

#RUN python -m nltk.download('punkt')

