FROM python:3

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

WORKDIR /app

ENV FLASK_APP backend.py

ENTRYPOINT [ "flask" ]
CMD [ "run" ]
