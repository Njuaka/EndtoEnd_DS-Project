FROM python:3.11
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8080
CMD ["gunicorn", "--workers=4", "--bind", "0.0.0.0:$PORT", "app:app"]