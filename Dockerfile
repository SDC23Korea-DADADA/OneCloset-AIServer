FROM python:3.10
  
WORKDIR /src

COPY . .

RUN pip install -r requirement.txt

RUN uvicorn main:app --host "0.0.0.0" --reload

EXPOSE 8000