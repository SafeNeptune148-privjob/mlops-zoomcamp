FROM continuumio/miniconda3

WORKDIR /app 

COPY environment.yml .

RUN conda env create -f environment.yml 
SHELL ["conda", "run", "-n", "dep_webService", "/bin/bash", "-c"] 
COPY predict.py lin_reg.bin ./

EXPOSE 9696 

CMD ["conda", "run", "-n", "dep_webService", "gunicorn", "predict:app", "--bind", "0.0.0.0:9696", "predict:app"]

