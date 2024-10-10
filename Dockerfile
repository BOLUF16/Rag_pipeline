FROM python:3.11.10

ENV PYTHONUNBUFFERED=1

COPY . /app/
WORKDIR /app/
RUN pip3 install --cache-dir=/var/tmp/ torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
ENV AIRFLOW_HOME="/app/airflow"
ENV AIRFLOW_CORE_DAGBAG_IMPORT_TIMEOUT=1000
ENV AIRFLOW_CORE_ENABLE_XCOM_PICKLING=True
RUN airflow db init
RUN airflow users create -e olumodejibolu@gmail.com -f bolu -l olumodeji -p admin -r Admin -u admin
RUN chmod 777 start.sh
RUN apt update -y
ENTRYPOINT [ "/bin/sh" ]
CMD ["start.sh"]
