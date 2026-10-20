FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PORT=7860 BIND_HOST=0.0.0.0
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY dataset/requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt
COPY --chown=app:app dataset/fmcw_dataset /app/dataset/fmcw_dataset
COPY --chown=app:app demo /app/demo
USER app
EXPOSE 7860
CMD ["python", "demo/server.py"]
