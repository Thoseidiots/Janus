FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN touch requirements.txt && pip install --no-cache-dir -r requirements.txt
COPY . .
ENTRYPOINT ["python", "-m", "trainer.task"]
