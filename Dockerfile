FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY cli/ cli/
COPY config/ config/
COPY core/ core/
COPY pipeline/ pipeline/
COPY viewers/ viewers/
COPY main.py .

COPY models/pipeline_c_rfdetr_medium_best_ema.pth /app/models/pipeline_c_rfdetr_medium_best_ema.pth

EXPOSE 8000

CMD ["python", "main.py", "serve", "--pipeline", "c", "--variant", "m", "--model", "/app/models/pipeline_c_rfdetr_medium_best_ema.pth", "--host", "0.0.0.0", "--port", "8000"]
