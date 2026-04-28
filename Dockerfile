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
COPY main.py .

COPY runs/rfdetr_crop2_medium_sliced/rfdetr_medium_converted.pth /app/models/rfdetr_medium.pth

EXPOSE 8000

CMD ["python", "main.py", "serve", "--model", "/app/models/rfdetr_medium.pth", "--host", "0.0.0.0", "--port", "8000", "--pipeline", "c", "--variant", "m"]
