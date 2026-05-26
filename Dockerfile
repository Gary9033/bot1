FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libegl1 \
    libgles2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- 這裡加入環境變數 ---
# 0 = ALL (預設), 1 = INFO, 2 = WARNING, 3 = ERROR
# 設定為 2 可以過濾掉你看到的那些 GPU/Feedback 警告訊息
ENV GLOG_minloglevel=2
# 也可以順便告訴 TensorFlow 不要去吵著找 GPU
ENV CUDA_VISIBLE_DEVICES=-1

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:5000", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "main:app"]