FROM python:3.11-slim

WORKDIR /app

# 安裝 OpenCV + MediaPipe 需要的系統依賴
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libgthread-2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# 複製 requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製全部程式
COPY . .

EXPOSE 5000

CMD ["python", "-u", "main.py"]
