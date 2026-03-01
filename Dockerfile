# 1. 改用完整版 Image (雖然大一點，但能解決 99% 的 MediaPipe 環境問題)
FROM python:3.11-bookworm

WORKDIR /app

# 2. 安裝所有 MediaPipe 與 OpenCV 運作所需的底層庫
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libgthread-2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxxf86vm1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. 確保 pip 是最新的，避免安裝到不相容的套件版本
RUN pip install --upgrade pip

# 4. 複製並安裝依賴 (增加逾時設定，避免下載 MediaPipe 時中斷)
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# 5. 複製程式碼
COPY . .

EXPOSE 5000

CMD ["python", "-u", "main.py"]