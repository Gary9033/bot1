# AI 健康評估後端服務

本專案是一個基於 Flask 開發的 AI 健康評估後端服務，旨在提供身體指標分析與健康問卷評估的功能。服務被打包成 Docker 镜像，便於部署與擴展。

## ✨ 主要功能

1.  **AI 圖像分析**:
    *   接收使用者全身照片，利用 **YOLOv8** 模型進行人體分割，分析輪廓。
    *   利用 **MediaPipe** 模型進行人體姿態估計，識別關鍵骨骼點。
    *   根據圖像分析結果，估算使用者的**身高**與**腰間距**（雙手食指間的距離）。

2.  **自動化問卷評分**:
    *   支援多種類型的線上健康問卷（認知功能、抑鬱情緒、營養狀況等）。
    *   根據預設規則自動計算問卷得分，並給出初步的評估等級（如“表現良好”、“需要加強”）。

3.  **數據持久化**:
    *   所有提交的問卷資料、使用者基本資料都會被分類存儲在伺服器上的 JSON 檔案中。
    *   分析後的圖像（帶有標記）會保存在 `uploads/` 目錄下。

## 🛠️ 技術棧

*   **後端框架**: Flask
*   **AI / 電腦視覺**:
    *   Ultralytics YOLOv8
    *   Google MediaPipe
    *   OpenCV
*   **容器化**: Docker
*   **語言**: Python 3.11

## 🚀 快速開始

### 1. 環境準備

請確保您的系統已安裝 Docker 和 Docker Compose。

### 2. 啟動服務

#### 方法一：使用原生 Docker 命令

**a. 構建 Docker 鏡像**

在專案根目錄下，執行以下命令來構建 Docker 镜像。此步驟會根據 `Dockerfile` 中的指令安裝所有作業系統層級和 Python 的依賴。

```bash
docker build -t nuwa_robot .
```

**b. 運行 Docker 容器**

構建成功後，執行以下命令來啟動服務。

```bash
# -d: 在背景運行容器
# -p 5000:5000: 將本機的 5000 端口映射到容器的 5000 端口
# --name nuwa_robot_container: 為容器命名，方便管理
docker run -d -p 5000:5000 --name nuwa_robot_container nuwa_robot
```

#### 方法二：使用 Docker Compose (推薦)

專案已包含 `docker-compose.yml` 檔案，它將啟動應用的所有配置都固化在檔案中，推薦使用此方法。

在專案根目錄下，執行以下單一命令即可啟動服務：

```bash
# -d 表示在背景 (detached mode) 運行
docker-compose up -d
```

若要停止並移除容器，請執行：

```bash
docker-compose down
```

### 3. 訪問服務

不論使用何種方法，服務啟動後，API 將在 `http://localhost:5000` 上提供服務。

## 📦 鏡像分發與共享

若您需要將此應用提供給他人，有以下幾種方式：

### 方式一：通過 Docker Hub (標準方式)

1.  登錄 Docker Hub: `docker login -u <您的用戶名>`
2.  為鏡像打上標籤: `docker tag nuwa_robot <您的用戶名>/nuwa_robot:latest`
3.  推送鏡像: `docker push <您的用戶名>/nuwa_robot:latest`
4.  對方通過 `docker pull <您的用戶名>/nuwa_robot:latest` 拉取鏡像。

### 方式二：導出為 TAR 檔案 (離線分享)

如果您不想使用線上倉庫，可以將鏡像打包成一個檔案。

1.  **在您的機器上，導出鏡像：**
    ```bash
    docker save -o nuwa_robot.tar nuwa_robot
    ```
2.  **在對方的機器上，加載鏡像：**
    對方收到 `nuwa_robot.tar` 檔案後，執行以下命令：
    ```bash
    docker load -i nuwa_robot.tar
    ```

### 方式三：共享原始碼

直接將整個專案資料夾打包發送給對方。對方在自己的環境中，使用 `docker build` 或 `docker-compose up --build` 即可構建並運行應用。

## 📖 API 使用說明

所有功能都通過向根目錄 `/` 發送 `POST` 請求來觸發，伺服器根據請求的 JSON 內容來決定執行哪個任務。

### 端點: `POST /`

#### 1. 身高與腰間距估算

**Request Body**:

```json
{
  "pic1": "<Base64 Encoded String of the Image>"
}
```

*   `pic1`: 使用者全身照片的 Base64 編碼字串。

**Success Response**:

```json
{
  "height": "175.3",
  "waist_distance": "30.1",
  "message": "success"
}
```

*   `height`: 估算的身高（公分）。
*   `waist_distance`: 估算的腰間距（公分）。

#### 2. 問卷評估

**Request Body**:

```json
{
  "Questionnaire_type": "3",
  "user": "test_user",
  "life_satisfaction": "1. 非常滿意",
  "boredom": "2. 否",
  "...": "..."
}
```

*   `Questionnaire_type`: 問卷類型編號（例如 "3" 代表抑鬱情緒問卷）。
*   其他鍵值對為問卷的題目與答案。

**Success Response**:

```json
{
  "level": "需要注意",
  "score": 7
}
```

*   `level`: 評估等級。
*   `score`: 計算出的總分。

#### 3. 儲存使用者資料

**Request Body**:

```json
{
  "save_user": true,
  "user_id": "user123",
  "name": "John Doe",
  "...": "..."
}
```

*   `save_user`: 觸發保存使用者資料的標誌。

**Success Response**:

```json
{
  "status": "success",
  "message": "stored user_info successfully!"
}
```
