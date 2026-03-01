from flask import Flask, request, jsonify, send_file, send_from_directory
from PIL import Image, ImageDraw
import base64, os, cv2, numpy as np, uuid
import mediapipe as mp
from mediapipe.tasks import python          
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.framework.formats import landmark_pb2
import re,json,datetime

app = Flask(__name__)


TARGET_LANDMARKS = {
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_ELBOW", "RIGHT_ELBOW",  # 如果需要手肘連線可以保留
    "LEFT_INDEX", "RIGHT_INDEX"
}
POSE_LANDMARK_NAMES = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER", "LEFT_EAR", "RIGHT_EAR",
    "MOUTH_LEFT", "MOUTH_RIGHT", "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX", "RIGHT_INDEX",
    "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
]


def media_pixel_to_height(h_px, f_mm=1.42903078, pixel_size_um=1.12, Z_m=3.6):
    f = f_mm / 1000.0          # mm -> m
    p = pixel_size_um * 1e-6   # um -> m
    H = ((h_px * p * Z_m) / f) * 100 -3
    return H
def media_pixel_to_height_new(h_px, f_mm=1.005988, pixel_size_um=1.12, Z_m=2.4):
    f = f_mm / 1000.0          # mm -> m
    p = pixel_size_um * 1e-6   # um -> m
    H = ((h_px * p * Z_m) / f) * 100 
    return H

def draw_selected_landmarks(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    selected_pixels = []

    for idx, pose_landmarks in enumerate(pose_landmarks_list):
        h, w, _ = annotated_image.shape
        pixel_coords = []

        # 遍歷每個關節
        for i, landmark in enumerate(pose_landmarks):
            name = POSE_LANDMARK_NAMES[i]
            if name in TARGET_LANDMARKS:
                x_px = int(landmark.x * w)
                y_px = int(landmark.y * h)
                pixel_coords.append((name, x_px, y_px))
                # 畫節點
                cv2.circle(annotated_image, (x_px, y_px), 5, (0,255,255), -1)

        selected_pixels.append(pixel_coords)

        # 畫連線（可選：只連肩膀到手、肩膀到臀部）
        connections = [
            # ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
            # ("LEFT_SHOULDER", "LEFT_ELBOW"),
            # ("LEFT_ELBOW", "LEFT_WRIST"),
            # ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
            # ("RIGHT_ELBOW", "RIGHT_WRIST"),
            # ("LEFT_SHOULDER", "LEFT_HIP"),
            # ("RIGHT_SHOULDER", "RIGHT_HIP"),
            # ("LEFT_HIP", "RIGHT_HIP"),
            ("LEFT_INDEX", "RIGHT_INDEX") 
        ]
        for start_name, end_name in connections:
            start = next(((x,y) for n,x,y in pixel_coords if n==start_name), None)
            end = next(((x,y) for n,x,y in pixel_coords if n==end_name), None)
            if start and end:
                cv2.line(annotated_image, start, end, (255,255,0), 2)

    return annotated_image, selected_pixels

def mediapipe(file_path):
    base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    # 載入圖片
    image = mp.Image.create_from_file(file_path)
    # image = mp.Image.create_from_file("image1.jpg")

    # 偵測輪廓
    detection_result = detector.detect(image)
    img_array = image.numpy_view()
    if detection_result.segmentation_masks:
        mask = detection_result.segmentation_masks[0].numpy_view()
        # 計算頭到腳的 pixel 高度
        y_indices, x_indices = np.where(mask > 0.5)
        if len(y_indices) > 0:
            top_y = np.min(y_indices)
            bottom_y = np.max(y_indices)
            height_pixels = bottom_y - top_y
            print(f"頭到腳的 pixel 高度: {height_pixels} 像素")
        else:
            height_pixels = 0
            print("未偵測到人體區域，請確認影像中有人物。")
        
        # 用透明藍色疊加在原圖
        img_array = image.numpy_view()  # 原圖
        color_mask = np.zeros_like(img_array, dtype=np.uint8)
        color_mask[mask > 0.5] = (0, 0, 255)  # 藍色
        alpha = 0.5
        overlay = cv2.addWeighted(img_array, 1.0, color_mask, alpha, 0)
        # 顯示或存檔
        h, w, _ = overlay.shape
        center_x = w // 2  # 取影像水平中間作為標記點的 x
        cv2.circle(overlay, (center_x, top_y), 8, (0, 255, 255), -1)    # 黃色圓點
        cv2.circle(overlay, (center_x, bottom_y), 8, (0, 255, 255), -1) # 黃色圓點
    
        text1 = f"Height: {height_pixels}px"
        text2 = f"Height: {media_pixel_to_height(height_pixels):.1f}cm"

        # 繪製第一個文字 (text1) - 改為紅色 (0, 0, 255)
        cv2.putText(overlay, text1, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # 繪製第二個文字 (text2) - 改為紅色 (0, 0, 255)
        cv2.putText(overlay, text2, (20, 80),  # y 軸往下 40px
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        out_path = os.path.splitext(file_path)[0] + "_overlay.png"

        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    # 視覺化 + 取得 pixel 座標
    annotated_image, all_landmark_pixels = draw_selected_landmarks(image.numpy_view(), detection_result)
    left_index = next(((x, y) for n, x, y in all_landmark_pixels[0] if n == "LEFT_INDEX"), None)
    right_index = next(((x, y) for n, x, y in all_landmark_pixels[0] if n == "RIGHT_INDEX"), None)
    hand = 0 
    if left_index and right_index:
        lx, ly = left_index
        rx, ry = right_index
        # 計算歐氏距離
        hand = round(((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5, 2)
    print(f"左右食指的距離: {hand} 像素")
    return hand,height_pixels


def height(data):
    try:
        b64_str = data.get("pic1")
        if not b64_str:
            return jsonify({"error": "pic1 not found"}), 400

        #  去掉 data:image/...;base64,
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]

        img_bytes = base64.b64decode(b64_str)

    except Exception as e:
        return jsonify({"error": "Invalid JSON or base64", "detail": str(e)}), 400

    os.makedirs("uploads", exist_ok=True)
    image_path = f"uploads/{uuid.uuid4().hex}.jpg"
    with open(image_path, "wb") as f:
        f.write(img_bytes)
    print("saved image size:", os.path.getsize(image_path))
    
    hand_px,height_px= mediapipe(image_path)
    dis = media_pixel_to_height(hand_px)
    Media_H_m = media_pixel_to_height(height_px)

    print(f"waist_distance = {dis:.1f} cm")
    print(f"Media_H_m = {Media_H_m:.1f} cm")

    return jsonify({
        "message": "success",
        "height": f"{Media_H_m:.1f}",
        "waist_distance": f"{dis:.1f}",
    })


def height_new(data):
    try:
        b64_str = data.get("pic2")
        if not b64_str:
            return jsonify({"error": "pic2 not found"}), 400

        #  去掉 data:image/...;base64,
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]

        img_bytes = base64.b64decode(b64_str)

    except Exception as e:
        return jsonify({"error": "Invalid JSON or base64", "detail": str(e)}), 400

    os.makedirs("uploads", exist_ok=True)
    image_path = f"uploads/{uuid.uuid4().hex}.jpg"
    with open(image_path, "wb") as f:
        f.write(img_bytes)
    print("saved image size:", os.path.getsize(image_path))
    
    hand_px,height_px= mediapipe(image_path)
    dis = media_pixel_to_height_new(hand_px)
    Media_H_m = media_pixel_to_height_new(height_px)

    print(f"waist_distance = {dis:.1f} cm")
    print(f"Media_H_m = {Media_H_m:.1f} cm")

    return jsonify({
        "message": "success",
        "height": f"{Media_H_m:.1f}",
        "waist_distance": f"{dis:.1f}",
    })


def temperature(data):
    try:
        print(data)
    except Exception as e:
        return jsonify({"error": "Invalid JSON or base64", "detail": str(e)}), 400
    return "success"

def questionnaire(data):
    try:
        print("問卷算分")
        summation = 0
        string =""

        match data.get('Questionnaire_type')[0]:
            case '1':                
                questionnaire_type = "認知功能問卷"
                print(questionnaire_type)     
                exclude_keys = {"start_timestamp", "user", "machine_ID_1", "Questionnaire_type"}
                for key, value in data.items():
                    if key in exclude_keys:
                        continue            
                    value_str = str(value) # 確保 value 是字串，然後取第一個字元
                    if value[0] == '1':
                        summation += 1
                print("總分:", summation)   
                if  2 <= summation <= 8:
                    string ="需要加強"
                    print("需要加強")
                elif summation == 1:
                    string ="需要注意"
                    print("需要注意") 
                elif summation == 0:
                    string ="表現良好"
                    print("表現良好")

            case '2':
                questionnaire_type = "視力健康問卷"
                print(questionnaire_type)                
                exclude_keys = {"start_timestamp", "user", "machine_ID_1", "Questionnaire_type"}
                for key, value in data.items():
                    if key in exclude_keys:
                        continue            
                    value_str = str(value) # 確保 value 是字串，然後取第一個字元
                    if value[0] == '1':
                        summation += 1
                print("總分:", summation)  
                if  1 <= summation <= 2:
                    string ="需要加強"
                    print("需要加強")
                elif summation == 0:
                    string ="表現良好"
                    print("表現良好")
            case '3':
                questionnaire_type = "抑鬱情緒問卷"
                print(questionnaire_type)
                score_rules = {
                    "life_satisfaction": {"1": 0, "2": 1},
                    "boredom": {"1": 1, "2": 0},
                    "hopeless": {"1": 1, "2": 0},
                    "home_preference": {"1": 1, "2": 0},
                    "worthless": {"1": 1, "2": 0},
                    "activity_loss": {"1": 1, "2": 0},
                    "empty": {"1": 1, "2": 0},
                    "good_spirit": {"1": 0, "2": 1},
                    "anxious": {"1": 1, "2": 0},
                    "happy": {"1": 0, "2": 1},
                    "memory_concern": {"1": 1, "2": 0},
                    "grateful_alive": {"1": 0, "2": 1},
                    "energetic": {"1": 0, "2": 1},
                    "no_hope": {"1": 1, "2": 0},
                    "envy_others": {"1": 1, "2": 0}
                }
                exclude_keys = {"start_timestamp", "user", "machine_ID_1", "Questionnaire_type"}
                details={}
                for key, value in data.items():
                    if key in exclude_keys:
                        continue
                    value_str = str(value)
                    number = value_str.split(".")[0]
                    score = score_rules.get(key, {}).get(number, 0)
                    details[key] = score
                    summation += score

                print("總分:", summation)
                print("得分:", details)

                if  10 <= summation <= 15:
                    string ="需要加強"
                    print("需要加強")
                elif 6 <= summation <= 9:
                    string ="需要注意"
                    print("需要注意") 
                else:
                    string ="表現良好"
                    print("表現良好")

            case '4':
                questionnaire_type = "營養問卷"
                print(questionnaire_type)
                score_rules = {
                    "appetite":  {"1": 0, "2": 1, "3": 2},
                    "weight_change":  {"1": 0, "2": 1, "3": 2, "4": 3},
                    "mobility":  {"1": 0, "2": 1, "3": 2},
                    "stress_health":  {"1": 0, "2": 2},
                    "mood_memory":  {"1": 1, "2": 2},
                }
                exclude_keys = {"start_timestamp", "user", "machine_ID_1", "Questionnaire_type"}
                details={}            
                for key, value in data.items():
                    if key in exclude_keys:
                        continue
                    if key == "bmi":
                        if not value:
                            continue
                        bmi_value = float(value)
                        if bmi_value < 19:
                            score = 0
                        elif 19 <= bmi_value < 21:
                            score = 1
                        elif 21 <= bmi_value < 23:
                            score = 2
                        elif bmi_value >= 23:
                            score = 3
                        details[key] = score
                        summation += score
                        continue
                    value_str = str(value)
                    number = value_str.split(".")[0]
                    score = score_rules.get(key, {}).get(number, 0)
                    details[key] = score
                    summation += score

                print("總分:", summation)
                print("得分:", details)

                if  12 <= summation <= 14:
                    string ="表現良好"
                    print("表現良好")
                else:
                    string ="需要加強"
                    print("需要加強")
         
            case '5':
                questionnaire_type = "支持評估問卷"
                print(questionnaire_type)
                exclude_keys = {"start_timestamp", "user", "machine_ID_1", "Questionnaire_type"}
                for key, value in data.items():
                    if key in exclude_keys:
                        continue            
                    value_str = str(value) # 確保 value 是字串，然後取第一個字元
                    if value[0] == '1':
                        summation += 1
                print("總分:", summation)  
                if  1 <= summation <= 6:
                    string ="需要加強"
                    print("需要加強")
                elif summation == 0:
                    string ="表現良好"
                    print("表現良好")
                    
            case '6':
                questionnaire_type = "視力"
                exclude_keys = {"start_timestamp", "user", "machine_ID_1", "Questionnaire_type"}
                scores = {}
                scores['near_distance'] = 1 if data['near_distance'] == "通過" else 0
                scores['far_distance'] = 1 if data['far_distance'] == "通過" else 0
                # 計算總分
                print("各項分數:", scores)
                summation = sum(scores.values())
                print("總分:", summation)  
                if  1 <= summation <= 2:
                    string ="表現良好"
                    print("表現良好")
                elif summation == 0:
                    string ="需要加強"
                    print("需要加強")

            case '7':
                questionnaire_type = "行動"
                exclude_keys = {"start_timestamp", "user", "machine_ID_1", "Questionnaire_type","mobility_start_time","mobility_end_time"}
                
                time_diff = int(data["mobility_end_time"]) - int(data["mobility_start_time"])
                time_diff = time_diff / 1000.0  # 毫秒轉秒
                print(f"走路時間差: {time_diff} 秒")
                time_diff1 = int(data["mobility_end_time1"]) - int(data["mobility_start_time1"])
                time_diff1 = time_diff1 / 1000.0  # 毫秒轉秒
                print(f"走路時間差: {time_diff1} 秒")
                scores = {}
                # 1. Side-by-side stance (保持10秒=1, 少於10秒=0)
                scores['side_by_side_score'] = 1 if data['side_by_side_stance'] == "保持10秒" else 0

                # 2. Semi-tandem stance (保持10秒=1, 少於10秒=0)
                scores['semi_tandem_score'] = 1 if data['semi_tandem_stance'] == "保持10秒" else 0

                # 3. Tandem stance (保持10秒=2, 3-9.99秒=1, <3秒=0)
                if data['tandem_stance'] == "保持10秒":
                    scores['tandem_score'] = 2
                elif data['tandem_stance'] == "保持3-9.99秒":
                    scores['tandem_score'] = 1
                else:
                    scores['tandem_score'] = 0

                # 4. Gait speed test (走路速度，越快分越高)
                # <11.19=4, 11.2-13.69=3, 13.7-16.69=2, 16.7-59.9=1, 60或無法完成=0
                if time_diff is None or time_diff >= 60:
                    scores['chair_stand_score'] = 0
                elif time_diff < 11.19:
                    scores['chair_stand_score'] = 4
                elif 11.19 <= time_diff <= 13.69:
                    scores['chair_stand_score'] = 3
                elif 13.7 <= time_diff <= 16.69:
                    scores['chair_stand_score'] = 2
                elif 16.7 <= time_diff < 60:
                    scores['chair_stand_score'] = 1

                # 5. Chair stand test (起身測試，越快分越高)
                # <3.62=4, 3.62-4.65=3, 4.66-6.52=2, >6.52=1, 無法完成=0
                if time_diff1 is None: # 無法完成
                    scores['gait_speed_score'] = 0
                elif time_diff1 < 3.62:
                    scores['gait_speed_score'] = 4
                elif 3.62 <= time_diff1 <= 4.65:
                    scores['gait_speed_score'] = 3
                elif 4.66 <= time_diff1 <= 6.52:
                    scores['gait_speed_score'] = 2
                else: # 雖然圖上寫 6.52=1，邏輯上應為 >6.52
                    scores['gait_speed_score'] = 1
                # 計算總分
                print("各項分數:", scores)
                summation = sum(scores.values())
                print("總分:", summation)  
                if  0 <= summation <= 9:
                    string ="行動能力障礙"
                    print("行動能力障礙")
                elif summation >=10:
                    string ="行動能力正常"
                    print("行動能力正常")
            
        ##存檔
        SAVE_FILE = questionnaire_type + ".json"  
        if os.path.exists(SAVE_FILE):
            with open(SAVE_FILE, "r", encoding="utf-8") as f:
                try:
                    all_data = json.load(f)
                    if not isinstance(all_data, list):
                        all_data = [all_data]  # 如果原本不是 list，就轉成 list
                except json.JSONDecodeError:
                    all_data = []
        else:
            all_data = []
        all_data.append(data)
        with open(SAVE_FILE, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)
        return jsonify({
            "level":string,
            "score": summation,
        })
    
    except Exception as e:
        return jsonify({"error": "Invalid JSON or base64", "detail": str(e)}), 400

@app.route("/", methods=["POST"])
def home():
    data = request.get_json(force=True)
    if 'start_timestamp' in data:
        raw_ts = data["start_timestamp"]
        now = datetime.datetime.now()
        if raw_ts in (None, "null", "", "None"):
            data["start_timestamp"] = now.strftime("%Y-%m-%d %H:%M:%S")
        else:
            try:
                ts_ms = int(raw_ts)  # "1764758552746" → 1764758552746
                ts_s = ts_ms / 1000.0
                dt = datetime.datetime.fromtimestamp(ts_s)
                data["start_timestamp"] = dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                # 若不是數字，也不報錯 → 改成 None
                data["start_timestamp"] = now.strftime("%Y-%m-%d %H:%M:%S")

    # ---------- 儲存原始資料 ----------
    SAVE_FILE = "pic.json"
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, "r", encoding="utf-8") as f:
            try:
                all_data = json.load(f)
                if not isinstance(all_data, list):
                    all_data = [all_data]
            except json.JSONDecodeError:
                all_data = []
    else:
        all_data = []

    all_data.append(data)
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)

    if 'pic1' in data:
        return height(data)
    if 'pic2' in data:
        return height_new(data)
    elif 'temp' in data:
        return temperature(data)
    elif 'Questionnaire_type' in data: 
        return questionnaire(data)
    elif 'stress' in data: ##簡單除小數點
        raw_stress = data.get('stress')
        if raw_stress in (None, "null", "", "None"):
            stress = None
        else:
            try:
                stress = round(float(raw_stress), 1)
            except (TypeError, ValueError):
                stress = None

        raw_sdnn = data.get('sdnn')
        if raw_sdnn in (None, "null", "", "None"):
            sdnn = None
        else:
            try:
                sdnn = round(float(raw_sdnn), 1)
            except (TypeError, ValueError):
                sdnn = None

        return jsonify({
           "stress":stress,
           "sdnn":sdnn
        })
    elif 'blood_status_string' in data:
        print("轉換血壓資料型態")
        blood_status_string = data.get('blood_status_string')
        print("原始血壓字串:", blood_status_string)
        blood_status = int(blood_status_string[0])-1        
        print("血壓狀態:",blood_status)   
        return jsonify({"blood_status": blood_status})
    elif 'gender_string' in data:
        print("轉性別資料型態")
        gender = 1 # 預設值，若無法解析則視為男生
        gender_string = data.get('gender_string')
        print("原始性別字串:", gender_string)
        if gender_string == "男生":
            gender = 1
        elif gender_string == "女生":
            gender = 0
        return jsonify({"gender": gender})
    elif 'title_1' in data:
        print("存使用者資料")
        SAVE_FILE = "user_data.json"
        # 取得對方送來的 JSON
        if os.path.exists(SAVE_FILE):
            with open(SAVE_FILE, "r", encoding="utf-8") as f:
                try:
                    all_data = json.load(f)
                    if not isinstance(all_data, list):
                        all_data = [all_data]  # 如果原本不是 list，就轉成 list
                except json.JSONDecodeError:
                    all_data = []
        else:
            all_data = []
        all_data.append(data)

        with open(SAVE_FILE, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)
        return jsonify({"status": "success", "message": "stored user_info successfully!"})
    elif 'user2'in data:
        print("去除身份證字號")
        s = data.get('user2')
        print("原始字串:", s)
        pattern = r"^\s*(.*?)\s*[\(（]\s*(.*?)\s*[\)）]\s*$"
        m = re.match(pattern, s)
        if m:
            name = m.group(1)
            id_num = m.group(2)
            print("姓名:", name)
            print("身份證字號:", id_num)
        return jsonify({"user2": name, "id_num": id_num})
    elif 'weight'in data:
        weight = data.get('weight')
        height_ = data.get('height')
        if weight is not None:
            try:
                if weight < 0:
                    return jsonify({"error": "Invalid weight value"}), 400
                weight = float(weight)
                bmi = float(weight) / ((float(height_)/100) ** 2)
                bmi = round(bmi, 1)
            except (ValueError, TypeError):
                return jsonify({"error": "Invalid weight value"}), 400
        return jsonify({"bmi": bmi})
    else:
        return jsonify({"error": "Invalid request"}), 400


def questionnaire_all(data):
    score = {
        "認知": 0,
        "視力問卷": 0,
        "情緒": 0,
        "營養": 0,
        "支持": 0,
        "視力": 0,
        "行動": 0
    }

    # ---------- 儲存原始資料 ----------
    SAVE_FILE = "pic.json"
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, "r", encoding="utf-8") as f:
            try:
                all_data = json.load(f)
                if not isinstance(all_data, list):
                    all_data = [all_data]
            except json.JSONDecodeError:
                all_data = []
    else:
        all_data = []

    all_data.append(data)
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)

    # ---------- 認知 ----------
    cognition_keys = {
        "decision", "hobbies", "repetitions", "house_machine",
        "what_date", "finicial_manage", "forget_date", "forget_thing"
    }
    for k in cognition_keys:
        val = data.get(k, "")
        if val == "":
            score["認知"] = -1
            break
        if str(val)[0] == "1":
            score["認知"] += 1

    # ---------- 視力問卷 ----------
    vision_keys = {"diabetes", "eyecheck_diabetic", "eyeproblem_nondiabetic"}
    for k in vision_keys:
        val = data.get(k, "")
        if val == "":
            score["視力問卷"] = -1
            break
        if str(val)[0] == "1":
            score["視力問卷"] += 1

    # ---------- 情緒 ----------
    emotion_rules = {
        "life_satisfaction": {"1": 0, "2": 1},
        "boredom": {"1": 1, "2": 0},
        "hopeless": {"1": 1, "2": 0},
        "home_preference": {"1": 1, "2": 0},
        "worthless": {"1": 1, "2": 0},
        "activity_loss": {"1": 1, "2": 0},
        "empty": {"1": 1, "2": 0},
        "good_spirit": {"1": 0, "2": 1},
        "anxious": {"1": 1, "2": 0},
        "happy": {"1": 0, "2": 1},
        "memory_concern": {"1": 1, "2": 0},
        "grateful_alive": {"1": 0, "2": 1},
        "energetic": {"1": 0, "2": 1},
        "no_hope": {"1": 1, "2": 0},
        "envy_others": {"1": 1, "2": 0}
    }
    for k, rule in emotion_rules.items():
        val = data.get(k, "")
        if val == "":
            score["情緒"] = -1
            break
        key = str(val).split(".")[0]
        score["情緒"] += rule.get(key, 0)

    # ---------- 營養 ----------
    nutrition_rules = {
        "appetite": {"1": 0, "2": 1, "3": 2},
        "weight_change": {"1": 0, "2": 1, "3": 2, "4": 3},
        "mobility": {"1": 0, "2": 1, "3": 2},
        "stress_health": {"1": 0, "2": 2},
        "mood_memory": {"1": 1, "2": 2},
    }
    bmi = data.get("bmi", "")
    if not bmi:
        score_ = 0
    else:
        try:            
            bmi = float(bmi)
            if bmi < 19:    
                score_ = 0
            elif 19 <= bmi < 21:
                score_ = 1
            elif 21 <= bmi < 23:
                score_ = 2
            elif bmi >= 23:
                score_ = 3
            score["營養"] += score_
        except (ValueError, TypeError):
            score_ = 0
       

    for k, rule in nutrition_rules.items():
        val = data.get(k, "")
        if val == "":
            score["營養"] = -1
            break
        key = str(val).split(".")[0]
        score["營養"] += rule.get(key, 0)

    # ---------- 支持 ----------
    support_keys = {
        "polypharmacy", "pain_sleep_med", "med_sideeffect",
        "daily_activity", "env_finance", "social_loneliness"
    }
    for k in support_keys:
        val = data.get(k, "")
        if val == "":
            score["支持"] = -1
            break
        if str(val)[0] == "1":
            score["支持"] += 1

    # ---------- 視力測試 ----------
    vision_test_keys = {"near_distance", "far_distance"}
    for k in vision_test_keys:
        val = data.get(k, "")
        if val == "":
            score["視力"] = -1
            break
        if str(val).strip() == "通過":
            score["視力"] += 1

    # ---------- 行動：時間計算 ----------
    time_diff = None
    time_diff1 = None
    try:
        if data.get("mobility_end_time") != "" and data.get("mobility_start_time") != "":
            time_diff = (int(data.get("mobility_end_time")) - int(data.get("mobility_start_time"))) / 1000.0
    except (TypeError, ValueError):
        pass

    try:
        if data.get("mobility_end_time1") != "" and data.get("mobility_start_time1") != "":
            time_diff1 = (int(data.get("mobility_end_time1")) - int(data.get("mobility_start_time1"))) / 1000.0
    except (TypeError, ValueError):
        pass

    # ---------- 行動：平衡 ----------
    mobility_scores = {
        "side_by_side": 1 if data.get("side_by_side_stance", "") == "保持10秒" else 0,
        "semi_tandem": 1 if data.get("semi_tandem_stance", "") == "保持10秒" else 0,
        "tandem": (
            2 if data.get("tandem_stance", "") == "保持10秒"
            else 1 if data.get("tandem_stance", "") == "保持3-9.99秒"
            else 0
        )
    }
    balance_score = sum(mobility_scores.values())

    # ---------- 行動：起立 ----------
   # 起立與步態計算 (確保 None 處理)
    if time_diff is None or time_diff1 is None:
        score["行動"] = -1
    else:
        # 起立分
        if time_diff < 11.19: chair_score = 4
        elif time_diff <= 13.69: chair_score = 3
        elif time_diff <= 16.69: chair_score = 2
        elif time_diff < 60: chair_score = 1
        else: chair_score = 0 # >= 60秒
        
        # 步態分
        if time_diff1 < 3.62: gait_score = 4
        elif time_diff1 <= 4.65: gait_score = 3
        elif time_diff1 <= 6.52: gait_score = 2
        else: gait_score = 1
        
        score["行動"] = balance_score + chair_score + gait_score

    print("score:", score)
    return score


def calculate_generic(score_value, rise_dict):
    values = list(rise_dict.values())
    keys = list(rise_dict.keys())

    if score_value >= values[0] and score_value < values[1]:
        key = keys[0]
    elif len(values) > 2 and score_value >= values[1] and score_value < values[2]:
        key = keys[1]
    else:
        key = keys[-2]

    idx = keys.index(key)
    total = values[idx+1] - values[idx]
    sub = score_value - values[idx]
    percent = round(sub / total, 2)
    return key, percent


@app.route('/image', methods=['GET','POST'])
def show_image():
    if request.method == 'GET':
        id_num_1 = request.args['id_num']
        result_dir = os.path.join(app.root_path, 'result')
        return send_from_directory(result_dir, f"{id_num_1}.png")
    
    if request.method == 'POST':
        data = request.get_json(force=True)
        score = questionnaire_all(data)

        認知_rise = dict(sorted({"interval_improve":2,"interval_notice":1,"interval_good":0,"interval_max":8}.items(), key=lambda x:x[1]))
        視力問卷_rise = dict(sorted({"interval_improve":1,"interval_good":0,"interval_max":2}.items(), key=lambda x:x[1]))
        視力_rise = dict(sorted({"interval_improve":0,"interval_good":1,"interval_max":2}.items(), key=lambda x:x[1]))
        情緒_rise = dict(sorted({"interval_improve":10,"interval_notice":6,"interval_good":0,"interval_max":15}.items(), key=lambda x:x[1]))
        營養_rise = dict(sorted({"interval_improve":0,"interval_good":12,"interval_max":14}.items(), key=lambda x:x[1]))
        支持_rise = dict(sorted({"interval_improve":1,"interval_good":0,"interval_max":6}.items(), key=lambda x:x[1]))
        行動_rise = dict(sorted({"interval_improve":0,"interval_good":10,"interval_max":12}.items(), key=lambda x:x[1]))
        
        # 打開原圖
        img = Image.open("statistics.png").convert("RGBA")

        # 原圖尺寸
        orig_width, orig_height = img.size  # 1980, 1160

        # 起點座標（基於原圖）
        認知_xy = {"x": 988,"y1":529,"y2":467}
        視力_xy = {"y":529,"x2":1048}
        情緒_xy = {"x":988,"y1":529,"y2":589}
        支持_xy = {"y":529,"x2":928}

        coords = []
        # 認知
        if score["認知"] == -1:
            coords.append((認知_xy["x"], 認知_xy["y1"]))  # 起點
        else:
            key, p = calculate_generic(score["認知"], 認知_rise)
            y = 認知_xy["y1"] if key=="interval_notice" else int(認知_xy["y2"] - (150*(1+p) if key=="interval_good" else 150*p))
            coords.append((認知_xy["x"], y))
            coords.append((認知_xy["x"], 認知_xy["y1"]))  # 起點

        # 視力問卷
        if score["視力問卷"] == -1:
            coords.append((認知_xy["x"], 認知_xy["y1"]))  # 起點
        else:
            key, p = calculate_generic(score["視力問卷"], 視力問卷_rise)
            x = 視力_xy["x2"] + 300 if key=="interval_good" else int(視力_xy["x2"] + 150*p)
            coords.append((x, 視力_xy["y"]))

        # 視力
        if score["視力"] == -1:
            coords.append((認知_xy["x"], 認知_xy["y1"]))  # 起點
        else:
            key, p = calculate_generic(score["視力"], 視力_rise)       
            x = (1137+106*p) if key=="interval_good" else (1032+106*p)
            coords.append((int(x), int(0.99*x - 450)))

        # 情緒
        if score["情緒"] == -1:
            coords.append((認知_xy["x"], 認知_xy["y1"]))  # 起點
        else:
            key, p = calculate_generic(score["情緒"], 情緒_rise)
            y = int(情緒_xy["y1"] + 150*p) if key=="interval_notice" else (情緒_xy["y2"]+300 if key=="interval_good" else int(情緒_xy["y2"] + 150*p))
            coords.append((情緒_xy["x"], y))

        # 營養
        if score["營養"] == -1:
            coords.append((認知_xy["x"], 認知_xy["y1"]))  # 起點
        else:
            key, p = calculate_generic(score["營養"], 營養_rise)
            x = (836-106*p) if key=="interval_good" else (943-106*p)
            coords.append((int(x), int(-0.99*x + 1503)))

        # 支持
        if score["支持"] == -1:
            coords.append((認知_xy["x"], 認知_xy["y1"]))  # 起點
        else:
            key, p = calculate_generic(score["支持"], 支持_rise)
            x = 支持_xy["x2"] - 300 if key=="interval_good" else int(支持_xy["x2"] - 150*p)
            coords.append((x, 支持_xy["y"]))

        # 體能評估
        if score["行動"] == -1:
            coords.append((認知_xy["x"], 認知_xy["y1"]))  # 起點
        else:
            key, p = calculate_generic(score["行動"], 行動_rise)
            x = (838-106*p) if key=="interval_good" else (945-106*p)
            coords.append((int(x), int(0.99*x - 450)))

        # Overlay
        overlay = Image.new("RGBA", img.size, (0,0,0,0))
        draw_overlay = ImageDraw.Draw(overlay)

        radius = 10          # 點半徑
        outline_width = 2    # 外框粗細
        line_color = (0,0,0,255)
        skin_color = (255,224,189,120)

        # 繪製多邊形
        draw_overlay.polygon(coords, fill=skin_color)

        # 合併 overlay
        img = Image.alpha_composite(img, overlay)

        # 繪製線
        draw = ImageDraw.Draw(img)
        draw.line(coords + [coords[0]], fill=line_color, width=outline_width)

        # 繪製每個點
        for x, y in coords:
            draw.ellipse(
                (x-radius, y-radius, x+radius, y+radius),
                fill=(255,255,255,255),
                outline=(0,0,0,255),
                width=outline_width
            )

        # 最後縮放存檔
        new_width, new_height = 1024, 600
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        os.makedirs("./result", exist_ok=True)
        s = data.get('id_num', str(uuid.uuid4().hex))
        path = f"./result/{s}.png"
        img_resized.save(path)

        return send_file(path, mimetype="image/png")




def draw_selected_landmarks(rgb_image, detection_result):
    """繪製選定的關節點"""
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    selected_pixels = []

    for idx, pose_landmarks in enumerate(pose_landmarks_list):
        h, w, _ = annotated_image.shape
        pixel_coords = []

        for i, landmark in enumerate(pose_landmarks):
            name = POSE_LANDMARK_NAMES[i]
            if name in TARGET_LANDMARKS:
                x_px = int(landmark.x * w)
                y_px = int(landmark.y * h)
                pixel_coords.append((name, x_px, y_px))
                cv2.circle(annotated_image, (x_px, y_px), 5, (0, 255, 255), -1)

        selected_pixels.append(pixel_coords)

        connections = [
            ("LEFT_INDEX", "RIGHT_INDEX")
        ]
        for start_name, end_name in connections:
            start = next(((x, y) for n, x, y in pixel_coords if n == start_name), None)
            end = next(((x, y) for n, x, y in pixel_coords if n == end_name), None)
            if start and end:
                cv2.line(annotated_image, start, end, (255, 255, 0), 2)

    return annotated_image, selected_pixels

def mediapipe_detect(file_path):
    """MediaPipe 姿勢偵測（保持圖片直立方向）"""
    
    model_path = 'pose_landmarker.task'
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型檔案: {model_path}")
        print("請從以下網址下載: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker")
        return None, None
    
    if not os.path.exists(file_path):
        print(f"❌ 找不到圖片檔案: {file_path}")
        return None, None
    
    img_cv = cv2.imread(file_path)
    if img_cv is None:
        print(f"❌ 無法讀取圖片: {file_path}")
        return None, None
    
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    print(f"📐 圖片尺寸: 高 {img_rgb.shape[0]} x 寬 {img_rgb.shape[1]}")
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    
    detection_result = detector.detect(mp_image)
    img_array = img_rgb

    height_pixels = 0
    if detection_result.segmentation_masks:
        mask = detection_result.segmentation_masks[0].numpy_view()

        if mask.ndim == 3 and mask.shape[2] == 1:
            mask_2d = mask[:, :, 0]
        elif mask.ndim == 2:
            mask_2d = mask
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")

        y_indices, x_indices = np.where(mask_2d > 0.5)
        if len(y_indices) > 0:
            top_y = np.min(y_indices)
            bottom_y = np.max(y_indices)
            height_pixels = bottom_y - top_y
            print(f"頭到腳的 pixel 高度: {height_pixels} 像素")
        else:
            top_y = bottom_y = 0
            print("未偵測到人體區域，請確認影像中有人物。")

        color_mask = np.zeros_like(img_array, dtype=np.uint8)
        color_mask[mask_2d > 0.5] = (0, 0, 255)
        alpha = 0.5
        overlay = cv2.addWeighted(img_array, 1.0, color_mask, alpha, 0)

        h, w, _ = overlay.shape
        center_x = w // 2
        cv2.circle(overlay, (center_x, top_y), 8, (0, 255, 255), -1)
        cv2.circle(overlay, (center_x, bottom_y), 8, (0, 255, 255), -1)

        text1 = f"Height: {height_pixels}px"
        cv2.putText(overlay, text1, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        out_path = os.path.splitext(file_path)[0] + "_overlay.png"
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"✅ 覆蓋圖已儲存: {out_path}")

    annotated_image, all_landmark_pixels = draw_selected_landmarks(img_array, detection_result)
    
    landmarks_path = os.path.splitext(file_path)[0] + "_landmarks.png"
    cv2.imwrite(landmarks_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    print(f"✅ 關節點標記圖已儲存: {landmarks_path}")

    left_index = None
    right_index = None
    
    if all_landmark_pixels and len(all_landmark_pixels) > 0:
        for name, x, y in all_landmark_pixels[0]:
            if name == "LEFT_INDEX":
                left_index = (x, y)
            elif name == "RIGHT_INDEX":
                right_index = (x, y)

    hand_distance = 0
    if left_index and right_index:
        lx, ly = left_index
        rx, ry = right_index
        hand_distance = round(np.sqrt((lx - rx) ** 2 + (ly - ry) ** 2), 2)
        print(f"左右食指的距離: {hand_distance} 像素")
    else:
        print("⚠️ 未偵測到左右食指")

    return hand_distance, height_pixels

def detect_and_crop_both_feet(file_path, padding_ratio=None, save_output=True):
    """
    偵測雙腳位置並裁切（padding 根據腳距佔圖片寬度的比例自動計算）
    
    Args:
        file_path: 圖片路徑
        padding_ratio: 如果為 None，則自動使用 feet_width/image_width 作為 padding_ratio
                      也可手動指定比例（例如 0.1 表示 padding = 圖片寬度的 10%）
        save_output: 是否儲存結果
    """
    
    model_path = 'pose_landmarker.task'
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型檔案: {model_path}")
        return None
    
    img_cv = cv2.imread(file_path)
    if img_cv is None:
        print(f"❌ 無法讀取圖片: {file_path}")
        return None
    
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    
    detection_result = detector.detect(mp_image)
    
    if not detection_result.pose_landmarks:
        print("❌ 未偵測到人體姿勢！")
        return None
    
    pose_landmarks = detection_result.pose_landmarks[0]
    
    foot_landmarks = [
        pose_landmarks[27],  # LEFT_ANKLE
        pose_landmarks[28],  # RIGHT_ANKLE
        pose_landmarks[29],  # LEFT_HEEL
        pose_landmarks[30],  # RIGHT_HEEL
        pose_landmarks[31],  # LEFT_FOOT_INDEX
        pose_landmarks[32],  # RIGHT_FOOT_INDEX
    ]
    
    foot_points = [(int(lm.x * w), int(lm.y * h)) for lm in foot_landmarks]
    
    x_coords = [pt[0] for pt in foot_points]
    y_coords = [pt[1] for pt in foot_points]
    
    feet_min_x = min(x_coords)
    feet_max_x = max(x_coords)
    feet_width = feet_max_x - feet_min_x
    
    feet_width_ratio = feet_width / w
    
    if padding_ratio is None:
        padding_ratio = feet_width_ratio + 0.03  # 預設在腳距比例基礎上增加 5% 的 padding
    
    padding = int(w * padding_ratio)
    
    print(f"📏 左右腳距離: {feet_width} 像素")
    print(f"📐 圖片寬度: {w} 像素")
    print(f"📊 腳距佔圖片寬度比: {feet_width_ratio*100:.2f}%")
    print(f"🔧 padding_ratio: {padding_ratio*100:.2f}%")
    print(f"✂️ 計算出的 padding: {padding} 像素 (圖片寬度的 {padding_ratio*100:.2f}%)")
    
    min_x = max(0, min(x_coords) - padding)
    max_x = min(w, max(x_coords) + padding)
    min_y = max(0, min(y_coords) - padding)
    max_y = min(h, max(y_coords) + padding)
    
    left_ankle_px = foot_points[0]
    right_ankle_px = foot_points[1]
    left_heel_px = foot_points[2]
    right_heel_px = foot_points[3]
    left_foot_px = foot_points[4]
    right_foot_px = foot_points[5]
    
    print(f"📦 雙腳裁切區域: ({min_x}, {min_y}) 到 ({max_x}, {max_y})")
    print(f"📐 裁切尺寸: {max_x - min_x} x {max_y - min_y} 像素")
    
    img_bgr = img_cv.copy()
    
    cv2.circle(img_bgr, left_foot_px, 5, (255, 0, 255), -1)
    cv2.circle(img_bgr, right_foot_px, 5, (255, 0, 255), -1)
    cv2.circle(img_bgr, left_heel_px, 5, (0, 255, 255), -1)
    cv2.circle(img_bgr, right_heel_px, 5, (0, 255, 255), -1)
    cv2.circle(img_bgr, left_ankle_px, 5, (0, 255, 0), -1)
    cv2.circle(img_bgr, right_ankle_px, 5, (0, 255, 0), -1)
    
    feet_crop = img_bgr[min_y:max_y, min_x:max_x]
    
    if save_output:
        crop_path = "both_feet_crop.png"
        cv2.imwrite(crop_path, feet_crop)
        print(f"✂️ 雙腳裁切圖已儲存: {crop_path}")
    
    result = {
        'left_foot_toe': left_foot_px,
        'left_foot_heel': left_heel_px,
        'left_ankle': left_ankle_px,
        'right_foot_toe': right_foot_px,
        'right_foot_heel': right_heel_px,
        'right_ankle': right_ankle_px,
        'crop_region': (min_x, min_y, max_x, max_y),
        'crop_size': (max_x - min_x, max_y - min_y),
        'feet_crop': feet_crop,
        'feet_width': feet_width,
        'feet_width_ratio': feet_width_ratio,
        'padding_used': padding
    }
    
    return result

def texture(crop_image_path='both_feet_crop.png'):
    """測量紙張距離並儲存各階段處理結果"""
    image = cv2.imread(crop_image_path)
    if image is None: 
        print("❌ 找不到裁切圖片")
        return 0
    
    # 1. 轉灰階
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. 二值化
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    cv2.imwrite('debug_step1_binary_fail.png', binary)
    # 3. 尋找並篩選最大輪廓 (只保留最大的白色區塊)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("❌ 找不到白色參考物")
        # 即使失敗也存一張二值化圖，方便 debug 為什麼找不到
        cv2.imwrite('debug_step1_binary_fail.png', binary)
        return 0
    
    max_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [max_contour], -1, 255, thickness=cv2.FILLED)
    # --- 儲存第一步：最大區域遮罩 ---
    cv2.imwrite('debug_step1_mask.png', mask)
    
    # 4. 2次侵蝕 + 1次膨脹
    kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=2) 
    refined_binary = cv2.dilate(eroded, kernel, iterations=1) 
    # --- 儲存第二步：去雜訊後結果 ---
    cv2.imwrite('debug_step2_refined.png', refined_binary)
    
    # 5. 計算測量線
    white_pixels = np.where(refined_binary == 255)
    dist = 0
    res_img = image.copy()
    
    if len(white_pixels[0]) > 0:
        y, x = white_pixels[0], white_pixels[1]
        left_pt = (x[np.argmin(x)], y[np.argmin(x)])
        right_pt = (x[np.argmax(x)], y[np.argmax(x)])
        dist = np.sqrt((right_pt[0]-left_pt[0])**2 + (right_pt[1]-left_pt[1])**2)
        
        # 在原圖畫線標註
        cv2.line(res_img, left_pt, right_pt, (0, 255, 0), 3)
        cv2.circle(res_img, left_pt, 8, (255, 0, 0), -1)
        cv2.circle(res_img, right_pt, 8, (0, 0, 255), -1)
        
        # 在圖片右上角寫上像素距離
        cv2.putText(res_img, f"{dist:.2f}px", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # --- 儲存第三步：最終結果標註 ---
    cv2.imwrite('debug_step3_result.png', res_img)
    
    print(f"✅ 影像處理完成，圖片已儲存至 debug_step1~3.png")
    return dist

@app.route('/height', methods=['GET','POST'])
def cal_height():
    data = request.get_json(force=True)
    try:
        b64_str = data.get("pic1")
        SAVE_FILE = "pic.json"  
        if os.path.exists(SAVE_FILE):
            with open(SAVE_FILE, "r", encoding="utf-8") as f:
                try:
                    all_data = json.load(f)
                    if not isinstance(all_data, list):
                        all_data = [all_data]  # 如果原本不是 list，就轉成 list
                except json.JSONDecodeError:
                    all_data = []
        else:
            all_data = []
        all_data.append(data)
        with open(SAVE_FILE, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)

        if not b64_str:
            return jsonify({"error": "pic1 not found"}), 400

        #  去掉 data:image/...;base64,
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]

        img_bytes = base64.b64decode(b64_str)

    except Exception as e:
        return jsonify({"error": "Invalid JSON or base64", "detail": str(e)}), 400

    os.makedirs("uploads", exist_ok=True)
    image_path = f"uploads/{uuid.uuid4().hex}.jpg"
    with open(image_path, "wb") as f:
        f.write(img_bytes)
    print("saved image size:", os.path.getsize(image_path))

    print("=" * 50)
    print("步驟 1: MediaPipe 姿勢偵測")
    print("=" * 50)
    hand_distance, height_pixels = mediapipe_detect(image_path)
    
    if hand_distance is not None:
        print("\n" + "=" * 50)
        print("步驟 2: 雙腳裁切")
        print("=" * 50)
        
        # 🔧 方式 1：自動使用腳距佔圖片寬度的比例（預設）
        result = detect_and_crop_both_feet(image_path, padding_ratio=None, save_output=True)
        
        # 🔧 方式 2：手動指定 padding = 圖片寬度的 10%
        # result = detect_and_crop_both_feet(file_path, padding_ratio=0.1, save_output=True)
        
        if result:
            print("\n" + "=" * 50)
            print("步驟 3: 腳部距離測量")
            print("=" * 50)
            feet_distance = texture()
            
            print("\n" + "=" * 50)
            print("最終結果")
            print("=" * 50)
            print(f"手指距離: {hand_distance} 像素")
            print(f"身體高度: {height_pixels} 像素")
            pixel = 42 / feet_distance #(cm/pixel) 
            height = pixel * height_pixels
            hand = pixel * hand_distance
            print(f"紙張距離: {feet_distance:.2f} 像素")
            print(f"左右腳關節點距離: {result['feet_width']} 像素")
            print(f"腳距佔圖片寬度比: {result['feet_width_ratio']*100:.2f}%")
            print(f"使用的 padding: {result['padding_used']} 像素")
            print(f"人高度: {height:.2f} cm")
            print(f"手指長度: {hand:.2f} cm")
            return jsonify({
                "message": "success",
                "height": f"{height:.1f}",
                "hand_length": f"{hand:.1f}",
            })


def detect_and_crop_both_feet_v2(file_path, padding_ratio=None, save_output=True):
    """
    偵測雙腳位置並裁切（padding 根據腳距佔圖片寬度的比例自動計算）
    
    Args:
        file_path: 圖片路徑
        padding_ratio: 如果為 None，則自動使用 feet_width/image_width 作為 padding_ratio
                      也可手動指定比例（例如 0.1 表示 padding = 圖片寬度的 10%）
        save_output: 是否儲存結果
    """
    
    model_path = 'pose_landmarker.task'
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型檔案: {model_path}")
        return None
    
    img_cv = cv2.imread(file_path)
    if img_cv is None:
        print(f"❌ 無法讀取圖片: {file_path}")
        return None
    
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    
    detection_result = detector.detect(mp_image)
    
    if not detection_result.pose_landmarks:
        print("❌ 未偵測到人體姿勢！")
        return None
    
    pose_landmarks = detection_result.pose_landmarks[0]
    
    foot_landmarks = [
        pose_landmarks[27],  # LEFT_ANKLE
        pose_landmarks[28],  # RIGHT_ANKLE
        pose_landmarks[29],  # LEFT_HEEL
        pose_landmarks[30],  # RIGHT_HEEL
        pose_landmarks[31],  # LEFT_FOOT_INDEX
        pose_landmarks[32],  # RIGHT_FOOT_INDEX
    ]
    
    foot_points = [(int(lm.x * w), int(lm.y * h)) for lm in foot_landmarks]
    
    x_coords = [pt[0] for pt in foot_points]
    y_coords = [pt[1] for pt in foot_points]
    
    feet_min_x = min(x_coords)
    feet_max_x = max(x_coords)
    feet_width = feet_max_x - feet_min_x
    foot_points[4][1]
    
    feet_width_ratio = feet_width / w
    
    if padding_ratio is None:
        padding_ratio = feet_width_ratio + 0.03  # 預設在腳距比例基礎上增加 5% 的 padding
    
    padding = int(w * padding_ratio)
    
    print(f"📏 左右腳距離: {feet_width} 像素")
    print(f"📐 圖片寬度: {w} 像素")
    print(f"📊 腳距佔圖片寬度比: {feet_width_ratio*100:.2f}%")
    print(f"🔧 padding_ratio: {padding_ratio*100:.2f}%")
    print(f"✂️ 計算出的 padding: {padding} 像素 (圖片寬度的 {padding_ratio*100:.2f}%)")
    
    min_x = max(0, min(x_coords) - padding)
    max_x = min(w, max(x_coords) + padding)
    min_y = max(0, min(y_coords) - padding)
    max_y = min(h, max(y_coords) + padding)
    
    left_ankle_px = foot_points[0]
    right_ankle_px = foot_points[1]
    left_heel_px = foot_points[2]
    right_heel_px = foot_points[3]
    left_foot_px = foot_points[4]
    right_foot_px = foot_points[5]
    
    print(f"📦 雙腳裁切區域: ({min_x}, {min_y}) 到 ({max_x}, {max_y})")
    print(f"📐 裁切尺寸: {max_x - min_x} x {max_y - min_y} 像素")
    
    img_bgr = img_cv.copy()
    
    cv2.circle(img_bgr, left_foot_px, 5, (255, 0, 255), -1)
    cv2.circle(img_bgr, right_foot_px, 5, (255, 0, 255), -1)
    cv2.circle(img_bgr, left_heel_px, 5, (0, 255, 255), -1)
    cv2.circle(img_bgr, right_heel_px, 5, (0, 255, 255), -1)
    cv2.circle(img_bgr, left_ankle_px, 5, (0, 255, 0), -1)
    cv2.circle(img_bgr, right_ankle_px, 5, (0, 255, 0), -1)
    right_foot_crop_x = right_foot_px[0] - min_x
    right_foot_crop_y = right_foot_px[1] - min_y
    feet_crop = img_bgr[min_y:max_y, min_x:max_x]
    
    if save_output:
        crop_path = "both_feet_crop.png"
        cv2.imwrite(crop_path, feet_crop)
        print(f"✂️ 雙腳裁切圖已儲存: {crop_path}")
    
    result = {
        'right_foot_toe': right_foot_crop_y,
        'feet_width': feet_width,
        'feet_width_ratio': feet_width_ratio,
        'padding_used': padding
    }
    
    return result

def texture_v2(right_foot, crop_image_path='both_feet_crop.png'):
    """測量紙張距離並儲存各階段處理結果"""
    print("❗是沒有膨脹和侵蝕的版本")
    image = cv2.imread(crop_image_path)
    if image is None: 
        print("❌ 找不到裁切圖片")
        return 0

    right_foot_y = right_foot
    # 1. 轉灰階
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. 二值化
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    cv2.imwrite('debug_step1_binary_fail.png', binary)
    # 3. 尋找並篩選最大輪廓 (只保留最大的白色區塊)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("❌ 找不到白色參考物")
        # 即使失敗也存一張二值化圖，方便 debug 為什麼找不到
        cv2.imwrite('debug_step1_binary_fail.png', binary)
        return 0
    
    # 🔧 3. 在 right_foot_y 這一行的 X方向找白色連續長度
    row = binary[right_foot_y, :]  # 取出這一整行的像素（1D array）
    white_indices = np.where(row == 255)[0]  # 找所有白色像素的X座標
    
    if len(white_indices) == 0:
        print(f"❌ Y={right_foot_y} 這行沒有白色像素")
        return 0
    
    # 找連續白色區段（假設只有一個主要紙張）
    left_x = np.min(white_indices)
    right_x = np.max(white_indices)
    paper_width = right_x - left_x + 1  # 包含左右端點
    
    print(f"📏 Y={right_foot_y} 行白色連續長度: {paper_width} 像素")
    print(f"   範圍: X={left_x} 到 X={right_x}")
    
    # 可視化：在原圖標註
    debug_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.line(debug_img, (left_x, right_foot_y), (right_x, right_foot_y), (0, 255, 0), 3)
    cv2.circle(debug_img, (left_x, right_foot_y), 8, (0, 0, 255), -1)
    cv2.circle(debug_img, (right_x, right_foot_y), 8, (255, 0, 0), -1)
    cv2.putText(debug_img, f"Width: {paper_width}px", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imwrite('debug_paper_width.png', cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
    print("✅ 測量結果已儲存: debug_paper_width.png")
    dist = paper_width

    # max_contour = max(contours, key=cv2.contourArea)
    # mask = np.zeros_like(binary)
    # cv2.drawContours(mask, [max_contour], -1, 255, thickness=cv2.FILLED)
    # # --- 儲存第一步：最大區域遮罩 ---
    # cv2.imwrite('debug_step1_mask.png', mask)
    
    # # 4. 2次侵蝕 + 1次膨脹
    # kernel = np.ones((5, 5), np.uint8)
    # eroded = cv2.erode(mask, kernel, iterations=2)  # 2次侵蝕
    # refined_binary = cv2.dilate(eroded, kernel, iterations=1) # 1次膨脹
    # # --- 儲存第二步：去雜訊後結果 ---
    # cv2.imwrite('debug_step2_refined.png', refined_binary) 
    
    # # 5. 計算測量線
    # white_pixels = np.where(refined_binary == 255)
    # dist = 0
    # res_img = image.copy()
    
    # if len(white_pixels[0]) > 0:
    #     y, x = white_pixels[0], white_pixels[1]
    #     left_pt = (x[np.argmin(x)], y[np.argmin(x)])
    #     right_pt = (x[np.argmax(x)], y[np.argmax(x)])
    #     dist = np.sqrt((right_pt[0]-left_pt[0])**2 + (right_pt[1]-left_pt[1])**2)
        
    #     # 在原圖畫線標註
    #     cv2.line(res_img, left_pt, right_pt, (0, 255, 0), 3)
    #     cv2.circle(res_img, left_pt, 8, (255, 0, 0), -1)
    #     cv2.circle(res_img, right_pt, 8, (0, 0, 255), -1)
        
    #     # 在圖片右上角寫上像素距離
    #     cv2.putText(res_img, f"{dist:.2f}px", (10, 30), 
    #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # # --- 儲存第三步：最終結果標註 ---
    # cv2.imwrite('debug_step3_result.png', res_img)
    
    # print(f"✅ 影像處理完成，圖片已儲存至 debug_step1~3.png")
    return dist  #pixel

@app.route('/height_v2', methods=['GET','POST'])  #取最大面積，跟腳踝平行的紙張寬度(去除膨脹、侵蝕)，改動detect_and_crop_both_feet_v2、texture_v2
def cal_height_v2():
    data = request.get_json(force=True)
    try:
        b64_str = data.get("pic1")
        SAVE_FILE = "pic.json"  
        if os.path.exists(SAVE_FILE):
            with open(SAVE_FILE, "r", encoding="utf-8") as f:
                try:
                    all_data = json.load(f)
                    if not isinstance(all_data, list):
                        all_data = [all_data]  # 如果原本不是 list，就轉成 list
                except json.JSONDecodeError:
                    all_data = []
        else:
            all_data = []
        all_data.append(data)
        with open(SAVE_FILE, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)

        if not b64_str:
            return jsonify({"error": "pic1 not found"}), 400

        #  去掉 data:image/...;base64,
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]

        img_bytes = base64.b64decode(b64_str)

    except Exception as e:
        return jsonify({"error": "Invalid JSON or base64", "detail": str(e)}), 400

    os.makedirs("uploads", exist_ok=True)
    image_path = f"uploads/{uuid.uuid4().hex}.jpg"
    with open(image_path, "wb") as f:
        f.write(img_bytes)
    print("saved image size:", os.path.getsize(image_path))

    print("=" * 50)
    print("步驟 1: MediaPipe 姿勢偵測")
    print("=" * 50)
    hand_distance, height_pixels = mediapipe_detect(image_path)
    
    if hand_distance is not None:
        print("\n" + "=" * 50)
        print("步驟 2: 雙腳裁切")
        print("=" * 50)
        
        # 🔧 方式 1：自動使用腳距佔圖片寬度的比例（預設）
        result = detect_and_crop_both_feet_v2(image_path, padding_ratio=None, save_output=True)
        
        # 🔧 方式 2：手動指定 padding = 圖片寬度的 10%
        # result = detect_and_crop_both_feet(file_path, padding_ratio=0.1, save_output=True)
        
        if result:
            print("\n" + "=" * 50)
            print("步驟 3: 腳部距離測量")
            print("=" * 50)
            paper_distance = texture_v2(result['right_foot_toe']) #pixel
            pixel = 42 / paper_distance #(cm/pixel) 
            height_2 = pixel * height_pixels
            hand = pixel * hand_distance
            print("\n" + "=" * 50)
            print("最終結果")
            print("=" * 50)
            print(f"手指距離: {hand_distance} 像素")
            print(f"身體高度: {height_pixels} 像素")
            print(f"紙張距離: {paper_distance:.2f} 像素")
          
            print(f"手指距離: {hand:.2f} cm")
            print(f"人高度: {height_2:.2f} cm")  
            print(f"紙張距離: {42} cm")
            return jsonify({
                "message": "success",
                "height": f"{height_2:.1f}",
                "hand_length": f"{hand:.1f}",
            })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

