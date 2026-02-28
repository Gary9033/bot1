from PIL import Image, ImageDraw
from flask import Flask, request, send_file
import os

from polars import datetime

from main import height
app = Flask(__name__)

score = {"認知":7,"視力":2,"情緒":7,"營養":8,"支持":5}
認知 = {"interval_improve": 2, "interval_notice": 1, "interval_good": 0, "interval_max": 8}
視力 = {"interval_improve": 1, "interval_good": 0, "interval_max": 2}
情緒 = {"interval_improve": 10, "interval_notice": 6, "interval_good": 0, "interval_max": 15}
營養 = {"interval_improve": 0, "interval_good": 12, "interval_max": 14}
支持 = {"interval_improve": 1, "interval_good": 0, "interval_max": 6}
認知_rise = dict(sorted(認知.items(), key=lambda item: item[1]))
視力_rise = dict(sorted(視力.items(), key=lambda item: item[1]))
情緒_rise = dict(sorted(情緒.items(), key=lambda item: item[1]))
營養_rise = dict(sorted(營養.items(), key=lambda item: item[1]))
支持_rise = dict(sorted(支持.items(), key=lambda item: item[1]))

認知_xy = {"x": 988, "y1": 529, "y2": 467}
視力_xy = {"y": 529, "x1": 988, "x2": 1048}
情緒_xy = {"x": 988, "y1": 529, "y2": 589}
支持_xy = {"y": 529, "x1": 988, "x2": 928}

def calculate_1():
    if (score["認知"]>=list(認知_rise.values())[0]) & (score["認知"]<list(認知_rise.values())[1]):
        key = list(認知_rise.keys())[0]
    elif (score["認知"]>=list(認知_rise.values())[1]) & (score["認知"]<list(認知_rise.values())[2]):
        key = list(認知_rise.keys())[1]
    else:
        key = list(認知_rise.keys())[2]
    print(key)
    a=list(認知_rise.keys()).index(key)
    total = list(認知_rise.values())[a+1] - list(認知_rise.values())[a] 
    sub =  score["認知"] - list(認知_rise.values())[a] 
    percent = round(sub/total,2)
    print(percent) #在當前分級的百分比
    認知_x = 認知_xy["x"]
    if key == "interval_notice":
        認知_y = 認知_xy["y1"]
    elif key == "interval_good":
        認知_y = int(認知_xy["y2"] - 150*(1+percent))
    else:
        認知_y = int(認知_xy["y2"] - percent*150)
    current_coord = (認知_x, 認知_y)
    return current_coord

def calculate_2():
    if (score["視力"]>=list(視力_rise.values())[0]) & (score["視力"]<list(視力_rise.values())[1]):
        key = list(視力_rise.keys())[0]
    else:
        key = list(視力_rise.keys())[1]
    print(key)
    a=list(視力_rise.keys()).index(key)
    total = list(視力_rise.values())[a+1] - list(視力_rise.values())[a] 
    sub =  score["視力"] - list(視力_rise.values())[a] 
    percent = round(sub/total,2)
    print(percent) #在當前分級的百分比
    視力_y = 視力_xy["y"]
    if key == "interval_good":
        視力_x = 視力_xy["x2"] + 300
    else:
        視力_x = int(視力_xy["x2"] + (percent)*150)
    current_coord = (視力_x, 視力_y)
    return current_coord

def calculate_3():
    if (score["情緒"]>=list(情緒_rise.values())[0]) & (score["情緒"]<list(情緒_rise.values())[1]):
        key = list(情緒_rise.keys())[0]
    elif (score["情緒"]>=list(情緒_rise.values())[1]) & (score["情緒"]<list(情緒_rise.values())[2]):
        key = list(情緒_rise.keys())[1]
    else:
        key = list(情緒_rise.keys())[2]
    print(key)
    a=list(情緒_rise.keys()).index(key)
    total = list(情緒_rise.values())[a+1] - list(情緒_rise.values())[a] 
    sub =  score["情緒"] - list(情緒_rise.values())[a] 
    percent = round(sub/total,2)
    print(percent) #在當前分級的百分比
    情緒_x = 情緒_xy["x"]
    if key == "interval_notice":
        情緒_y = int(情緒_xy["y1"] + 150*percent)
    elif key == "interval_good":
        情緒_y = 情緒_xy["y2"] + 300
    else:
        情緒_y = int(情緒_xy["y2"] + percent*150)
    current_coord = (情緒_x, 情緒_y)
    return current_coord

def calculate_4():
    if (score["營養"]>=list(營養_rise.values())[0]) & (score["營養"]<list(營養_rise.values())[1]):
        key = list(營養_rise.keys())[0]
    else:
        key = list(營養_rise.keys())[1]
    print(key) 
    a=list(營養_rise.keys()).index(key)
    total = list(營養_rise.values())[a+1] - list(營養_rise.values())[a] 
    sub =  score["營養"] - list(營養_rise.values())[a] 
    percent = round(sub/total,2)
    print(percent) #在當前分級的百分比
    if key == "interval_good":
        營養_x = 836-(106*percent)
    else:
        營養_x = 942-(106*percent) 
    營養_y = -0.99*營養_x+1503
    current_coord = (int(營養_x), int(營養_y))
    return current_coord

def calculate_5():
    if (score["支持"]>=list(支持_rise.values())[0]) & (score["支持"]<list(支持_rise.values())[1]):
        key = list(支持_rise.keys())[0]
    else:
        key = list(支持_rise.keys())[1]
    print(key)
    a=list(支持_rise.keys()).index(key)
    total = list(支持_rise.values())[a+1] - list(支持_rise.values())[a] 
    sub =  score["支持"] - list(支持_rise.values())[a] 
    percent = round(sub/total,2)
    print(percent) #在當前分級的百分比
    支持_y = 支持_xy["y"]
    if key == "interval_good":
        支持_x = 支持_xy["x2"] - 300
    else:
        支持_x = int(支持_xy["x2"] - percent*(150))
    current_coord = (支持_x, 支持_y)
    return current_coord


@app.route('/image')
def show_image():
    functions = [calculate_1, calculate_2, calculate_3, calculate_4, calculate_5]
    all_coords = []
    for func in functions:
        result = func() 
        all_coords.append(result)
    
    print(all_coords)
    
    img_path = 'statistics.png' 
    img = Image.open(img_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    
    # 這裡建議 a 可以用動態的，或者固定一個檔名
    a = 1
    save_directory = "./result"
    save_path = f"{save_directory}/output_result_{a}.png"
    
    # 檢查資料夾是否存在，不存在就建立（避免報錯）
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # 設定點的半徑與顏色
    radius = 5
    color = (255, 0, 0, 255) 
    
    for x, y in all_coords:
        ix, iy = int(round(x)), int(round(y))
        draw.ellipse((ix - radius, iy - radius, ix + radius, iy + radius), fill=color)
    
    # 1. 先執行存檔
    img.save(save_path)
    print(f"繪製完成！已儲存為 {save_path}")
    
    # 2. 關鍵修正：使用 send_file 傳送剛存好的檔案路徑
    return send_file(save_path, mimetype='image/png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

