焦距 (f): 24 mm
像素大小 (Pixel Size): 1.12 µm

aku@gap.cgu.edu.tw
Dm36723672

https://roflow-user-preview.nuwarobotics.com/projectList

https://support.nuwarobotics.com/zh-hant/docs/biztools/reception/tutorial/

啟動gemini cli
在powershell裡
npx https://github.com/google-gemini/gemini-cli


conda:
查看有哪些env: conda env list
輸入gemini打開geminiCLI
儲存對話：/chat save <tag>
恢復對話：/chat resume <tag>
<tag> 可以自行定義任何關鍵字，舉例 /chat save abc123


/chat save 1210

把image pull下來
docker pull b1029033/nuwabot:latest
把圖存到本機(還沒正式try過)
docker run -it -p 8080:5000 -v D:/some/path/uploads:/app/uploads b1029033/nuwabot:latest

docker compose:
