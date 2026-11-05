"""AI Beauty Advisor 後端（FastAPI）

啟動（本機）：
    uvicorn main:app --reload --port 8000
然後用瀏覽器開 http://127.0.0.1:8000/

金鑰一律放在環境變數或 .env 檔（不要寫進程式、不要上傳到 GitHub）：
    GEMINI_API_KEY、OPENAI_API_KEY
"""
import asyncio
import base64
import copy
import io
import json
import os
import time
from collections import defaultdict, deque

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

import google.generativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from openai import OpenAI

load_dotenv()

# ==========================================
# 設定（全部從環境變數讀）
# ==========================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")  # 兩個名字都接受
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2")
RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "30"))
MAX_UPLOAD_MB = 8
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS",
        "https://411277024.github.io,http://127.0.0.1:8000,http://localhost:8000",
    ).split(",")
    if o.strip()
]

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("⚠️ 沒有設定 GEMINI_API_KEY：照片分析停用，文字報告改用備用內容", flush=True)

openai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=60) if OPENAI_API_KEY else None
if openai_client is None:
    print("⚠️ 沒有設定 OPENAI_API_KEY：眼妝圖片生成停用（前端會用預設圖）", flush=True)

app = FastAPI(title="AI Beauty Advisor API")

# CORS：只開給你的網站，不要用 "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ==========================================
# 每個 IP 每小時的請求上限（保護你的 API 額度）
# ==========================================
_hits = defaultdict(deque)


def rate_limit(request: Request):
    ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0].strip()
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= RATE_LIMIT_PER_HOUR:
        raise HTTPException(status_code=429, detail="請求太頻繁，請稍後再試")
    q.append(now)


@app.get("/api/health")
def health():
    return {"status": "ok", "gemini": bool(GEMINI_API_KEY), "openai": openai_client is not None}


# ==========================================
# 1. 接收照片，請 Gemini 做五官分析
# ==========================================
@app.post("/api/analyze-photo", dependencies=[Depends(rate_limit)])
def analyze_photo(file: UploadFile = File(...)):
    print(f"📷 收到照片分析請求：{file.filename}", flush=True)
    if not GEMINI_API_KEY:
        return {"status": "error", "message": "伺服器尚未設定 Gemini 金鑰"}
    try:
        contents = file.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
        if len(contents) > MAX_UPLOAD_MB * 1024 * 1024:
            return {"status": "error", "message": f"照片太大，請小於 {MAX_UPLOAD_MB} MB"}

        # 照片只在記憶體中處理，不會存檔
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.thumbnail((512, 512))
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)

        prompt_text = """
        你現在是一位極度精準的「亞洲人像五官分析專家」。請拿出放大鏡，非常仔細地觀察照片中人物的真實特徵，絕對不要憑空瞎猜。
        請嚴格依照以下格式輸出（只能輸出這 5 行，絕對不要有任何廢話或解釋）：
        臉型：(從 圓臉、方形臉、長臉、鵝蛋臉、心形臉、菱形臉、六角形臉、方圓臉 中挑選)
        眼型：(從 杏仁眼、圓眼、丹鳳眼、桃花眼、狐狸眼、狗狗眼、細長眼 中挑選)
        眼皮：(從 單眼皮、雙眼皮、內雙 中挑選)
        鼻型：(從 小巧鼻、蒜頭鼻、水滴鼻、直挺鼻、寬鼻型、精緻尖鼻、肉感鼻、小翹鼻、鷹鉤鼻、扁鼻 中挑選)
        唇形：(從 薄唇、厚唇、微笑唇、M 字唇、自然唇形 中挑選)
        """
        image_part = {"mime_type": "image/jpeg", "data": buffered.getvalue()}

        model = genai.GenerativeModel(GEMINI_MODEL)
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        response = model.generate_content([prompt_text, image_part], safety_settings=safety_settings)
        result_text = response.text.strip()
        print(f"✅ Gemini 分析完成：\n{result_text}", flush=True)

        analysis_data = {}
        for line in result_text.split("\n"):
            line = line.strip()
            if "：" in line:
                key, value = line.split("：", 1)
                analysis_data[key.strip()] = value.strip()
            elif ":" in line:
                key, value = line.split(":", 1)
                analysis_data[key.strip()] = value.strip()
        return {"status": "success", "data": analysis_data}

    except Exception as e:
        print(f"❌ 照片分析發生錯誤：{e}", flush=True)
        return {"status": "error", "message": f"分析失敗：{e}"}


# ==========================================
# 2. 接收表單資料，產生造型建議報告 + 眼妝圖
# ==========================================
class UserFeatures(BaseModel):
    faceShape: str
    eyeShape: str
    eyelidType: str
    noseShape: str
    lipShape: str
    skinTone: str
    hairLength: str
    style: str
    occasion: str


# 前端下拉選單送來的是英文代碼（round_eye、double…），先換成中文再交給 Gemini，
# 否則 AI 會把代碼原樣寫進報告裡（例如「圓眼（round_eye）」）
LABELS = {
    "faceShape": {"round": "圓臉", "square": "方形臉", "long": "長臉", "oval": "鵝蛋臉", "heart": "心形臉",
                  "diamond": "菱形臉", "hexagonal": "六角形臉", "squareround": "方圓臉"},
    "eyeShape": {"almond": "杏仁眼", "round_eye": "圓眼", "phoenix": "丹鳳眼", "peach": "桃花眼",
                 "fox": "狐狸眼", "dog": "狗狗眼", "narrow": "細長眼"},
    "eyelidType": {"double": "雙眼皮", "single": "單眼皮", "hooded": "內雙"},
    "lipShape": {"thin": "薄唇", "thick": "厚唇", "smile": "微笑唇", "m_shape": "M 字唇", "natural_lip": "自然唇形"},
    "hairLength": {"short": "短髮", "medium": "中長髮", "long": "長髮"},
    "occasion": {"daily": "日常通勤", "date": "浪漫約會", "interview": "面試職場", "travel": "旅遊打卡",
                 "concert": "音樂祭／派對", "night_out": "酒吧夜店", "wedding_guest": "婚禮宴會", "photoshoot": "專業棚拍"},
}
# 畫圖用英文描述效果比較好
EYELID_EN = {"double": "double eyelid", "single": "monolid", "hooded": "hooded inner-double eyelid"}
EYE_EN = {"almond": "almond-shaped eyes", "round_eye": "round eyes", "phoenix": "upturned phoenix eyes",
          "peach": "peach-blossom eyes", "fox": "fox eyes with upturned outer corners",
          "dog": "downturned puppy eyes", "narrow": "narrow elongated eyes"}


# Gemini 失敗或沒設定金鑰時使用的備用報告
FALLBACK_REPORT = {
        "best_hairstyles": ["側分大波浪", "高馬尾辮髮", "復古法式捲"],
        "avoid_hairstyles": ["厚重齊瀏海", "貼頭皮直髮"],
        "hair_colors": [
            {"name": "焦糖拿鐵棕", "desc": "溫暖的棕色調能提亮膚色，帶來柔和親切的氛圍。", "hex": "A67C52"},
            {"name": "冷調黑茶色", "desc": "低調質感的冷黑色，完美襯托高級與清冷氣質。", "hex": "3E2723"},
            {"name": "玫瑰粉棕色", "desc": "帶有微紅光澤的粉棕，增添甜美與純慾撩人的氣息。", "hex": "8B5A2B"}
        ],
        "makeup_steps": [
            {"title": "底妝與修容",
             "desc": "輕薄透亮的半霧面粉底液，針對下顎線與顴骨下方輕掃陰影，打造精緻輪廓。"},
            {"title": "眼妝設計",
             "desc": "大地色系眼影自然暈染，搭配細緻的內眼線與根根分明的睫毛膏，強調溫柔神韻。"},
            {"title": "唇頰搭配", "desc": "珊瑚蜜桃色系的微珠光腮紅，搭配水潤光澤的玫瑰色唇釉，展現純慾氛圍。"}
        ],
        "summary": "這套造型完美結合了你的臉型與五官優勢，大膽展現專屬於你的獨特魅力吧！"
    }


@app.post("/api/recommend", dependencies=[Depends(rate_limit)])
def get_recommendation(features: UserFeatures):
    print("====== 開始處理報告 ======", flush=True)

    # 代碼 → 中文（給 Gemini）；畫圖要用的英文描述先存起來
    eyelid_en = EYELID_EN.get(features.eyelidType, features.eyelidType)
    eye_en = EYE_EN.get(features.eyeShape, features.eyeShape)
    for field, table in LABELS.items():
        value = getattr(features, field)
        setattr(features, field, table.get(value, value))

    # ==========================================
    # 💄 1. Step 1 (底妝與修容) 文案拼圖系統
    # ==========================================
    style_desc = ""
    if features.style in ["自然清新", "日系透明感"]:
        style_desc = "建議選擇輕薄透亮的水光型粉底液，保留肌膚原生質感與自然光澤。局部使用遮瑕修飾黑眼圈與泛紅區域，再以細緻蜜粉輕壓T字部位，打造乾淨清爽且不厚重的妝感，呈現彷彿天生好膚質的透明感效果。"
    elif features.style in ["韓系溫柔", "甜美可愛"]:
        style_desc = "建議使用柔霧光澤粉底打造均勻細膩的肌膚質感，搭配局部提亮眼下、額頭中央及下巴位置。利用淡淡陰影修飾臉部邊緣與鼻樑兩側，讓五官更顯柔和立體，營造自然甜美且充滿親和力的韓系氛圍。"
    elif features.style in ["成熟優雅", "富家千金"]:
        style_desc = "建議選擇持妝度佳的半霧面粉底液，打造平滑細緻且高級感十足的底妝效果。於顴骨下方、下顎線與髮際邊緣輕掃修容，加強臉部立體輪廓；額頭中央、鼻樑與下巴適度提亮，呈現精緻俐落且氣質優雅的妝容風格。"
    elif features.style == "純慾撩人":
        style_desc = "建議使用帶有微光澤感的粉底產品，營造健康且富有光澤的肌膚狀態。透過鼻樑、眼下與唇峰提亮，加強五官聚焦效果；顴骨下方及下顎線則以自然陰影修飾輪廓，使臉部線條更加立體流暢，展現精緻且富有魅力的純慾氛圍。"
    elif features.style == "歐美辣妹":
        style_desc = "建議選擇遮瑕力較佳的霧面粉底液，打造均勻無瑕的底妝基底。利用修容加強顴骨、鼻樑及下顎線的輪廓塑形，搭配高光提亮額頭、鼻尖及顴骨高點，提升五官立體度與臉部層次感，展現自信鮮明且充滿氣場的歐美妝容特色。"
    elif features.style == "清冷個性":
        style_desc = "建議選擇輕霧面質地的底妝產品，打造乾淨俐落且不過度厚重的肌膚質感。透過鼻樑兩側與臉部外輪廓的精準修容，強調骨相線條與輪廓感；局部提亮眼下與鼻樑中央，使整體妝容更具高級感與清冷氣質。"
    elif features.style == "復古港風":
        style_desc = "建議使用帶有自然光澤感的底妝產品，保留肌膚原有紋理與健康膚色。利用暖色調修容輕微加深輪廓，搭配局部提亮增添立體感，使整體妝容兼具復古韻味與現代精緻度，呈現經典港風女神般的優雅魅力。"
    elif features.style == "中性少年感":
        style_desc = "建議以薄透自然的底妝為主，避免過度遮蓋原生肌膚質感。透過輕微修容修飾臉部邊界與鼻樑結構，弱化過度圓潤的線條；局部提亮眉骨與鼻樑中央，提升五官清晰度，打造乾淨俐落且富有少年感的中性氛圍。"
    else:
        style_desc = "建議選擇輕薄透亮的粉底液，保留肌膚原生質感。"

    skin_desc = ""
    if "白皙" in features.skinTone:
        skin_desc = "底妝色號建議選擇自然偏粉調或中性色調，避免過度提亮造成死白感。"
    elif "自然" in features.skinTone:
        skin_desc = "建議選擇最接近原生膚色的色號，保留健康自然的氣色。"
    elif "小麥" in features.skinTone:
        skin_desc = "建議選擇暖調底妝產品，並適度保留原生膚色特色，展現健康且有活力的妝感。"

    nose_desc = ""
    if "小巧鼻" in features.noseShape:
        nose_desc = "鼻型本身比例勻稱且線條自然，建議避免過重修容。可於山根至鼻尖輕刷提亮產品，維持乾淨立體感即可，讓五官看起來更加清秀耐看。"
    elif "蒜頭鼻" in features.noseShape:
        nose_desc = "建議於鼻翼兩側及鼻頭外圍輕掃陰影，弱化鼻頭圓潤感；鼻樑中央以細緻高光提亮，視覺上拉長鼻型比例，使整體輪廓更加俐落精緻。"
    elif "水滴鼻" in features.noseShape:
        nose_desc = "水滴鼻具有柔和且帶有立體度的鼻尖特色，可利用鼻樑中央提亮加強縱向線條，並於鼻翼兩側輕微修飾，進一步凸顯優雅自然的五官比例。"
    elif "直挺鼻" in features.noseShape:
        nose_desc = "鼻型條件較佳，建議以高光輕刷山根、鼻樑及鼻尖位置，維持立體感即可。避免過度修容，讓鼻部線條自然成為整體妝容的視覺焦點。"
    elif "寬鼻型" in features.noseShape:
        nose_desc = "可於鼻樑兩側繪製較窄的修容線條，並加強山根至鼻尖的縱向提亮效果，藉由明暗對比縮小鼻部寬度，讓五官比例更加集中。"
    elif "尖鼻" in features.noseShape:
        nose_desc = "鼻型本身線條俐落且立體，建議以少量高光點綴鼻樑中央與鼻尖位置即可。透過自然光澤感呈現高級且精緻的妝容氛圍。"
    elif "肉感鼻" in features.noseShape:
        nose_desc = "建議於鼻翼外側及鼻頭周圍加強陰影修飾，適度弱化鼻部厚實感；搭配鼻樑中央提亮，使鼻型視覺上更加立體修長。"
    elif "翹鼻" in features.noseShape:
        nose_desc = "小翹鼻具有俏皮且年輕的特徵，可著重於山根與鼻樑中央的自然提亮，避免過度修容，保留鼻尖微翹所帶來的活潑感與親和力。"
    elif "鷹勾鼻" in features.noseShape:
        nose_desc = "建議減少鼻樑整段高光範圍，將提亮重點放在山根及鼻尖位置；鼻樑轉折處可輕微修飾，使鼻部線條看起來更柔和流暢。"
    elif "扁鼻" in features.noseShape:
        nose_desc = "建議加強山根、鼻樑中央及鼻尖的提亮效果，搭配鼻側陰影修飾，增加縱向輪廓感，打造更立體精緻的五官印象。"
    step1_final_text = f"{style_desc} {skin_desc} {nose_desc}"
    # --- 1. 準備給 Gemini 的提示詞 ---
    prompt_text = f"""
    你現在是一位頂級形象顧問。特徵：{features.faceShape},{features.eyeShape},{features.eyelidType}, 鼻型{features.noseShape}, 唇形{features.lipShape}, 膚色{features.skinTone}, 髮長{features.hairLength}, 風格{features.style}, 場合{features.occasion}。

    【🚨髮型命名強制規則🚨】：
    1. 髮型名稱長度：必須嚴格控制在 **4 到 7 個中文字符**。
    2. 語法要求：必須為完整、優雅的術語（例如：「側分大波浪」），嚴禁「縮寫」、「怪異名稱」、「英文」、「括號」。

    【✨妝容步驟要求 (必須詳細，禁止偷懶)✨】：
    - Step 1 (底妝與修容)：🚨請直接一字不漏地回傳此文字："{step1_final_text}" 🚨
    - Step 2 (眼妝設計)：針對「{features.eyelidType}」與「{features.eyeShape}」詳細說明。需包含：暈染範圍、眼線畫法、睫毛具體步驟。字數要求：必須在 80 字以上。
    - Step 3 (唇頰搭配)：針對「{features.lipShape}」與「{features.skinTone}」說明。需包含：腮紅位置/色系、口紅質地/色系。字數要求：必須在 80 字以上。

    【回傳格式 (JSON 嚴格鎖定)】：
    請嚴格依照 JSON 格式回傳，且在 makeup_steps 的 desc 欄位中，不要有任何簡化：
    {{
        "best_hairstyles": ["髮型1", "髮型2", "髮型3"],
        "avoid_hairstyles": ["地雷1", "地雷2"],
        "hair_colors": [
            {{"name": "色1", "desc": "色彩描述需包含層次感", "hex": "A67C52"}}, 
            {{"name": "色2", "desc": "色彩描述需包含層次感", "hex": "8B5A2B"}}, 
            {{"name": "色3", "desc": "色彩描述需包含層次感", "hex": "3E2723"}}
        ],
        "makeup_steps": [
            {{"title": "Step 1", "desc": "{step1_final_text}"}}, 
            {{"title": "Step 2", "desc": "這裡撰寫詳細眼妝，字數大於80字..."}}, 
            {{"title": "Step 3", "desc": "這裡撰寫詳細唇頰，字數大於80字..."}}
        ],
        "summary": "50字以上的專業總結，口吻要親切鼓勵。"
    }}
    """


    # --- 文字報告：Gemini，失敗就用備用報告 ---
    report_data = None
    if GEMINI_API_KEY:
        try:
            print("🧠 Gemini 正在撰寫文字報告...", flush=True)
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(
                prompt_text,
                generation_config={"temperature": 0.3, "response_mime_type": "application/json"},
            )
            report_data = json.loads(response.text)
            print("✅ Gemini 報告撰寫成功", flush=True)
        except Exception as e:
            print(f"⚠️ Gemini 遇到限制或錯誤，改用備用報告：{e}", flush=True)
    if not isinstance(report_data, dict) or len(report_data.get("makeup_steps", [])) < 3:
        report_data = copy.deepcopy(FALLBACK_REPORT)

    # --- 眼妝圖：OpenAI，失敗就不給圖（前端會用預設圖）---
    final_images = {}
    if openai_client is not None:
        try:
            eye_makeup_desc = report_data["makeup_steps"][1]["desc"]
            image_prompt = (
                "Professional macro beauty photography, extreme close-up of a beautiful East Asian woman's eyes. "
                f"Subject's eye features: {eyelid_en} and {eye_en}. "
                f"Makeup style description: {eye_makeup_desc}. "
                "Sharp focus on eye makeup, professional studio lighting, high resolution."
            )
            res = openai_client.images.generate(
                model=OPENAI_IMAGE_MODEL, prompt=image_prompt, n=1, size="1024x1024"
            )
            item = res.data[0]
            if getattr(item, "b64_json", None):
                final_images["step2_eye"] = f"data:image/png;base64,{item.b64_json}"
            elif getattr(item, "url", None):
                final_images["step2_eye"] = item.url
            print("✅ 專屬眼妝圖生成成功", flush=True)
        except Exception as e:
            print(f"⚠️ OpenAI 畫圖失敗或達上限，前端會用預設圖。原因：{e}", flush=True)

    report_data["images"] = final_images
    report_data["status"] = "success"
    return report_data


# ==========================================
# 3. 本機模式：由後端自己提供網頁與圖片
#    （放在 GitHub Pages 時，這一段用不到，資料夾不存在就自動略過）
#    網頁檔放在 static/，圖片放在 static/images/
# ==========================================
if os.path.isdir("static/images"):
    app.mount("/images", StaticFiles(directory="static/images"), name="images")
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
    if os.path.isfile("static/index.html"):
        app.mount("/", StaticFiles(directory="static", html=True), name="site")