# AI Beauty Advisor API

[AI Beauty Advisor](https://411277024.github.io/AI_beauty_project/) 的後端，使用 FastAPI 實作。負責接收照片與問卷條件，串接 Gemini 與 OpenAI，回傳個人化的髮型、髮色與妝容報告。

前端 repo：[AI_beauty_project](https://github.com/411277024/AI_beauty_project)

## API

| 方法 | 路徑 | 說明 |
|---|---|---|
| GET | `/api/health` | 健康檢查，同時回報兩把金鑰是否設定 |
| POST | `/api/analyze-photo` | 上傳照片（`multipart/form-data`，欄位 `file`），由 Gemini 辨識臉型、眼型、眼皮、鼻型、唇形 |
| POST | `/api/recommend` | 送出問卷條件，回傳個人化報告與眼妝示意圖 |

### `POST /api/recommend`

Request（JSON）：

```json
{
  "faceShape": "round",
  "eyeShape": "almond",
  "eyelidType": "double",
  "noseShape": "小巧鼻",
  "lipShape": "thin",
  "skinTone": "自然",
  "hairLength": "long",
  "style": "韓系溫柔",
  "occasion": "date"
}
```

Response：

```json
{
  "status": "success",
  "best_hairstyles": ["..."],
  "avoid_hairstyles": ["..."],
  "hair_colors": [{ "name": "焦糖奶茶色", "desc": "...", "hex": "A67C52" }],
  "makeup_steps": [{ "title": "Step 1", "desc": "..." }],
  "summary": "...",
  "images": { "step2_eye": "data:image/png;base64,..." }
}
```

## 設計重點

- **多層容錯**：Gemini 遇到 429 或其他錯誤時，改用備用報告；OpenAI 畫圖失敗時，`images` 回傳空物件，由前端使用預設圖。API 不會因為外部服務的限制而中斷。
- **代碼轉中文再送 AI**：前端送來的是 `round_eye`、`double` 這類代碼，後端先轉成中文再交給 Gemini，避免代碼直接出現在報告裡；畫圖則改用英文描述。
- **安全**
  - 金鑰只從環境變數讀取，程式中沒有任何金鑰
  - CORS 採白名單（`ALLOWED_ORIGINS`），不使用 `*`
  - 每個 IP 每小時請求上限（預設 30 次），超過回 HTTP 429
  - 上傳照片限制 8 MB，縮圖後才送分析，只在記憶體處理，不存檔

## 環境變數

| 名稱 | 必填 | 說明 |
|---|---|---|
| `GEMINI_API_KEY` | 是 | Google Gemini 金鑰（也接受 `GOOGLE_API_KEY`） |
| `OPENAI_API_KEY` | 否 | 沒有的話不產生眼妝圖，前端改用預設圖 |
| `GEMINI_MODEL` | 否 | 預設 `gemini-2.5-flash` |
| `OPENAI_IMAGE_MODEL` | 否 | 畫圖模型名稱 |
| `ALLOWED_ORIGINS` | 否 | 允許呼叫的網域，用逗號分開，預設包含本專案的 GitHub Pages |
| `RATE_LIMIT_PER_HOUR` | 否 | 每個 IP 每小時請求上限，預設 30 |

## 在本機執行

```bash
pip install -r requirements.txt
```

在專案資料夾建立 `.env`（**不要上傳 Git**）：

```
GEMINI_API_KEY=你的金鑰
OPENAI_API_KEY=你的金鑰
```

啟動：

```bash
uvicorn main:app --reload --port 8000
```

打開 <http://127.0.0.1:8000/api/health>，看到 `{"status":"ok","gemini":true,"openai":true}` 就成功了。

若在同一資料夾放 `static/`（網頁檔）與 `static/images/`（圖庫），後端會自己提供完整網站，直接開 <http://127.0.0.1:8000/>。

## 部署（Render）

| 設定 | 值 |
|---|---|
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| Environment | 設定上表的環境變數 |

免費方案閒置 15 分鐘會休眠，第一次請求需要 30～60 秒喚醒；前端在網頁開啟時會先呼叫 `/api/health` 叫醒後端。

## 隱私

照片會傳送到 Google Gemini 做分析，後端只在記憶體處理，不儲存。

## 技術

Python、FastAPI、Pydantic、Pillow、Gemini API、OpenAI API、Render

## 已知限制

`google-generativeai` 套件官方已停止更新，之後計畫改用 `google-genai`。
