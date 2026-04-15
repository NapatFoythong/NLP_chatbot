"""
LINE Chatbot — ระบบทำความเข้าใจข้อความลูกค้าภาษาไทย
รองรับ: พิมพ์ผิด | code-mixing | Intent | Sentiment | Entity Extraction

เครื่องมือและไลบรารีที่ใช้ (ตามรายงาน):
  - PyThaiNLP  : ตัดคำภาษาไทย / แก้คำพิมพ์ผิด / เติมวรรณยุกต์
  - Regex      : ตรวจจับรูปแบบข้อความ
  - Gensim     : Word2Vec / FastText (โหลด pre-trained เพื่อหาความใกล้เคียงเชิงความหมาย)
  - Scikit-learn: Logistic Regression สำหรับ Intent Classification
  - Flask      : Web server รับ Webhook จาก LINE

ติดตั้ง:
  pip install flask line-bot-sdk pythainlp gensim scikit-learn
"""

import os
import re

# ── Flask & LINE SDK ────────────────────────────────────────────────────────
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, FollowEvent

# ── PyThaiNLP ───────────────────────────────────────────────────────────────
from pythainlp.tokenize import word_tokenize          # ตัดคำ (Morphological Level)
from pythainlp.spell import correct as thai_spell     # แก้คำสะกดผิด

# ── Gensim (Word2Vec / FastText) ────────────────────────────────────────────
# ใช้วัดความใกล้เคียงเชิงความหมายของคำ (Lexical Similarity)
# รองรับ: โหลดโมเดล pre-trained หรือ train เองจาก corpus
try:
    from gensim.models import KeyedVectors, Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("[WARN] gensim ไม่ถูกติดตั้ง — ปิดฟีเจอร์ semantic similarity")

# ── Scikit-learn ────────────────────────────────────────────────────────────
# Logistic Regression สำหรับ Intent Classification (Document Classification)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# ══════════════════════════════════════════════════════════════════════════════
# ✏️  ใส่ค่าจาก LINE Developers Console
# ══════════════════════════════════════════════════════════════════════════════
CHANNEL_ACCESS_TOKEN = "q8iFhjH9a/ort7uxxt6OnAkMqoiiwcfnCjuDd3fBBdR12d9qsQRZv+bEZEbJMY7NIJCwYfOS6wrtovLn+Xgl+4kRABVQjOrx/vVSYQAZJ/WB1tvSnAkycx+BPkM6UBhtwHXG0QL5HANUXmYu03LZEgdB04t89/1O/w1cDnyilFU="
CHANNEL_SECRET       = "d2961ac14e172499e09556f9ee0055d8"
# ══════════════════════════════════════════════════════════════════════════════

app      = Flask(__name__)
line_bot = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler  = WebhookHandler(CHANNEL_SECRET)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5.2  MORPHOLOGICAL LEVEL — PyThaiNLP + Regex                          ║
# ║  แก้คำพิมพ์ผิด / คำไม่เป็นทางการ / เติมวรรณยุกต์                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# พจนานุกรมคำไม่เป็นทางการ → คำมาตรฐาน  (Regex-based lookup)
INFORMAL: dict[str, str] = {
    "มั้ย": "ไหม",
    "คับ":  "ครับ",
    "ครับบ": "ครับ",
    "ละยัง": "หรือยัง",
    "ค่า":  "ค่ะ",
    "จ้า":  "ค่ะ",
    "โอเค": "ตกลง",
    "เดี๋ยว": "สักครู่",
    "ป่าว": "หรือเปล่า",
    "ได้มั้ย": "ได้ไหม",
}

# Regex pattern: ตรวจจับคำซ้ำพยัญชนะท้าย เช่น "ครับบบ" → "ครับ"
_REPEAT_CHARS = re.compile(r'(.)\1{2,}')

def normalize_morphology(text: str) -> str:
    """
    5.2 Morphological Level
    ขั้นตอน:
      1. ลบอักขระซ้ำด้วย Regex
      2. แทนคำไม่เป็นทางการด้วยพจนานุกรม
      3. ใช้ PyThaiNLP spell-correct แก้คำสะกดผิดที่เหลือ
    """
    # (1) Regex — ลบอักขระซ้ำ เช่น "ครับบบ" → "ครับ"
    text = _REPEAT_CHARS.sub(r'\1\1', text)

    # (2) แทนคำไม่เป็นทางการ (ค้นหาทั้งประโยคด้วย word boundaries เชิง Thai)
    for informal, formal in INFORMAL.items():
        text = re.sub(re.escape(informal), formal, text)

    # (3) PyThaiNLP spell correction (ทีละคำ หลังตัดคำ)
    tokens  = word_tokenize(text, engine="newmm")
    corrected = []
    for tok in tokens:
        try:
            corrected.append(thai_spell(tok))
        except Exception:
            corrected.append(tok)
    return "".join(corrected)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5.3  LEXICAL LEVEL — Code-Mixing (EN→TH) + Gensim Similarity          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# พจนานุกรมแปล EN→TH ที่สร้างเอง  (Custom dictionary)
EN_TH: dict[str, str] = {
    "check":    "ตรวจสอบ",
    "stock":    "สินค้าคงเหลือ",
    "order":    "สั่งซื้อ",
    "tracking": "ติดตามพัสดุ",
    "cancel":   "ยกเลิก",
    "transfer": "โอนเงิน",
    "size":     "ขนาด",
    "color":    "สี",
    "price":    "ราคา",
    "confirm":  "ยืนยัน",
    "pay":      "ชำระเงิน",
    "refund":   "คืนเงิน",
}

# Regex — ตรวจจับ token ภาษาอังกฤษ
_EN_TOKEN = re.compile(r'[A-Za-z]+')

def normalize_lexical(text: str) -> str:
    """
    5.3 Lexical Level
    - แปลคำอังกฤษใน code-mixed text เป็นภาษาไทย
    - ใช้ Gensim หาคำใกล้เคียงหากไม่มีใน dictionary (optional)
    """
    tokens = text.split()
    result = []
    for tok in tokens:
        if _EN_TOKEN.fullmatch(tok):
            mapped = EN_TH.get(tok.lower())
            if mapped:
                result.append(mapped)
            elif GENSIM_AVAILABLE and _w2v_model:
                # ถ้าไม่อยู่ใน dict → ลองหาคำใกล้เคียงด้วย Word2Vec
                try:
                    similar = _w2v_model.most_similar(tok.lower(), topn=1)
                    result.append(similar[0][0])
                except Exception:
                    result.append(tok)
            else:
                result.append(tok)
        else:
            result.append(tok)
    return " ".join(result)

# โหลด Gensim Word2Vec model (ถ้ามีไฟล์ pre-trained)
_w2v_model = None
_W2V_PATH  = os.environ.get("W2V_MODEL_PATH", "")   # ระบุ path ผ่าน env var
if GENSIM_AVAILABLE and _W2V_PATH and os.path.exists(_W2V_PATH):
    try:
        _w2v_model = KeyedVectors.load_word2vec_format(_W2V_PATH, binary=True)
        print(f"[INFO] โหลด Word2Vec model จาก {_W2V_PATH} สำเร็จ")
    except Exception as e:
        print(f"[WARN] โหลด Word2Vec ไม่สำเร็จ: {e}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5.4  INTENT CLASSIFICATION — Scikit-learn (Logistic Regression)        ║
# ║  Document Classification ตาม section 9 ของรายงาน                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Training data (ข้อความตัวอย่าง → label)  — ขยายชุดข้อมูลจาก Data Preparation
_TRAIN_TEXTS = [
    # สั่งซื้อสินค้า
    "สั่งของได้ไหมครับ", "จะเอาตัวนี้เลย", "ขอสั่งซื้อด้วย",
    "โอนเงินแล้วนะครับ", "เอาเสื้อตัวนี้", "ยืนยันการสั่งซื้อ",
    # สอบถามสินค้า
    "มีไซส์ M ไหม", "ราคาเท่าไหร่", "มีสีอะไรบ้าง",
    "ขอตรวจสอบสินค้าคงเหลือหน่อย", "รุ่นนี้มีกี่แบบ", "มีสีดำไหมครับ",
    # ติดตามการจัดส่ง
    "ของมาหรือยัง", "พัสดุถึงไหนแล้ว", "เลขพัสดุคืออะไร",
    "รอนานมากยังไม่ได้ของเลย", "ส่งให้เมื่อไหร่", "ติดตามพัสดุหน่อย",
    # ยกเลิก/เปลี่ยน
    "ขอยกเลิกออเดอร์", "เปลี่ยนสีได้ไหม", "ขอคืนสินค้า", "ไม่เอาแล้วครับ",
]
_TRAIN_LABELS = (
    ["🛒 สั่งซื้อสินค้า"]    * 6 +
    ["🔍 สอบถามสินค้า"]     * 6 +
    ["🚚 ติดตามการจัดส่ง"]  * 6 +
    ["❌ ยกเลิก/เปลี่ยน"]   * 4
)

# Pipeline: TF-IDF (character n-gram เพื่อรองรับภาษาไทย) + Logistic Regression
_intent_clf = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))),
    ("clf",   LogisticRegression(max_iter=500, C=5.0)),
])
_intent_clf.fit(_TRAIN_TEXTS, _TRAIN_LABELS)

def classify_intent(text: str) -> tuple[str, float]:
    """คืนค่า (intent_label, confidence_score)"""
    proba = _intent_clf.predict_proba([text])[0]
    idx   = proba.argmax()
    label = _intent_clf.classes_[idx]
    conf  = proba[idx]
    return label, conf


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5.5  SENTIMENT ANALYSIS                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Lexicon-based (สามารถต่อยอดด้วย WangchanBERTa / Wisesight Corpus ได้)
_POS_WORDS = {"ขอบคุณ", "ดีมาก", "ชอบ", "เยี่ยม", "ประทับใจ", "สวย", "ดีเลย"}
_NEG_WORDS = {"รอนาน", "แย่", "ไม่ได้", "ผิดหวัง", "ช้ามาก", "หาย", "เสีย", "พัง"}

def analyze_sentiment(tokens: list[str]) -> str | None:
    tok_set = set(tokens)
    if tok_set & _NEG_WORDS:
        return "😟 เชิงลบ"
    if tok_set & _POS_WORDS:
        return "😊 เชิงบวก"
    return None


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5.6  ENTITY EXTRACTION — Regex + PyThaiNLP tokenizer                  ║
# ║  สกัด: สินค้า / สี / ขนาด / จำนวน  (ตามตัวอย่างในรายงาน)              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

_SIZE_PATTERN  = re.compile(r'\b(XS|S|M|L|XL|XXL|\d+)\b', re.IGNORECASE)
_QTY_PATTERN   = re.compile(r'(\d+)\s*(ตัว|ชิ้น|อัน|คู่|แพ็ค|โหล)')
_COLOR_KEYWORDS = {
    "ดำ": "ดำ", "ขาว": "ขาว", "แดง": "แดง", "น้ำเงิน": "น้ำเงิน",
    "เขียว": "เขียว", "เหลือง": "เหลือง", "ชมพู": "ชมพู", "เทา": "เทา",
}
_PRODUCT_KEYWORDS = {
    "เสื้อ", "กางเกง", "รองเท้า", "กระเป๋า", "หมวก", "แจ็คเก็ต",
    "เดรส", "กระโปรง", "เข็มขัด", "นาฬิกา",
}

def extract_entities(text: str, tokens: list[str]) -> dict:
    entities: dict[str, list] = {}

    # สินค้า — ตรวจจับจาก keyword list
    products = [tok for tok in tokens if tok in _PRODUCT_KEYWORDS]
    if products:
        entities["สินค้า"] = products

    # สี — keyword matching
    colors = [COLOR for kw, COLOR in _COLOR_KEYWORDS.items() if kw in text]
    if colors:
        entities["สี"] = colors

    # ขนาด — Regex
    sizes = _SIZE_PATTERN.findall(text)
    if sizes:
        entities["ขนาด"] = sizes

    # จำนวน — Regex  เช่น "2 ตัว"
    quantities = [f"{m[0]} {m[1]}" for m in _QTY_PATTERN.findall(text)]
    if quantities:
        entities["จำนวน"] = quantities

    return entities


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PIPELINE MASTER — รวมทุกขั้นตอน                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def process(raw_text: str) -> str:
    """
    รัน full NLP pipeline และคืน reply string
    """
    # ── 5.2 Morphological Level ──────────────────────────────────────────
    normalized = normalize_morphology(raw_text)

    # ── 5.3 Lexical Level ────────────────────────────────────────────────
    normalized = normalize_lexical(normalized)

    # ── Tokenize (PyThaiNLP) ─────────────────────────────────────────────
    tokens = word_tokenize(normalized, engine="newmm", keep_whitespace=False)

    # ── 5.4 Intent Classification (Scikit-learn) ─────────────────────────
    intent, confidence = classify_intent(normalized)
    conf_pct = int(confidence * 100)

    # ── 5.5 Sentiment Analysis ────────────────────────────────────────────
    sentiment = analyze_sentiment(tokens)

    # ── 5.6 Entity Extraction ─────────────────────────────────────────────
    entities = extract_entities(normalized, tokens)

    # ── Build Reply ───────────────────────────────────────────────────────
    lines = [
        f"📝 ข้อความที่แก้ไขแล้ว: {normalized}",
        f"🎯 Intent: {intent} ({conf_pct}%)",
    ]
    if sentiment:
        lines.append(f"💬 Sentiment: {sentiment}")
    if entities:
        ent_str = " | ".join(f"{k}: {', '.join(v)}" for k, v in entities.items())
        lines.append(f"🏷️ Entities: {ent_str}")

    return "\n".join(lines)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Flask Webhook                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@app.route("/webhook", methods=["POST"])
def webhook():
    sig  = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, sig)
    except InvalidSignatureError:
        abort(400)
    return "OK"


@handler.add(MessageEvent, message=TextMessage)
def on_message(event):
    reply = process(event.message.text)
    line_bot.reply_message(event.reply_token, TextSendMessage(text=reply))


@handler.add(FollowEvent)
def on_follow(event):
    line_bot.reply_message(
        event.reply_token,
        TextSendMessage(text=(
            "สวัสดีครับ! 👋 ระบบวิเคราะห์ข้อความลูกค้า\n"
            "ลองพิมพ์ดูเลยครับ เช่น\n"
            "• สั่งของได้มั้ยคับ\n"
            "• มีไซส์ M มั้ย\n"
            "• ของมาละยัง\n"
            "• ขอ check stock หน่อย\n"
            "• สั่งเสื้อสีดำไซส์ L 2 ตัว"
        ))
    )


@app.route("/")
def health():
    return "OK"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
