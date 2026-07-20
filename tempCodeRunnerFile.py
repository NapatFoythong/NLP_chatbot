"""
LINE Chatbot — ระบบทำความเข้าใจข้อความลูกค้าภาษาไทย
ใช้ PyThaiNLP เต็มประสิทธิภาพ:
  - word_tokenize   : ตัดคำภาษาไทย
  - spell_candidates: แก้คำสะกดผิดแม่นขึ้น
  - pos_tag         : วิเคราะห์ชนิดคำ → ช่วย Entity Extraction
  - sentiment       : วิเคราะห์อารมณ์จาก Wisesight Corpus

ติดตั้ง:
  pip install flask "line-bot-sdk>=3.0.0" pythainlp[full] scikit-learn gensim
"""

import os
import re

# ── Flask ────────────────────────────────────────────────────────────────────
from flask import Flask, request, abort

# ── LINE SDK v3 ──────────────────────────────────────────────────────────────
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi,
    ReplyMessageRequest, TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent, FollowEvent

# ── PyThaiNLP (ใช้เต็มประสิทธิภาพ) ─────────────────────────────────────────
from pythainlp.tokenize import word_tokenize          # ตัดคำ
from pythainlp.spell import correct as thai_spell_correct  # แก้คำสะกดผิด
from pythainlp.tag import pos_tag                     # POS Tagging → ช่วย Entity
try:
    from pythainlp.classify import GradientBoostingClassifier
    _sentiment_model = GradientBoostingClassifier()
    def thai_sentiment(text: str) -> str:
        return _sentiment_model.predict(text)
except Exception:
    thai_sentiment = None

# ── Scikit-learn ─────────────────────────────────────────────────────────────
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ── Gensim ───────────────────────────────────────────────────────────────────
try:
    from gensim.models import KeyedVectors
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# ✏️  ใส่ค่าจาก LINE Developers Console
# ══════════════════════════════════════════════════════════════════════════════
CHANNEL_ACCESS_TOKEN = "q8iFhjH9a/ort7uxxt6OnAkMqoiiwcfnCjuDd3fBBdR12d9qsQRZv+bEZEbJMY7NIJCwYfOS6wrtovLn+Xgl+4kRABVQjOrx/vVSYQAZJ/WB1tvSnAkycx+BPkM6UBhtwHXG0QL5HANUXmYu03LZEgdB04t89/1O/w1cDnyilFU="
CHANNEL_SECRET       = "d2961ac14e172499e09556f9ee0055d8"
# ══════════════════════════════════════════════════════════════════════════════

app           = Flask(__name__)
configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
handler       = WebhookHandler(CHANNEL_SECRET)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5.2  MORPHOLOGICAL LEVEL                                               ║
# ║  Regex ลบตัวซ้ำ → INFORMAL dict → PyThaiNLP spell_candidates           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

INFORMAL = {
    "มั้ย": "ไหม",   "คับ": "ครับ",     "ครับบ": "ครับ",
    "ละยัง": "หรือยัง", "ค่า": "ค่ะ",   "จ้า": "ค่ะ",
    "โอเค": "ตกลง",  "เดี๋ยว": "สักครู่", "ป่าว": "หรือเปล่า",
    "ได้มั้ย": "ได้ไหม", "นะคะ": "นะครับ", "อ่ะ": "",
    "อ่ะครับ": "ครับ", "เนอะ": "", "หนะ": "",
    # แสลง
    "โคตร": "มาก",   "ห่วยแตก": "แย่มาก", "ห่วย": "แย่",
    "วะ": "",         "เนี่ย": "",          "อ่ะ": "",
    "ปิ๊ง": "ชอบ",   "โกย": "สั่งซื้อ",   "ติดใจ": "ชอบ",
}

_REPEAT_CHARS = re.compile(r'(.)\1{2,}')

# คำที่ไม่ควรแก้ด้วย spell (คำแสลง / ชื่อเฉพาะ)
_SKIP_SPELL = {"M", "L", "XL", "S", "XS", "XXL", "check", "stock", "order"}

def normalize_morphology(text: str) -> str:
    """
    ขั้นตอน:
    1. Regex — ลบอักขระซ้ำ เช่น "ครับบบ" → "ครับ"
    2. INFORMAL dict — แทนคำไม่เป็นทางการ
    3. PyThaiNLP spell_candidates — แก้คำสะกดผิดที่เหลือ
    """
    # (1) ลบอักขระซ้ำ
    text = _REPEAT_CHARS.sub(r'\1\1', text)

    # (2) แทนคำไม่เป็นทางการ
    for informal, formal in INFORMAL.items():
        text = re.sub(re.escape(informal), formal, text)
    text = text.strip()

    # (3) ใช้แค่ word_tokenize — ไม่ spell correct เพราะช้าเกินไป
    return text


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5.3  LEXICAL LEVEL — Code-Mixing EN→TH                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

EN_TH = {
    "check": "ตรวจสอบ",   "stock": "สินค้าคงเหลือ",
    "order": "สั่งซื้อ",   "tracking": "ติดตามพัสดุ",
    "cancel": "ยกเลิก",   "transfer": "โอนเงิน",
    "size": "ขนาด",       "color": "สี",
    "price": "ราคา",      "confirm": "ยืนยัน",
    "pay": "ชำระเงิน",    "refund": "คืนเงิน",
    "sale": "ลดราคา",     "sold": "ขายหมด",
}

_EN_TOKEN = re.compile(r'^[A-Za-z]+$')

def normalize_lexical(text: str) -> str:
    """แปลคำอังกฤษใน code-mixed text เป็นภาษาไทย"""
    tokens = text.split()
    return " ".join(EN_TH.get(t.lower(), t) if _EN_TOKEN.match(t) else t for t in tokens)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5.4  INTENT CLASSIFICATION — Scikit-learn TF-IDF + Logistic Regression ║
# ╚══════════════════════════════════════════════════════════════════════════╝

_TRAIN_TEXTS = [
    # 🛒 สั่งซื้อสินค้า
    "สั่งของได้ไหมครับ", "จะเอาตัวนี้เลย", "ขอสั่งซื้อด้วย",
    "โอนเงินแล้วนะครับ", "เอาเสื้อตัวนี้", "ยืนยันการสั่งซื้อ",
    "ขอสั่งเลยได้ไหม", "จะซื้อตัวนี้", "สั่งได้เลยไหมครับ",
    "ขอซื้อเลยนะครับ", "จะเอาเลยครับ", "ซื้อได้เลยไหม",
    # แสลง สั่งซื้อ
    "จะเอาเลยอ่ะ", "ปิ๊งเลยจะเอาตัวนี้", "ติดใจอ่ะเอาเลยได้มั้ย",
    "ขอโกยเลยได้มั้ย", "เอาเลยอ่ะ", "สั่งเลยได้เลยมั้ยวะ",
    # 🔍 สอบถามสินค้า
    "มีไซส์ M ไหม", "ราคาเท่าไหร่", "มีสีอะไรบ้าง",
    "ขอตรวจสอบสินค้าคงเหลือหน่อย", "รุ่นนี้มีกี่แบบ", "มีสีดำไหมครับ",
    "ของชิ้นนี้ราคาเท่าไหร่", "มีสินค้าไหม", "ขอดูรายละเอียดหน่อย",
    "มีของพร้อมส่งไหม", "สินค้ามีกี่สี", "ขอดูรูปสินค้าหน่อย",
    # แสลง สอบถาม
    "เท่าไหร่วะครับ", "ตัวนี้เหลือมั้ยอ่ะ", "มีสีอื่นมั้ยอ่ะ",
    "ราคาเท่าไหร่วะ ถูกมั้ย", "ของดีมั้ยอ่ะ คุ้มมั้ย",
    "แพงจังเลย มีถูกกว่านี้มั้ย", "อยากได้แต่แพงอ่ะ มีโปรมั้ย",
    # 🚚 ติดตามการจัดส่ง
    "ของมาหรือยัง", "พัสดุถึงไหนแล้ว", "เลขพัสดุคืออะไร",
    "ยังไม่ได้ของเลย", "ส่งให้เมื่อไหร่", "ติดตามพัสดุหน่อย",
    "ของยังไม่มาเลย", "พัสดุอยู่ที่ไหน", "ส่งของวันไหนครับ",
    "ของหายไหมครับ", "ขนส่งอัปเดตล่าสุดเมื่อไหร่", "ของค้างอยู่ที่ไหน",
    "ขอตามสถานะการจัดส่งครับ", "ขอเช็คสถานะพัสดุครับ", "ตามสถานะออเดอร์หน่อย",
    "สถานะการจัดส่งเป็นยังไงครับ", "ขอดูสถานะพัสดุ", "ของส่งไปหรือยังครับ",
    "เช็คสถานะการส่งของหน่อย", "อยากทราบสถานะการจัดส่ง", "ของถึงไหนแล้วครับ",
    # แสลง ติดตาม
    "ของไปไหนวะ ไม่มาเลย", "รอจนเบื่อแล้วอ่ะ ของถึงไหน",
    "พัสดุหายไปไหนเนี่ย", "รอมาเป็นอาทิตย์แล้ว ของไปไหน",
    # ❌ ยกเลิก/เปลี่ยน
    "ขอยกเลิกออเดอร์", "เปลี่ยนสีได้ไหม", "ขอคืนสินค้า",
    "ไม่เอาแล้วครับ", "ขอยกเลิกการสั่งซื้อ", "เปลี่ยนไซส์ได้ไหม",
    "ขอเปลี่ยนที่อยู่จัดส่ง", "ขอยกเลิกด่วน", "คืนเงินได้ไหม",
    # 😤 ร้องเรียน/ไม่พอใจ
    "ช้ามากครับ", "บริการแย่มาก", "ผิดหวังมาก", "รอนานมาก",
    "ทำไมช้าจัง", "ไม่พอใจเลย", "แย่มากเลย", "รอมานานมากแล้ว",
    "ทำไมถึงช้าขนาดนี้", "ไม่ประทับใจเลย", "บริการห่วยมาก", "งานแย่มาก",
    # แสลง ร้องเรียน
    "โคตรช้าเลย รอมาเป็นอาทิตย์แล้ว", "ห่วยแตกมากเลยครับ",
    "แย่สุดๆเลยอ่ะ ไม่ประทับใจเลย", "บริการห่วยโคตรๆ",
    "โคตรช้า รับไม่ได้เลย", "แย่มากวะ ไม่ประทับใจ",
]
_TRAIN_LABELS = (
    ["🛒 สั่งซื้อสินค้า"]     * 18 +
    ["🔍 สอบถามสินค้า"]      * 19 +
    ["🚚 ติดตามการจัดส่ง"]   * 25 +
    ["❌ ยกเลิก/เปลี่ยน"]    * 9  +
    ["😤 ร้องเรียน/ไม่พอใจ"] * 18
)

_intent_clf = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))),
    ("clf",   LogisticRegression(max_iter=1000, C=5.0)),
])
_intent_clf.fit(_TRAIN_TEXTS, _TRAIN_LABELS)

def classify_intent(text: str) -> tuple[str, int]:
    proba = _intent_clf.predict_proba([text])[0]
    idx   = proba.argmax()
    return _intent_clf.classes_[idx], int(proba[idx] * 100)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5.5  SENTIMENT ANALYSIS — PyThaiNLP (Wisesight Corpus)                ║
# ║  ใช้ PyThaiNLP sentiment() โดยตรง แม่นกว่า keyword list               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

_POS_WORDS = {"ขอบคุณ", "ดีมาก", "ชอบ", "เยี่ยม", "ประทับใจ", "สวย", "ดีเลย", "พอใจ"}
_NEG_WORDS = {"รอนาน", "แย่", "ผิดหวัง", "ช้ามาก", "ช้าจัง", "ไม่พอใจ", "แย่มาก", "นานมาก"}

def analyze_sentiment(text: str) -> str | None:
    # (1) ลองใช้ PyThaiNLP GradientBoosting model ก่อน
    if thai_sentiment:
        try:
            result = thai_sentiment(text)
            if result == "neg":
                return "😟 เชิงลบ"
            elif result == "pos":
                return "😊 เชิงบวก"
        except Exception:
            pass

    # (2) Fallback — keyword matching
    if any(w in text for w in _NEG_WORDS):
        return "😟 เชิงลบ"
    if any(w in text for w in _POS_WORDS):
        return "😊 เชิงบวก"
    return None


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5.6  ENTITY EXTRACTION — PyThaiNLP POS Tag + Regex                    ║
# ║  ใช้ pos_tag ดึงคำนาม (สินค้า) แทน keyword list อย่างเดียว            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

_SIZE_PATTERN   = re.compile(r'\b(XS|S|M|L|XL|XXL|\d{1,3})\b', re.IGNORECASE)
_QTY_PATTERN    = re.compile(r'(\d+)\s*(ตัว|ชิ้น|อัน|คู่|แพ็ค|โหล)')
_COLOR_KEYWORDS = {"ดำ", "ขาว", "แดง", "น้ำเงิน", "เขียว", "เหลือง", "ชมพู", "เทา", "ม่วง", "ส้ม"}
_PRODUCT_KEYWORDS = {
    "เสื้อ", "กางเกง", "รองเท้า", "กระเป๋า", "หมวก", "แจ็คเก็ต",
    "เดรส", "กระโปรง", "เข็มขัด", "นาฬิกา", "ถุงเท้า", "เสื้อยืด",
}

def extract_entities(text: str, tokens: list[str]) -> dict:
    """
    รวม 2 วิธี:
    1. PyThaiNLP POS Tag — ดึงคำนามที่ไม่อยู่ใน keyword list
    2. Regex — ดึงขนาด / จำนวน
    """
    entities: dict = {}

    # (1) POS Tag → ดึงคำนามทั่วไป (NCMN) และคำนามเฉพาะ (NTTL)
    try:
        tagged   = pos_tag(tokens, engine="perceptron", corpus="orchid")
        # กรอง: ต้องเป็นคำนาม และความยาว > 1 ตัวอักษร
        pos_nouns = [w for w, tag in tagged if tag in ("NCMN", "NTTL") and len(w) > 1]
    except Exception:
        pos_nouns = []

    # รวม keyword list + POS nouns แล้วกรองเอาเฉพาะที่เกี่ยวกับสินค้า
    products = list({t for t in tokens if t in _PRODUCT_KEYWORDS} |
                    {n for n in pos_nouns if n in _PRODUCT_KEYWORDS})
    if products:
        entities["สินค้า"] = products

    # (2) สี — keyword matching
    colors = [c for c in _COLOR_KEYWORDS if c in text]
    if colors:
        entities["สี"] = colors

    # (3) ขนาด — Regex
    sizes = _SIZE_PATTERN.findall(text)
    if sizes:
        entities["ขนาด"] = list(dict.fromkeys(sizes))  # กรองซ้ำ

    # (4) จำนวน — Regex
    qtys = [f"{m[0]} {m[1]}" for m in _QTY_PATTERN.findall(text)]
    if qtys:
        entities["จำนวน"] = qtys

    return entities


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PIPELINE MASTER                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def process(raw_text: str) -> str:
    raw_lower = raw_text.strip().lower()

    # ── ตรวจจับข้อความขอบคุณก่อนเลย ─────────────────────────────────────────
    _THANK_WORDS = [
        "ขอบคุณ", "ขอบคุณครับ", "ขอบคุณค่ะ", "ขอบคุณมาก",
        "ขอบใจ", "thank you", "thanks", "thx", "ty", "thank",
        "ขอบคุณนะครับ", "ขอบคุณนะคะ", "ขอบคุณมากครับ",
    ]
    if any(w in raw_lower for w in _THANK_WORDS):
        return (
            "ขอบคุณเช่นกันครับ 😊🙏\n"
            "ยินดีให้บริการเสมอนะครับ\n"
            "มีอะไรให้ช่วยเหลือเพิ่มเติมได้เลยครับ"
        )

    # ── ตรวจจับข้อความทักทาย ─────────────────────────────────────────────────
    _GREET_WORDS = {
        "สวัสดี", "หวัดดี", "ดีจ้า", "ดีครับ", "ดีค่ะ", "hello", "hi",
        "hey", "hii", "หวัดดีครับ", "หวัดดีค่ะ", "สวัสดีครับ", "สวัสดีค่ะ",
        "ดีนะ", "ดีๆ", "โอฮาโย", "ฮัลโหล",
    }
    if any(w in raw_lower for w in _GREET_WORDS):
        return (
            "สวัสดีครับ! 😊 ยินดีต้อนรับครับ\n"
            "มีอะไรให้ช่วยเหลือไหมครับ?\n\n"
            "บริการของเรา:\n"
            "🛒 สั่งซื้อสินค้า\n"
            "🔍 สอบถามข้อมูลสินค้า\n"
            "📦 ติดตามการจัดส่ง\n"
            "❌ ยกเลิก/เปลี่ยนแปลงออเดอร์\n\n"
            "พิมพ์แจ้งความต้องการได้เลยนะครับ 🙏"
        )

    # 5.2 Morphological
    normalized = normalize_morphology(raw_text)
    # 5.3 Lexical
    normalized = normalize_lexical(normalized)
    # Tokenize
    tokens = word_tokenize(normalized, engine="newmm", keep_whitespace=False)

    # ── ตรวจจับ check stock โดยตรง ก่อน intent classifier ──────────────────
    _CHECK_PATTERNS = ["ตรวจสอบ สินค้าคงเหลือ", "ตรวจสอบสินค้า", "check stock",
                       "เช็คสต็อก", "เช็คสินค้า", "ดูสต็อก", "สต็อกเหลือไหม"]
    is_check_stock = any(p in normalized.lower() or p in raw_text.lower()
                         for p in _CHECK_PATTERNS)

    # ── ตรวจจับการติดตามการจัดส่งโดยตรง ────────────────────────────────────
    _TRACKING_PATTERNS = [
        "สถานะการจัดส่ง", "ตามสถานะ", "เช็คสถานะ", "ดูสถานะ",
        "สถานะพัสดุ", "สถานะออเดอร์", "สถานะการส่ง",
        "ติดตามสถานะ", "ขอตาม", "ของถึงไหน", "พัสดุถึงไหน",
        "order ถึงไหน", "สั่งซื้อถึงไหน", "tracking number",
        "ติดตาม order", "เช็ค order",
    ]
    is_tracking = any(p in normalized.lower() or p in raw_text.lower()
                      for p in _TRACKING_PATTERNS)

    # ── ตรวจจับยกเลิก/เปลี่ยน (รองรับ EN คำ เช่น cancel, refund) ──────────
    _CANCEL_PATTERNS = [
        "cancel", "refund", "ยกเลิก", "คืนเงิน", "คืนสินค้า",
        "เปลี่ยนสินค้า", "เปลี่ยนไซส์", "เปลี่ยนสี", "ไม่เอาแล้ว",
        "ขอยกเลิก", "ยกเลิก order", "cancel order",
    ]
    is_cancel = any(p in normalized.lower() or p in raw_text.lower()
                    for p in _CANCEL_PATTERNS)

    # ── ตรวจจับร้องเรียนแบบแสลง ─────────────────────────────────────────────
    _COMPLAINT_SLANG = [
        "โคตรช้า", "ห่วยแตก", "ห่วยโคตร", "แย่สุดๆ",
        "รอมาเป็นอาทิตย์", "รอมาเป็นวัน", "รับไม่ได้เลย",
        "ไม่ประทับใจเลย", "บริการห่วย", "โกรธมาก",
    ]
    is_complaint_slang = any(p in raw_text for p in _COMPLAINT_SLANG)

    # ── ตรวจจับสอบถามราคา/สินค้าแบบแสลง ────────────────────────────────────
    _INQUIRY_SLANG = [
        "แพงจัง", "ถูกมั้ย", "คุ้มมั้ย", "ของดีมั้ย",
        "มีโปรมั้ย", "อยากได้แต่", "เท่าไหร่วะ",
        "ตัวนี้เหลือมั้ย", "มีสีอื่นมั้ย",
    ]
    is_inquiry_slang = any(p in raw_text for p in _INQUIRY_SLANG)

    # 5.4 Intent
    intent, conf = classify_intent(normalized)

    # ถ้าตรวจจับ check stock ได้ → บังคับเป็นสอบถามสินค้า
    if is_check_stock:
        intent = "🔍 สอบถามสินค้า"
    # ถ้าตรวจจับร้องเรียนแสลงได้ → บังคับเป็นร้องเรียน
    elif is_complaint_slang:
        intent = "😤 ร้องเรียน/ไม่พอใจ"
    # ถ้าตรวจจับสอบถามแสลงได้ → บังคับเป็นสอบถาม
    elif is_inquiry_slang:
        intent = "🔍 สอบถามสินค้า"
    # ถ้าตรวจจับยกเลิก/เปลี่ยนได้ → บังคับเป็นยกเลิก/เปลี่ยน
    elif is_cancel:
        intent = "❌ ยกเลิก/เปลี่ยน"
    # ถ้าตรวจจับติดตามการจัดส่งได้ → บังคับเป็นติดตามการจัดส่ง
    elif is_tracking:
        intent = "🚚 ติดตามการจัดส่ง"

    # 5.5 Sentiment
    sentiment = analyze_sentiment(normalized)
    # 5.6 Entity
    entities = extract_entities(normalized, tokens)

    # ── ตรวจจับว่าลูกค้าโอนเงินแล้ว ─────────────────────────────────────────
    _PAYMENT_PATTERNS = ["โอนเงินแล้ว", "โอนแล้ว", "จ่ายแล้ว", "ชำระแล้ว",
                         "โอนให้แล้ว", "โอนเงินไปแล้ว", "สลิปโอน"]
    is_paid = any(p in normalized or p in raw_text for p in _PAYMENT_PATTERNS)

    # ── ตอบกลับตาม Intent (พร้อมบอก Intent ให้ลูกค้าทราบ) ─────────────────
    if intent == "🛒 สั่งซื้อสินค้า":
        if is_paid:
            reply = (
                "🎯 ประเภท: สั่งซื้อสินค้า\n\n"
                "ขอบคุณที่มาอุดหนุนนะครับ 🙏😊\n"
                "ทีมงานได้รับการแจ้งชำระเงินแล้ว\n"
                "จะรีบดำเนินการจัดส่งให้โดยเร็วที่สุดเลยครับ 📦"
            )
        else:
            reply = (
                "🎯 ประเภท: สั่งซื้อสินค้า\n\n"
                "ขอบคุณที่สนใจสั่งซื้อสินค้านะครับ 😊\n"
                "รบกวนแจ้งรายละเอียดสินค้าที่ต้องการ (ชื่อ/ไซส์/สี/จำนวน) ได้เลยครับ"
            )

    elif intent == "🔍 สอบถามสินค้า":
        reply = (
            "🎯 ประเภท: สอบถามสินค้า\n\n"
            "ยินดีให้ข้อมูลสินค้าครับ 🙏\n"
            "รบกวนระบุชื่อสินค้าหรือรายละเอียดที่ต้องการทราบได้เลยนะครับ\n"
            "ทีมงานจะรีบตรวจสอบและแจ้งกลับโดยเร็วครับ"
        )

    elif intent == "🚚 ติดตามการจัดส่ง":
        reply = (
            "🎯 ประเภท: ติดตามการจัดส่ง\n\n"
            "รับทราบครับ 📦\n"
            "รบกวนแจ้งชื่อหรือเลขออเดอร์เพื่อให้ทีมงานตรวจสอบสถานะพัสดุให้ได้เลยนะครับ"
        )

    elif intent == "❌ ยกเลิก/เปลี่ยน":
        reply = (
            "🎯 ประเภท: ยกเลิก/เปลี่ยนแปลง\n\n"
            "รับทราบครับ 🙏\n"
            "รบกวนแจ้งเลขออเดอร์และรายละเอียดที่ต้องการยกเลิก/เปลี่ยนแปลง\n"
            "ทีมงานจะดำเนินการให้โดยเร็วครับ"
        )

    elif intent == "😤 ร้องเรียน/ไม่พอใจ":
        reply = (
            "🎯 ประเภท: ร้องเรียน/ไม่พอใจ\n\n"
            "ต้องขออภัยในความไม่สะดวกอย่างสูงนะครับ 🙏\n"
            "รบกวนแจ้งรายละเอียดเพิ่มเติม ทีมงานจะรีบดำเนินการแก้ไขให้โดยเร็วที่สุดครับ"
        )

    else:
        reply = (
            "ขอบคุณที่ติดต่อมานะครับ 😊\n"
            "มีอะไรให้ช่วยเหลือไหมครับ? สามารถสอบถามได้เลยครับ"
        )

    # ── เพิ่มข้อความพิเศษถ้า Sentiment เชิงลบ ────────────────────────────
    if sentiment == "😟 เชิงลบ" and intent != "😤 ร้องเรียน/ไม่พอใจ":
        reply += "\n\nต้องขออภัยที่ทำให้ไม่สะดวกนะครับ 🙏 ทีมงานพร้อมช่วยเหลือครับ"

    # ── แสดง Entity ที่ดึงได้ เพื่อยืนยันความเข้าใจ ───────────────────────
    if entities:
        ent_lines = []
        if "สินค้า" in entities:
            ent_lines.append(f"สินค้า: {', '.join(entities['สินค้า'])}")
        if "สี" in entities:
            ent_lines.append(f"สี: {', '.join(entities['สี'])}")
        if "ขนาด" in entities:
            ent_lines.append(f"ไซส์: {', '.join(entities['ขนาด'])}")
        if "จำนวน" in entities:
            ent_lines.append(f"จำนวน: {', '.join(entities['จำนวน'])}")
        if ent_lines:
            reply += "\n\n📋 รับทราบรายละเอียด: " + " | ".join(ent_lines)

    return reply


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Flask Webhook                                                          ║
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


@handler.add(MessageEvent, message=TextMessageContent)
def on_message(event):
    reply_text = process(event.message.text)
    with ApiClient(configuration) as api_client:
        MessagingApi(api_client).reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text)],
            )
        )


@handler.add(FollowEvent)
def on_follow(event):
    with ApiClient(configuration) as api_client:
        MessagingApi(api_client).reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=(
                    "สวัสดีครับ! 👋 ระบบวิเคราะห์ข้อความลูกค้า\n"
                    "ลองพิมพ์ดูเลยครับ เช่น\n"
                    "• สั่งของได้มั้ยคับ\n"
                    "• มีไซส์ M มั้ย\n"
                    "• ของมาละยัง\n"
                    "• ขอ check stock หน่อย\n"
                    "• สั่งเสื้อสีดำไซส์ L 2 ตัว\n"
                    "• ช้ามากเลยครับ"
                ))],
            )
        )


@app.route("/")
def health():
    return "OK"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)