"""
LINE Chatbot Demo — ไฟล์เดียวจบ
วิธีใช้:
  pip install flask line-bot-sdk
  python chatbot.py
"""

import os
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, FollowEvent

# ══════════════════════════════════════════
# ✏️  ใส่ค่าจาก LINE Developers Console
# ══════════════════════════════════════════
CHANNEL_ACCESS_TOKEN = "q8iFhjH9a/ort7uxxt6OnAkMqoiiwcfnCjuDd3fBBdR12d9qsQRZv+bEZEbJMY7NIJCwYfOS6wrtovLn+Xgl+4kRABVQjOrx/vVSYQAZJ/WB1tvSnAkycx+BPkM6UBhtwHXG0QL5HANUXmYu03LZEgdB04t89/1O/w1cDnyilFU="
CHANNEL_SECRET       = "d2961ac14e172499e09556f9ee0055d8"
# ══════════════════════════════════════════

app      = Flask(__name__)
line_bot = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler  = WebhookHandler(CHANNEL_SECRET)

# ─── แก้คำพิมพ์ผิด / คำไม่เป็นทางการ ─────────────────
INFORMAL = {
    "มั้ย": "ไหม", "คับ": "ครับ", "ครับบ": "ครับ",
    "ละยัง": "หรือยัง", "ค่า": "ค่ะ", "จ้า": "ค่ะ",
    "โอเค": "ตกลง", "เดี๋ยว": "สักครู่",
}

# ─── แปลคำภาษาอังกฤษ (code-mixing) ───────────────────
EN_TH = {
    "check": "ตรวจสอบ", "stock": "สินค้าคงเหลือ",
    "order": "สั่งซื้อ", "tracking": "ติดตามพัสดุ",
    "cancel": "ยกเลิก", "transfer": "โอนเงิน",
}

# ─── กฎ Intent ────────────────────────────────────────
INTENTS = {
    "🛒 สั่งซื้อสินค้า":   ["สั่ง", "ซื้อ", "เอาเลย", "จะเอา", "โอนแล้ว"],
    "🔍 สอบถามสินค้า":     ["มีไหม", "ไซส์", "ราคา", "สี", "รุ่น", "แบบ", "ตรวจสอบ", "สินค้าคงเหลือ"],
    "🚚 ติดตามการจัดส่ง":  ["ของมาหรือยัง", "ถึงไหน", "เลขพัสดุ", "ติดตามพัสดุ", "ยังไม่ได้ของ", "รอนาน"],
    "❌ ยกเลิก/เปลี่ยน":   ["ยกเลิก", "เปลี่ยน", "คืน", "ไม่เอาแล้ว"],
}

# ─── Sentiment ───────────────────────────────────────
POS_WORDS = ["ขอบคุณ", "ดีมาก", "ชอบ", "เยี่ยม", "ประทับใจ"]
NEG_WORDS = ["รอนาน", "แย่", "ไม่ได้", "ผิดหวัง", "ช้ามาก"]


def process(text):
    # 1. normalize
    t = text.strip()
    for w, c in INFORMAL.items():
        t = t.replace(w, c)
    words = t.split()
    t = " ".join(EN_TH.get(w.lower(), w) for w in words)

    # 2. intent
    intent = "❓ ไม่ทราบเจตนา"
    for label, keywords in INTENTS.items():
        if any(kw in t for kw in keywords):
            intent = label
            break

    # 3. sentiment
    if any(w in t for w in NEG_WORDS):
        mood = "😟 เชิงลบ"
    elif any(w in t for w in POS_WORDS):
        mood = "😊 เชิงบวก"
    else:
        mood = None

    # 4. reply
    reply = f"Intent: {intent}"
    if mood:
        reply += f"\nSentiment: {mood}"
    return reply


@app.route("/webhook", methods=["POST"])
def webhook():
    sig  = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    print("hi1")
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
            "สวัสดีครับ! 👋\nลองพิมพ์ดูเลยครับ เช่น\n"
            "• สั่งของได้มั้ยคับ\n"
            "• มีไซส์ M มั้ย\n"
            "• ของมาละยัง\n"
            "• ขอ check stock หน่อย"
        ))
    )


@app.route("/")
def health():
    return "OK"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port,debug=True)
