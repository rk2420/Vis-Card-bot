from telegram.ext import Application, MessageHandler, CommandHandler, filters
from telegram import Update
import pytesseract
from PIL import Image
import cv2
import numpy as np
from openpyxl import load_workbook
from datetime import datetime
import json
import os
import re
import logging
from groq import Groq

# ================= CONFIG =================
BOT_TOKEN = os.getenv("BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"
EXCEL_FILE = "visiting_cards.xlsx"

if not BOT_TOKEN:
    raise ValueError("❌ BOT_TOKEN not set")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not set")

# Windows only – update if needed
pytesseract.pytesseract.tesseract_cmd = "tesseract"
# =========================================

logging.basicConfig(level=logging.INFO)
print("✅ Telegram Visiting Card Bot is STARTING...")

client = Groq(api_key=GROQ_API_KEY)

# ---------- HELPERS ----------
def safe(value):
    return value if value and str(value).strip() else "Not Found"

def clean_text(text):
    replacements = {
        "(at)": "@",
        "[at]": "@",
        "O": "0",
        "o": "0",
        "l": "1",
        "I": "1",
        "|": "1",
        "S": "5",
        "s": "5"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def extract_phone(text):
    match = re.search(r'(\+?\d{1,3}[\s\-]?)?\d{9,10}', text)
    return match.group() if match else "Not Found"

def extract_email(text):
    match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    return match.group() if match else "Not Found"

def extract_website(text):
    match = re.search(r'(www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|https?://[^\s]+)', text)
    return match.group() if match else "Not Found"

def save_to_excel(data, user_id, timestamp):
    wb = load_workbook(EXCEL_FILE)
    ws = wb.active
    ws.append([
        data["name"],
        data["designation"],
        data["company"],
        data["phone"],
        data["email"],
        data["website"],
        data["address"],
        data["industry"],
        ", ".join(data["services"]),
        timestamp,
        user_id
    ])
    wb.save(EXCEL_FILE)

def safe_json_load(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
        return None


def call_groq(prompt, temperature=0):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print("❌ Groq error:", e)
        return None


# ---------- /start ----------
async def start(update: Update, context):
    await update.message.reply_text(
        "✅ Bot is running!\n\n"
        "📸 Send a visiting card image\n"
        "📊 Data saved to Excel\n"
        "💬 Ask follow-up questions"
    )

# ---------- IMAGE HANDLER ----------
async def handle_image(update: Update, context):
    status_msg = await update.message.reply_text(
        "📸 Image received & analyzing…"
    )
    scan_time = datetime.now()
    formatted_time = scan_time.strftime("%d %b %Y | %I:%M %p")

    photo = update.message.photo[-1]
    file = await photo.get_file()
    await file.download_to_drive("card.jpg")

    # ---------- IMAGE PREPROCESSING ----------
    img = cv2.imread("card.jpg")
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    ocr_text = pytesseract.image_to_string(
        thresh,
        config="--oem 3 --psm 6"
    )

    cleaned_text = clean_text(ocr_text)

    # ---------- REGEX EXTRACTION ----------
    phone = extract_phone(cleaned_text)
    email = extract_email(cleaned_text)
    website = extract_website(cleaned_text)

    # ---------- AI PROMPT ----------
    prompt = f"""
You are a data extraction engine.

RULES:
- Output ONLY valid JSON
- No text, no markdown
- Empty string if missing

JSON FORMAT:
{{
  "name": "",
  "designation": "",
  "company": "",
  "address": "",
  "industry": "",
  "services": []
}}

TEXT:
{ocr_text}
"""

    ai_text = call_groq(prompt)

    if not ai_text:
        await update.message.reply_text(
            "⚠️ AI service unavailable. Try again."
        )
        return

    raw = safe_json_load(ai_text)

    if not raw:
        await update.message.reply_text(
            "⚠️ Could not extract structured data.\n"
            "Please try a clearer image."
        )
        return

    data = {
        "name": safe(raw.get("name")),
        "designation": safe(raw.get("designation")),
        "company": safe(raw.get("company")),
        "phone": safe(phone),
        "email": safe(email),
        "website": safe(website),
        "address": safe(raw.get("address")),
        "industry": safe(raw.get("industry")),
        "services": raw.get("services") if raw.get("services") else ["Not Found"]
    }

    context.user_data["company"] = data["company"]
    context.user_data["website"] = data["website"]

    save_to_excel(
        data,
        update.message.from_user.id,
        scan_time.strftime("%Y-%m-%d %H:%M:%S")
    )
    services_text = "\n- ".join(data.get("services", [])) if data.get("services") else "Not Found"

    reply = f"""
📇 Visiting Card Details

Name: {data.get('name', 'Not Found')}
Designation: {data.get('designation', 'Not Found')}
Company: {data.get('company', 'Not Found')}
Phone: {data.get('phone', 'Not Found')}
Email: {data.get('email', 'Not Found')}
Website: {data.get('website', 'Not Found')}
Address: {data.get('address', 'Not Found')}
Industry: {data.get('industry', 'Not Found')}
Services:
- {services_text}

🕒 Scanned On: {formatted_time}
"""

    await update.message.reply_text(reply)

# ---------- FOLLOW-UP ----------
async def handle_text(update: Update, context):
    company = context.user_data.get("company")

    if not company or company == "Not Found":
        await update.message.reply_text("📸 Please upload a visiting card first.")
        return

    prompt = f"""
    Company: {company}
    Website: {context.user_data.get("website")}

    Explain:
    • What the company does
    • Potential customers
    • Potential vendors
    Focus on India.
    """

    response = client.chat.completions.create(
        model= MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    await update.message.reply_text(response.choices[0].message.content)

# ---------- RUN ----------
app = Application.builder().token(BOT_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.PHOTO, handle_image))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

print("🚀 Bot is LIVE and listening...")
app.run_polling()




