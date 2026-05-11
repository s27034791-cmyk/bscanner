import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import pytesseract
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()
app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_FILE = BASE_DIR / "knowledge_base.json"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()

if TESSERACT_CMD:
    tesseract_path = Path(TESSERACT_CMD)
    # If user points to install folder, use tesseract.exe inside it.
    if tesseract_path.is_dir():
        tesseract_path = tesseract_path / "tesseract.exe"
    pytesseract.pytesseract.tesseract_cmd = str(tesseract_path)

TARGET_FIELDS = [
    "name",
    "number",
    "address",
    "website",
    "company_name",
    "designation",
]


def score_ocr_text(text: str) -> int:
    cleaned = text.strip()
    if not cleaned:
        return 0
    alnum = sum(ch.isalnum() for ch in cleaned)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    bonus = 15 if len(lines) >= 3 else 0
    digits_bonus = min(20, sum(ch.isdigit() for ch in cleaned))
    return alnum + bonus + digits_bonus


def token_overlap_ratio(text_a: str, text_b: str) -> float:
    tokens_a = {tok.lower() for tok in text_a.split() if len(tok) > 2}
    tokens_b = {tok.lower() for tok in text_b.split() if len(tok) > 2}
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / float(len(tokens_a | tokens_b))


def decode_image(image_bytes: bytes) -> np.ndarray:
    np_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image data.")

    max_width = 1000
    height, width = image.shape[:2]
    if width > max_width:
        scale = max_width / float(width)
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return image


def simple_preprocess_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def apply_custom_lut(image: np.ndarray) -> np.ndarray:
    # Slight gamma lift to brighten low-contrast card text.
    gamma = 0.85
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, lut)


def enhance_with_dft(image: np.ndarray) -> np.ndarray:
    float_img = np.float32(image)
    dft = cv2.dft(float_img, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft, axes=(0, 1))

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    radius = max(10, min(rows, cols) // 18)

    # High-pass mask: suppress low-frequency illumination/background.
    mask = np.ones((rows, cols, 2), np.float32)
    cv2.circle(mask, (ccol, crow), radius, (0, 0), -1)

    filtered_shift = dft_shift * mask
    filtered = np.fft.ifftshift(filtered_shift, axes=(0, 1))
    restored = cv2.idft(filtered)
    magnitude = cv2.magnitude(restored[:, :, 0], restored[:, :, 1])

    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Stronger denoising for low-quality captures.
    denoised = cv2.fastNlMeansDenoising(gray, None, h=14, templateWindowSize=7, searchWindowSize=21)
    blurred = cv2.GaussianBlur(denoised, (3, 3), 0)

    # 1) Local contrast enhancement.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(blurred)

    # 2) Non-linear brightness curve.
    lut_enhanced = apply_custom_lut(contrast)

    # 3) Spatial sharpening.
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5.5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(lut_enhanced, -1, sharpen_kernel)

    # 4) Frequency-domain enhancement.
    freq_enhanced = enhance_with_dft(sharpened)

    # 5) Blend and binarize for better OCR stability.
    blended = cv2.addWeighted(sharpened, 0.65, freq_enhanced, 0.35, 0)
    return cv2.normalize(blended, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def extract_text_from_image(image: np.ndarray, config: str = "--oem 3 --psm 6") -> str:
    try:
        text = pytesseract.image_to_string(image, config=config)
    except OSError as exc:
        if "WinError 5" in str(exc):
            raise RuntimeError(
                "Tesseract path is not executable. Set TESSERACT_CMD to the full exe path, "
                "for example: C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
            ) from exc
        raise
    return text.strip()


def choose_raw_or_enhanced_text(image: np.ndarray) -> str:
    simple_image = simple_preprocess_image(image)
    simple_text_1 = extract_text_from_image(simple_image, config="--oem 3 --psm 6")
    simple_text_2 = extract_text_from_image(simple_image, config="--oem 3 --psm 11")

    score_1 = score_ocr_text(simple_text_1)
    score_2 = score_ocr_text(simple_text_2)
    simple_best = simple_text_1 if score_1 >= score_2 else simple_text_2
    simple_best_score = max(score_1, score_2)
    overlap = token_overlap_ratio(simple_text_1, simple_text_2)

    # If plain OCR is decent and consistent, skip heavy OpenCV enhancements.
    if simple_best_score >= 45 and (overlap >= 0.4 or abs(score_1 - score_2) <= 10):
        return simple_best

    enhanced_image = preprocess_image(image)
    enhanced_text = extract_text_from_image(enhanced_image)

    # Mirrored fallback for front-camera captures.
    flipped_text = extract_text_from_image(cv2.flip(enhanced_image, 1))
    enhanced_best = enhanced_text if score_ocr_text(enhanced_text) >= score_ocr_text(flipped_text) else flipped_text

    # Missing fields are valid; choose whichever OCR text is stronger.
    if score_ocr_text(enhanced_best) > simple_best_score:
        return enhanced_best
    return simple_best


def normalize_response(data: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {}
    for field in TARGET_FIELDS:
        value = data.get(field)
        if value in ("", "N/A", "n/a", "null", "None"):
            value = None
        normalized[field] = value
    return normalized


def get_structured_data_with_groq(raw_text: str) -> Dict[str, Any]:
    if not GROQ_API_KEY:
        raise RuntimeError("Missing GROQ_API_KEY in .env file.")

    prompt = (
        "You are an information extraction assistant for business cards.\n"
        "Extract data and return ONLY valid JSON with keys:\n"
        "name, number, address, website, company_name, designation.\n"
        "Rules:\n"
        "1) Use null for missing values.\n"
        "2) If multiple phone numbers exist, combine into one string separated by comma.\n"
        "3) Do not add extra keys.\n"
        "4) Return JSON only, no markdown.\n\n"
        f"Business card OCR text:\n{raw_text}"
    )

    payload = {
        "model": GROQ_MODEL,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": "You extract business card details to strict JSON."},
            {"role": "user", "content": prompt},
        ],
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=45)
    response.raise_for_status()
    model_text = response.json()["choices"][0]["message"]["content"].strip()

    if model_text.startswith("```"):
        model_text = model_text.replace("```json", "").replace("```", "").strip()

    parsed = json.loads(model_text)
    return normalize_response(parsed)


def append_to_knowledge_base(entry: Dict[str, Any]) -> None:
    if KNOWLEDGE_BASE_FILE.exists():
        with KNOWLEDGE_BASE_FILE.open("r", encoding="utf-8") as f:
            try:
                records = json.load(f)
            except json.JSONDecodeError:
                records = []
    else:
        records = []

    records.append(
        {
            "confirmed_at": datetime.utcnow().isoformat() + "Z",
            "data": normalize_response(entry),
        }
    )

    with KNOWLEDGE_BASE_FILE.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/scan")
def scan_card():
    image_file = request.files.get("image")
    if not image_file:
        return jsonify({"error": "No image uploaded."}), 400

    try:
        image = decode_image(image_file.read())
        raw_text = choose_raw_or_enhanced_text(image)
        structured = get_structured_data_with_groq(raw_text)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({"raw_text": raw_text, "data": structured})


@app.post("/confirm")
def confirm_data():
    payload = request.get_json(silent=True) or {}
    data = payload.get("data", {})
    confirmed = normalize_response(data)
    append_to_knowledge_base(confirmed)
    return jsonify({"message": "Saved to knowledge base.", "data": confirmed})


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)