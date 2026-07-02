from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.tflite"
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
from flask import Flask, flash, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.utils import secure_filename
from werkzeug.serving import make_server


@contextmanager
def quiet_stderr():
    original_stderr = sys.stderr
    original_fd = os.dup(2)
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        try:
            sys.stderr = devnull
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            os.dup2(original_fd, 2)
            os.close(original_fd)
            sys.stderr = original_stderr


with quiet_stderr():
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        try:
            import tensorflow.lite as tflite
        except ImportError:
            raise ImportError("Neither tflite_runtime nor tensorflow.lite could be imported.")

interpreter = tflite.Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR
UPLOAD_FOLDER = BASE_DIR / "uploads"
HISTORY_FILE = BASE_DIR / "scan_history.json"
MODEL_PATH = BASE_DIR / "model.tflite"

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "backend" / "templates"),
    static_folder=str(BASE_DIR / "backend" / "static"),
)
app.secret_key = "dermacareai-exam-friendly-secret"
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Model loaded successfully as TFLite interpreter
labels = ["Acne", "Eczema", "Psoriasis", "Rosacea", "Vitiligo", "Melasma", "Healthy"]

CONCERN_PROFILES: dict[str, dict[str, str]] = {
    "acne": {
        "title": "Acne",
        "diseaseName": "Acne Vulgaris",
        "description": "A skin condition that occurs when pores get clogged with oil and dead cells.",
        "recommendationDescription": "Use oil-control cleansing, anti-inflammatory actives, and non-comedogenic hydration while keeping your routine simple and consistent.",
    },
    "open_pores": {
        "title": "Open Pores",
        "diseaseName": "Enlarged Pores with Oily Skin Tendency",
        "description": "Visible enlarged pores usually linked with excess sebum and reduced skin elasticity.",
        "recommendationDescription": "Focus on pore-refining ingredients like niacinamide and salicylic acid and maintain daily sun protection.",
    },
    "pigmentation": {
        "title": "Pigmentation",
        "diseaseName": "Hyperpigmentation",
        "description": "Uneven skin tone and dark patches caused by melanin overproduction.",
        "recommendationDescription": "Prioritize brightening ingredients, UV protection, and gentle exfoliation to gradually reduce dark spots.",
    },
    "dark_circles": {
        "title": "Dark Circles",
        "diseaseName": "Periorbital Hyperpigmentation",
        "description": "Darkness under the eyes due to pigmentation, fatigue, or vascular visibility.",
        "recommendationDescription": "Use hydrating eye care, improve sleep hygiene, and include caffeine or vitamin C based products.",
    },
    "acne_marks": {
        "title": "Acne Marks & Scars",
        "diseaseName": "Post-Acne Marks",
        "description": "Residual marks and texture changes after inflammatory acne lesions heal.",
        "recommendationDescription": "Support skin repair with barrier-friendly products, targeted actives, and regular broad-spectrum SPF.",
    },
}

PRODUCTS = [
    {
        "id": "1",
        "name": "CeraVe Foaming Facial Cleanser",
        "brand": "CeraVe",
        "matchScore": 97,
        "price": "Rs. 1,335",
        "image": "https://images.unsplash.com/photo-1556228578-0d85b1a4d571?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=CeraVe+Foaming+Cleanser",
        "description": "Gentle foaming cleanser for oily and acne-prone skin.",
        "concerns": ["acne", "open_pores"]
    },
    {
        "id": "2",
        "name": "Paula's Choice 2% BHA Liquid Exfoliant",
        "brand": "Paula's Choice",
        "matchScore": 96,
        "price": "Rs. 2,672",
        "image": "https://images.unsplash.com/photo-1620916566398-39f1143ab7be?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=Paula%27s+Choice+2%25+BHA",
        "description": "Salicylic acid exfoliant for clogged pores and bumps.",
        "concerns": ["acne", "open_pores", "acne_marks"]
    },
    {
        "id": "3",
        "name": "Minimalist 10% Niacinamide Serum",
        "brand": "Minimalist",
        "matchScore": 94,
        "price": "Rs. 599",
        "image": "https://images.unsplash.com/photo-1611930022073-b7a4ba5fcccd?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=Minimalist+10%25+Niacinamide",
        "description": "Balances sebum and helps improve visible pores.",
        "concerns": ["acne", "open_pores", "pigmentation"]
    },
    {
        "id": "4",
        "name": "The Derma Co 2% Kojic Acid Serum",
        "brand": "The Derma Co",
        "matchScore": 95,
        "price": "Rs. 549",
        "image": "https://images.unsplash.com/photo-1571781926291-c477ebfd024b?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=Kojic+Acid+Serum",
        "description": "Targets pigmentation and uneven skin tone.",
        "concerns": ["pigmentation", "dark_circles", "acne_marks"]
    },
    {
        "id": "5",
        "name": "La Roche-Posay Anthelios SPF 50",
        "brand": "La Roche-Posay",
        "matchScore": 93,
        "price": "Rs. 1,899",
        "image": "https://images.unsplash.com/photo-1556228720-195a672e8a03?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=La+Roche+Posay+SPF+50",
        "description": "Broad-spectrum sunscreen to prevent worsening pigmentation.",
        "concerns": ["acne", "open_pores", "pigmentation", "dark_circles", "acne_marks"]
    },
    {
        "id": "6",
        "name": "Mamaearth Under Eye Cream",
        "brand": "Mamaearth",
        "matchScore": 91,
        "price": "Rs. 399",
        "image": "https://images.unsplash.com/photo-1522335789203-aabd1fc54bc9?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=under+eye+cream",
        "description": "Eye cream with caffeine and peptides for tired under-eyes.",
        "concerns": ["dark_circles"]
    },
    {
        "id": "7",
        "name": "The Ordinary Caffeine Solution 5%",
        "brand": "The Ordinary",
        "matchScore": 90,
        "price": "Rs. 950",
        "image": "https://images.unsplash.com/photo-1608248597279-f99d160bfcbc?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=The+Ordinary+Caffeine+Solution",
        "description": "Lightweight serum for puffiness and dark under-eye tone.",
        "concerns": ["dark_circles"]
    },
    {
        "id": "8",
        "name": "Bioderma Sebium Pore Refiner",
        "brand": "Bioderma",
        "matchScore": 89,
        "price": "Rs. 1,499",
        "image": "https://images.unsplash.com/photo-1596462502278-27bfdc403348?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=Bioderma+Sebium",
        "description": "Mattifies skin and helps blur enlarged pores.",
        "concerns": ["open_pores", "acne"]
    },
    {
        "id": "9",
        "name": "Mederma PM Intensive Overnight Scar Cream",
        "brand": "Mederma",
        "matchScore": 92,
        "price": "Rs. 1,250",
        "image": "https://images.unsplash.com/photo-1600185365483-26d7a4cc7519?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=scar+cream",
        "description": "Supports overnight recovery of marks and texture.",
        "concerns": ["acne_marks"]
    },
    {
        "id": "10",
        "name": "Cetaphil Bright Healthy Radiance Night Cream",
        "brand": "Cetaphil",
        "matchScore": 88,
        "price": "Rs. 1,199",
        "image": "https://images.unsplash.com/photo-1573575155376-b5010099301c?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=Cetaphil+Bright+Healthy+Radiance",
        "description": "Hydrating brightening cream for stubborn dark spots.",
        "concerns": ["pigmentation", "acne_marks"]
    }
]

SKIN_TYPES = [
    {"key": "normal", "title": "Normal", "image": "normal.webp"},
    {"key": "sensitive", "title": "Sensitive", "image": "sensitive.jpeg"},
]
TEXTURES = [
    {"key": "oily", "title": "Oily", "image": "texture-oily.jpg"},
    {"key": "dry", "title": "Dry", "image": "texture-dry.jpg"},
    {"key": "combination", "title": "Combination", "image": "texture-combination.webp"},
]
CONCERNS = [
    {"key": "acne", "title": "Acne", "subtitle": "A skin condition that occurs when hair follicles become clogged.", "image": "concern-acne.svg"},
    {"key": "open_pores", "title": "Open Pores", "subtitle": "Visible pores usually linked with high sebum production.", "image": "concern-open-pores.svg"},
    {"key": "pigmentation", "title": "Pigmentation", "subtitle": "Uneven patches due to melanin build-up.", "image": "concern-pigmentation.svg"},
    {"key": "dark_circles", "title": "Dark Circles", "subtitle": "Darkness under the eyes due to fatigue or pigmentation.", "image": "concern-dark-circles.svg"},
    {"key": "acne_marks", "title": "Acne Marks & Scars", "subtitle": "Post-inflammatory marks and scars after breakouts.", "image": "concern-acne-marks.svg"},
]


def predict_image(filepath: Path) -> str:
    image = cv2.imread(str(filepath))
    if image is None:
        return "Unable to read image"
    image = cv2.resize(image, (224, 224))
    image = np.asarray(image)
    image = (image.astype(np.float32) / 127.5) - 1
    image = np.reshape(image, (1, 224, 224, 3))
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    index = int(np.argmax(prediction))
    return labels[index]


def selected_products(concern: str) -> list[dict[str, Any]]:
    return sorted([p for p in PRODUCTS if concern in p["concerns"]], key=lambda p: p["matchScore"], reverse=True)


def score_for(concern: str, skin_type: str, texture: str, image_name: str) -> str:
    base = {"acne": 6.3, "open_pores": 6.7, "pigmentation": 6.1, "dark_circles": 6.5, "acne_marks": 6.0}
    texture_shift = {"oily": -0.4, "dry": 0.3, "combination": -0.1}
    type_shift = {"normal": 0.4, "sensitive": -0.2}
    seed_shift = (len(image_name or "capture") % 5) * 0.09
    value = base.get(concern, 6.2) + texture_shift.get(texture, 0) + type_shift.get(skin_type, 0) + seed_shift
    return f"{max(4.8, min(9.4, value)):.1f}"


def read_history() -> list[dict[str, str]]:
    if not HISTORY_FILE.exists():
        return []
    return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))


def write_history(rows: list[dict[str, str]]) -> None:
    HISTORY_FILE.write_text(json.dumps(rows, indent=2), encoding="utf-8")


@app.route("/")
def home():
    return render_template("landing.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        if name:
            session["user_name"] = name
            return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user_name" not in session:
        return redirect(url_for("login"))
    if request.method == "POST":
        image = request.files.get("image")
        if not image or not image.filename:
            flash("Please upload a clear skin image before analysis.")
            return redirect(url_for("dashboard"))
        filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{secure_filename(image.filename)}"
        filepath = UPLOAD_FOLDER / filename
        image.save(filepath)
        session["scan_image"] = filename
        return redirect(url_for("analysis", step="skin_type"))
    return render_template("dashboard.html", user_name=session["user_name"])


@app.route("/analysis", methods=["GET", "POST"])
def analysis():
    if "user_name" not in session:
        return redirect(url_for("login"))
    step = request.args.get("step", "skin_type")
    if request.method == "POST":
        for key in ("skin_type", "texture", "concern"):
            if request.form.get(key):
                session[key] = request.form[key]
        next_step = request.form.get("next_step", "result")
        return redirect(url_for("analysis", step=next_step))

    if step == "skin_type":
        return render_template("question.html", step=step, title="What is your skin type?", description="This helps us with the right ingredients for your skin.", options=SKIN_TYPES, field="skin_type", next_step="skin_texture", stage=0)
    if step == "skin_texture":
        return render_template("question.html", step=step, title="What is your skin texture?", description="This helps us know the moisture content in your skin.", options=TEXTURES, field="texture", next_step="concern", back_step="skin_type", stage=0)
    if step == "concern":
        return render_template("question.html", step=step, title="Which of these describe your concern?", description="Select any one.", options=CONCERNS, field="concern", next_step="result", back_step="skin_texture", stage=1)
    return redirect(url_for("result"))


@app.route("/result")
def result():
    concern = session.get("concern")
    skin_type = session.get("skin_type")
    texture = session.get("texture")
    image_name = session.get("scan_image")
    if not all([concern, skin_type, texture, image_name]):
        return redirect(url_for("dashboard"))

    image_path = UPLOAD_FOLDER / image_name
    model_label = predict_image(image_path)
    profile = CONCERN_PROFILES[concern]
    score = score_for(concern, skin_type, texture, image_name)
    recommendations = [profile["recommendationDescription"]]
    recommendations.append("Use lightweight gel products and wash twice daily." if texture == "oily" else "Add ceramide-rich moisturizers and avoid over-cleansing." if texture == "dry" else "Balance T-zone oil with gentle hydration for cheeks.")
    recommendations.append("Choose fragrance-free formulas and patch test first." if skin_type == "sensitive" else "Use a consistent AM/PM routine and avoid frequent product switching.")

    rows = read_history()
    record_id = f"{session['user_name']}-{image_name}"
    if not any(row.get("id") == record_id for row in rows):
        rows.append({"id": record_id, "userName": session["user_name"], "score": score, "disease": profile["diseaseName"], "createdAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        write_history(rows)

    return render_template("result.html", profile=profile, score=score, model_label=model_label, image_name=image_name, recommendations=recommendations, products=selected_products(concern)[:4])


@app.route("/products")
def products():
    return render_template("products.html", products=PRODUCTS)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        flash("Thank you for your message! We'll get back to you soon.")
        return redirect(url_for("contact"))
    return render_template("contact.html")


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    error = ""
    if request.method == "POST":
        if request.form.get("name", "").strip().lower() == "admin" and request.form.get("password") == "admin123":
            session["admin"] = True
            return redirect(url_for("admin_dashboard"))
        error = "Invalid admin credentials."
    return render_template("admin_login.html", error=error)


@app.route("/admin/logout")
def admin_logout():
    session.pop("admin", None)
    return redirect(url_for("admin_login"))


@app.route("/admin/dashboard")
def admin_dashboard():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))
    return render_template("admin_dashboard.html", rows=read_history())


@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    logging.getLogger("werkzeug").disabled = True
    print("DermaCareAI website: http://127.0.0.1:5000", flush=True)
    app.run(host="0.0.0.0", port=5000, debug=True)