from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, flash, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.utils import secure_filename

# Paths (fully self-contained in the root directory)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.tflite"
UPLOAD_FOLDER = BASE_DIR / "uploads"
HISTORY_FILE = BASE_DIR / "scan_history.json"

# Create uploads folder if it doesn't exist
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Initialize Flask with template and static folders pointing to backend subfolder
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "backend" / "templates"),
    static_folder=str(BASE_DIR / "backend" / "static"),
)
app.secret_key = "dermacareai-exam-friendly-secret"
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

# TFLite Interpreter lazy loader
interpreter = None
input_details = None
output_details = None

def get_interpreter():
    global interpreter, input_details, output_details
    if interpreter is None:
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
    return interpreter, input_details, output_details

labels = ["Acne", "Eczema", "Psoriasis", "Rosacea", "Vitiligo", "Melasma", "Healthy"]

CONCERN_PROFILES: dict[str, dict[str, str]] = {
    "acne": {
        "title": "Acne",
        "diseaseName": "Acne Vulgaris",
        "description": "A skin condition that occurs when pores get clogged with oil and dead cells.",
        "recommendationDescription": "Use oil-control cleansing, anti-inflammatory actives, and non-comedogenic hydration while keeping your routine simple and consistent.",
        "icon": "🔴",
    },
    "open_pores": {
        "title": "Open Pores",
        "diseaseName": "Enlarged Pores with Oily Skin Tendency",
        "description": "Visible enlarged pores usually linked with excess sebum and reduced skin elasticity.",
        "recommendationDescription": "Focus on pore-refining ingredients like niacinamide and salicylic acid and maintain daily sun protection.",
        "icon": "⭕",
    },
    "pigmentation": {
        "title": "Pigmentation",
        "diseaseName": "Hyperpigmentation",
        "description": "Uneven skin tone and dark patches caused by melanin overproduction.",
        "recommendationDescription": "Prioritize brightening ingredients, UV protection, and gentle exfoliation to gradually reduce dark spots.",
        "icon": "🟤",
    },
    "dark_circles": {
        "title": "Dark Circles",
        "diseaseName": "Periorbital Hyperpigmentation",
        "description": "Darkness under the eyes due to pigmentation, fatigue, or vascular visibility.",
        "recommendationDescription": "Use hydrating eye care, improve sleep hygiene, and include caffeine or vitamin C based products.",
        "icon": "🌑",
    },
    "acne_marks": {
        "title": "Acne Marks & Scars",
        "diseaseName": "Post-Acne Marks (PIH)",
        "description": "Residual marks and texture changes after inflammatory acne lesions heal.",
        "recommendationDescription": "Support skin repair with barrier-friendly products, targeted actives, and regular broad-spectrum SPF.",
        "icon": "⚡",
    },
}

# ── Products: 5 UNIQUE products per concern (no cross-concern overlap) ──────
PRODUCTS = [
    # ── ACNE (5 dedicated products) ──────────────────────────────────────────
    {
        "id": "a1",
        "name": "La Roche-Posay Effaclar Duo+",
        "brand": "La Roche-Posay",
        "matchScore": 98,
        "price": "Rs. 1,650",
        "image": "https://images.unsplash.com/photo-1556228578-0d85b1a4d571?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=La+Roche+Posay+Effaclar+Duo",
        "description": "Dual-action acne treatment with LHA + Niacinamide. Unclogs pores and prevents new breakouts.",
        "concerns": ["acne"],
        "keyIngredient": "LHA + Niacinamide",
        "howToUse": "Apply a thin layer on cleansed face morning and/or evening.",
        "step": "Treatment"
    },
    {
        "id": "a2",
        "name": "CeraVe Foaming Facial Cleanser",
        "brand": "CeraVe",
        "matchScore": 97,
        "price": "Rs. 1,335",
        "image": "https://images.unsplash.com/photo-1620916566398-39f1143ab7be?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=CeraVe+Foaming+Cleanser",
        "description": "Non-comedogenic foaming cleanser with ceramides. Removes excess oil without stripping skin barrier.",
        "concerns": ["acne"],
        "keyIngredient": "Ceramides + Niacinamide",
        "howToUse": "Massage onto wet face twice daily, rinse thoroughly.",
        "step": "Cleanser"
    },
    {
        "id": "a3",
        "name": "Minimalist 2% Salicylic Acid Serum",
        "brand": "Minimalist",
        "matchScore": 95,
        "price": "Rs. 449",
        "image": "https://images.unsplash.com/photo-1611930022073-b7a4ba5fcccd?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=Minimalist+Salicylic+Acid+Serum",
        "description": "BHA serum that exfoliates inside pores, reduces whiteheads and blackheads effectively.",
        "concerns": ["acne"],
        "keyIngredient": "2% Salicylic Acid (BHA)",
        "howToUse": "Apply 2–3 drops on cleansed skin at night, 3x per week.",
        "step": "Exfoliant"
    },
    {
        "id": "a4",
        "name": "Neutrogena Oil-Free Acne Moisturizer",
        "brand": "Neutrogena",
        "matchScore": 93,
        "price": "Rs. 799",
        "image": "https://images.unsplash.com/photo-1596462502278-27bfdc403348?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=Neutrogena+Oil+Free+Acne+Moisturizer",
        "description": "Oil-free moisturizer with salicylic acid. Hydrates without clogging pores.",
        "concerns": ["acne"],
        "keyIngredient": "0.5% Salicylic Acid",
        "howToUse": "Apply daily after cleansing as your moisturizer step.",
        "step": "Moisturizer"
    },
    {
        "id": "a5",
        "name": "Acne.org Benzoyl Peroxide 2.5%",
        "brand": "Acne.org",
        "matchScore": 91,
        "price": "Rs. 1,100",
        "image": "https://images.unsplash.com/photo-1608248597279-f99d160bfcbc?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=Benzoyl+Peroxide+acne+treatment",
        "description": "Kills acne-causing bacteria directly. Spot treatment for active pimples and cysts.",
        "concerns": ["acne"],
        "keyIngredient": "2.5% Benzoyl Peroxide",
        "howToUse": "Apply a small amount only on active breakouts at night.",
        "step": "Spot Treatment"
    },

    # ── OPEN PORES (5 dedicated products) ────────────────────────────────────
    {
        "id": "p1",
        "name": "Bioderma Sebium Pore Refiner",
        "brand": "Bioderma",
        "matchScore": 97,
        "price": "Rs. 1,499",
        "image": "https://images.unsplash.com/photo-1571781926291-c477ebfd024b?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=Bioderma+Sebium+Pore+Refiner",
        "description": "Minimises pore appearance, mattifies and blurs enlarged pores with Zinc & Fluidactiv.",
        "concerns": ["open_pores"],
        "keyIngredient": "Zinc PCA + Fluidactiv",
        "howToUse": "Apply morning after serum, before sunscreen.",
        "step": "Treatment"
    },
    {
        "id": "p2",
        "name": "The Ordinary Niacinamide 10% + Zinc 1%",
        "brand": "The Ordinary",
        "matchScore": 96,
        "price": "Rs. 590",
        "image": "https://images.unsplash.com/photo-1600185365483-26d7a4cc7519?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=The+Ordinary+Niacinamide+Zinc",
        "description": "Controls sebum, reduces pore visibility and blemishes. Clinical-grade concentration.",
        "concerns": ["open_pores"],
        "keyIngredient": "10% Niacinamide + 1% Zinc PCA",
        "howToUse": "Apply 2–3 drops twice daily to full face before moisturizer.",
        "step": "Serum"
    },
    {
        "id": "p3",
        "name": "Plum Green Tea Pore Cleansing Face Wash",
        "brand": "Plum",
        "matchScore": 93,
        "price": "Rs. 299",
        "image": "https://images.unsplash.com/photo-1522335789203-aabd1fc54bc9?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=Plum+Green+Tea+Face+Wash",
        "description": "Green tea antioxidants with glycolic acid to deep-cleanse pores and control oil.",
        "concerns": ["open_pores"],
        "keyIngredient": "Green Tea Extract + Glycolic Acid",
        "howToUse": "Use morning and evening as your cleanser.",
        "step": "Cleanser"
    },
    {
        "id": "p4",
        "name": "Innisfree No-Sebum Mineral Powder",
        "brand": "Innisfree",
        "matchScore": 90,
        "price": "Rs. 750",
        "image": "https://images.unsplash.com/photo-1573575155376-b5010099301c?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=Innisfree+No+Sebum+Powder",
        "description": "Volcanic ash powder that absorbs excess sebum and blurs pores for matte finish.",
        "concerns": ["open_pores"],
        "keyIngredient": "Jeju Volcanic Ash",
        "howToUse": "Dust lightly over T-zone after moisturizer to set and mattify.",
        "step": "Finishing"
    },
    {
        "id": "p5",
        "name": "Neutrogena Pore Refining Toner",
        "brand": "Neutrogena",
        "matchScore": 88,
        "price": "Rs. 950",
        "image": "https://images.unsplash.com/photo-1556228720-195a672e8a03?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=Neutrogena+Pore+Refining+Toner",
        "description": "Alpha and beta hydroxy acids exfoliate and tighten pores while restoring skin's pH.",
        "concerns": ["open_pores"],
        "keyIngredient": "AHA + BHA Blend",
        "howToUse": "Apply with cotton pad after cleansing, avoid eye area.",
        "step": "Toner"
    },

    # ── PIGMENTATION (5 dedicated products) ──────────────────────────────────
    {
        "id": "g1",
        "name": "The Derma Co 2% Kojic Acid Serum",
        "brand": "The Derma Co",
        "matchScore": 98,
        "price": "Rs. 549",
        "image": "https://images.unsplash.com/photo-1611930022073-b7a4ba5fcccd?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=Derma+Co+Kojic+Acid+Serum",
        "description": "Inhibits tyrosinase enzyme to fade dark spots and uneven tone at the source.",
        "concerns": ["pigmentation"],
        "keyIngredient": "2% Kojic Acid",
        "howToUse": "Apply 4–5 drops at night on cleansed skin. Use SPF next morning.",
        "step": "Serum (Night)"
    },
    {
        "id": "g2",
        "name": "Minimalist Alpha Arbutin 2% + HA",
        "brand": "Minimalist",
        "matchScore": 96,
        "price": "Rs. 399",
        "image": "https://images.unsplash.com/photo-1620916566398-39f1143ab7be?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=Minimalist+Alpha+Arbutin",
        "description": "Slowly releases hydroquinone to reduce melanin. Safe, gentle and highly effective.",
        "concerns": ["pigmentation"],
        "keyIngredient": "2% Alpha Arbutin + Hyaluronic Acid",
        "howToUse": "Apply a few drops twice daily before moisturizer.",
        "step": "Serum"
    },
    {
        "id": "g3",
        "name": "Neutrogena Rapid Tone Repair Vitamin C Serum",
        "brand": "Neutrogena",
        "matchScore": 94,
        "price": "Rs. 1,299",
        "image": "https://images.unsplash.com/photo-1556228578-0d85b1a4d571?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=Neutrogena+Vitamin+C+Serum",
        "description": "Stabilised Vitamin C brightens, evens tone and shields against free radical damage.",
        "concerns": ["pigmentation"],
        "keyIngredient": "Vitamin C (Ascorbic Acid)",
        "howToUse": "Apply 3–4 drops every morning before SPF.",
        "step": "Serum (AM)"
    },
    {
        "id": "g4",
        "name": "La Roche-Posay Anthelios SPF 50+",
        "brand": "La Roche-Posay",
        "matchScore": 92,
        "price": "Rs. 1,899",
        "image": "https://images.unsplash.com/photo-1608248597279-f99d160bfcbc?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=La+Roche+Posay+Anthelios+SPF+50",
        "description": "Broad-spectrum UVA/UVB protection. Daily SPF is essential to prevent worsening pigmentation.",
        "concerns": ["pigmentation"],
        "keyIngredient": "Mexoryl SX + XL Filters",
        "howToUse": "Apply generously as the last morning step. Reapply every 2 hours outdoors.",
        "step": "Sunscreen"
    },
    {
        "id": "g5",
        "name": "Mamaearth Bye Bye Dark Spots Face Serum",
        "brand": "Mamaearth",
        "matchScore": 89,
        "price": "Rs. 549",
        "image": "https://images.unsplash.com/photo-1596462502278-27bfdc403348?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=Mamaearth+Dark+Spots+Serum",
        "description": "Daisy flower extract and Vitamin C work together to reduce dark spots and boost radiance.",
        "concerns": ["pigmentation"],
        "keyIngredient": "Daisy Extract + Vitamin C",
        "howToUse": "Apply 2–4 drops on face twice daily after cleansing.",
        "step": "Serum"
    },

    # ── DARK CIRCLES (5 dedicated products) ──────────────────────────────────
    {
        "id": "d1",
        "name": "Mamaearth Bye Bye Dark Circles Eye Cream",
        "brand": "Mamaearth",
        "matchScore": 97,
        "price": "Rs. 399",
        "image": "https://images.unsplash.com/photo-1522335789203-aabd1fc54bc9?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=Mamaearth+Bye+Bye+Dark+Circles",
        "description": "Caffeine + peptides + daisy extract reduce puffiness, darkness and fine lines around eyes.",
        "concerns": ["dark_circles"],
        "keyIngredient": "Caffeine + Peptides",
        "howToUse": "Gently pat a small amount around eye area morning and night.",
        "step": "Eye Cream"
    },
    {
        "id": "d2",
        "name": "The Ordinary Caffeine Solution 5% + EGCG",
        "brand": "The Ordinary",
        "matchScore": 95,
        "price": "Rs. 950",
        "image": "https://images.unsplash.com/photo-1573575155376-b5010099301c?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=The+Ordinary+Caffeine+Solution+5%25",
        "description": "High-strength caffeine constricts blood vessels, visibly reducing dark circles and puffiness.",
        "concerns": ["dark_circles"],
        "keyIngredient": "5% Caffeine + EGCG",
        "howToUse": "Apply a few drops under eyes each morning before moisturizer.",
        "step": "Serum"
    },
    {
        "id": "d3",
        "name": "MCaffeine Coffee Under Eye Cream",
        "brand": "MCaffeine",
        "matchScore": 93,
        "price": "Rs. 499",
        "image": "https://images.unsplash.com/photo-1600185365483-26d7a4cc7519?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=MCaffeine+Under+Eye+Cream",
        "description": "Coffee extracts boost circulation; hyaluronic acid plumps and hydrates the delicate eye area.",
        "concerns": ["dark_circles"],
        "keyIngredient": "Coffee Extract + Hyaluronic Acid",
        "howToUse": "Apply with ring finger gently in circular motion twice daily.",
        "step": "Eye Cream"
    },
    {
        "id": "d4",
        "name": "Plum Bright Years Under Eye Recovery Gel",
        "brand": "Plum",
        "matchScore": 90,
        "price": "Rs. 425",
        "image": "https://images.unsplash.com/photo-1556228720-195a672e8a03?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=Plum+Under+Eye+Recovery+Gel",
        "description": "Retinol + antioxidants firm the eye area and reduce appearance of fatigue and darkness.",
        "concerns": ["dark_circles"],
        "keyIngredient": "Retinol + Ceramides",
        "howToUse": "Apply at night around orbital bone; avoid direct contact with eyes.",
        "step": "Treatment (Night)"
    },
    {
        "id": "d5",
        "name": "Himalaya Revitalizing Under Eye Cream",
        "brand": "Himalaya",
        "matchScore": 87,
        "price": "Rs. 249",
        "image": "https://images.unsplash.com/photo-1571781926291-c477ebfd024b?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=Himalaya+Under+Eye+Cream",
        "description": "Aloe vera + rose extracts soothe, hydrate and lighten the under-eye area naturally.",
        "concerns": ["dark_circles"],
        "keyIngredient": "Aloe Vera + Rose Extract",
        "howToUse": "Massage gently under eyes morning and evening. Safe for daily use.",
        "step": "Moisturizer"
    },

    # ── ACNE MARKS / SCARS (5 dedicated products) ─────────────────────────────
    {
        "id": "m1",
        "name": "Mederma PM Intensive Overnight Scar Cream",
        "brand": "Mederma",
        "matchScore": 98,
        "price": "Rs. 1,250",
        "image": "https://images.unsplash.com/photo-1620916566398-39f1143ab7be?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=Mederma+PM+Scar+Cream",
        "description": "Cepalin + Ceramide-NP work overnight to heal and soften raised scars and marks.",
        "concerns": ["acne_marks"],
        "keyIngredient": "Cepalin + Ceramide-NP",
        "howToUse": "Apply to affected area every night. Allow 8 weeks for visible improvement.",
        "step": "Treatment (Night)"
    },
    {
        "id": "m2",
        "name": "Minimalist Tranexamic Acid 3% + Niacinamide",
        "brand": "Minimalist",
        "matchScore": 96,
        "price": "Rs. 549",
        "image": "https://images.unsplash.com/photo-1556228578-0d85b1a4d571?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=Minimalist+Tranexamic+Acid",
        "description": "Tranexamic acid blocks melanin transfer, rapidly fading post-inflammatory hyperpigmentation.",
        "concerns": ["acne_marks"],
        "keyIngredient": "3% Tranexamic Acid + Niacinamide",
        "howToUse": "Apply 3–4 drops twice daily to areas with marks.",
        "step": "Serum"
    },
    {
        "id": "m3",
        "name": "Cetaphil Bright Healthy Radiance Night Cream",
        "brand": "Cetaphil",
        "matchScore": 93,
        "price": "Rs. 1,199",
        "image": "https://images.unsplash.com/photo-1596462502278-27bfdc403348?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=Cetaphil+Bright+Healthy+Radiance",
        "description": "Vitamin B3 + Turmeric brighten and even out skin tone while repairing barrier overnight.",
        "concerns": ["acne_marks"],
        "keyIngredient": "Vitamin B3 + Turmeric Extract",
        "howToUse": "Apply as the last step of your night routine.",
        "step": "Moisturizer (Night)"
    },
    {
        "id": "m4",
        "name": "The Ordinary Alpha Arbutin 2% + HA",
        "brand": "The Ordinary",
        "matchScore": 91,
        "price": "Rs. 640",
        "image": "https://images.unsplash.com/photo-1608248597279-f99d160bfcbc?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=The+Ordinary+Alpha+Arbutin",
        "description": "Fades dark spots and post-acne marks with precision. Suitable for all skin tones.",
        "concerns": ["acne_marks"],
        "keyIngredient": "2% Alpha Arbutin + Hyaluronic Acid",
        "howToUse": "Apply to full face twice daily after cleansing and before moisturizer.",
        "step": "Serum"
    },
    {
        "id": "m5",
        "name": "WOW Skin Science 10% Vitamin C Serum",
        "brand": "WOW Skin Science",
        "matchScore": 88,
        "price": "Rs. 499",
        "image": "https://images.unsplash.com/photo-1522335789203-aabd1fc54bc9?w=400&h=400&fit=crop",
        "amazonUrl": "https://www.amazon.in/s?k=WOW+Vitamin+C+Serum",
        "description": "High-potency Vitamin C brightens scars, boosts collagen and protects skin from UV damage.",
        "concerns": ["acne_marks"],
        "keyIngredient": "10% Vitamin C + Hyaluronic Acid",
        "howToUse": "Apply 3–4 drops every morning before SPF.",
        "step": "Serum (AM)"
    },
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


def is_skin_image(filepath: Path) -> bool:
    """Check if the image contains a reasonable amount of skin color in YCrCb space."""
    try:
        image = cv2.imread(str(filepath))
        if image is None:
            return False
        
        # Downscale for performance since we only need color statistics
        h, w = image.shape[:2]
        max_dim = 400
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

        # Convert to YCrCb color space
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        
        # Standard skin color range for YCrCb: Cr in [130, 180], Cb in [75, 135]
        lower_skin = np.array([0, 130, 75], dtype=np.uint8)
        upper_skin = np.array([255, 180, 135], dtype=np.uint8)
        
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        # Calculate the percentage of skin pixels
        total_pixels = mask.size
        skin_pixels = cv2.countNonZero(mask)
        percentage = (skin_pixels / total_pixels) * 100.0
        
        # Log to stdout for tracking
        print(f"Skin verification percentage: {percentage:.2f}% for {filepath.name}", flush=True)
        
        # If at least 10% of the image is skin, it passes
        return percentage >= 10.0
    except Exception as e:
        print(f"Error during skin check: {e}", flush=True)
        return False


def predict_image(filepath: Path) -> tuple[str, float]:
    """Run TFLite inference and return (predicted_label, confidence_percent)."""
    image = cv2.imread(str(filepath))
    if image is None:
        return "Unable to read image", 0.0
    image = cv2.resize(image, (224, 224))
    image = np.asarray(image)
    image = (image.astype(np.float32) / 127.5) - 1
    image = np.reshape(image, (1, 224, 224, 3))

    interpreter, input_details, output_details = get_interpreter()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]  # shape: (7,)

    index = int(np.argmax(prediction))
    confidence = float(prediction[index]) * 100.0
    return labels[index], round(confidence, 1)


def selected_products(concern: str) -> list[dict[str, Any]]:
    """Return all products for the given concern, sorted by matchScore descending."""
    return sorted([p for p in PRODUCTS if concern in p["concerns"]], key=lambda p: p["matchScore"], reverse=True)


def ai_skin_score(confidence: float, concern: str, skin_type: str, texture: str) -> str:
    """Derive a 0-10 Skin Health Score from the real AI confidence."""
    base = 4.0 + (confidence / 100.0) * 5.5
    texture_adj = {"oily": -0.2, "dry": +0.1, "combination": 0.0}.get(texture, 0)
    type_adj = {"normal": +0.1, "sensitive": -0.1}.get(skin_type, 0)
    value = base + texture_adj + type_adj
    return f"{max(4.0, min(9.5, value)):.1f}"


def build_recommendations(concern: str, skin_type: str, texture: str) -> list[str]:
    """Return a tailored set of 4 recommendations based on concern + skin profile."""
    concern_tips: dict[str, str] = {
        "acne": "Use oil-control cleansing, avoid touching your face, and apply spot treatments with salicylic acid or benzoyl peroxide.",
        "open_pores": "Use niacinamide and BHA to refine pores. Never skip SPF — UV breaks down collagen and widens pores.",
        "pigmentation": "Brighten with Vitamin C in the AM and a tyrosinase inhibitor (kojic acid / arbutin) at night. SPF is non-negotiable.",
        "dark_circles": "Apply a caffeine eye serum in the morning to reduce puffiness. Sleep 7-9 hours and elevate your head slightly.",
        "acne_marks": "Use tranexamic acid or alpha arbutin to fade marks. Avoid picking skin — it deepens PIH significantly.",
    }
    texture_tips: dict[str, str] = {
        "oily": "Choose lightweight gel moisturisers and double-cleanse at night to remove excess sebum buildup.",
        "dry": "Layer a ceramide-rich moisturiser over your serums and avoid foaming cleansers that strip natural oils.",
        "combination": "Apply richer hydration to dry cheeks and lighter, mattifying products only on the oily T-zone.",
    }
    skin_type_tips: dict[str, str] = {
        "sensitive": "Patch-test every new product, choose fragrance-free formulas, and introduce actives one at a time.",
        "normal": "Maintain a consistent AM/PM routine — stability prevents sensitisation and keeps your skin barrier strong.",
    }
    morning_tip = "Every morning: Cleanse → Vitamin C Serum → Moisturiser → SPF 30+. Never skip sunscreen indoors or outdoors."
    return [
        concern_tips.get(concern, ""),
        texture_tips.get(texture, ""),
        skin_type_tips.get(skin_type, ""),
        morning_tip,
    ]


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
        
        # Verify if it's actually a skin image
        if not is_skin_image(filepath):
            try:
                filepath.unlink()  # delete invalid image
            except Exception:
                pass
            flash("Error: Please upload a valid skin image.")
            return redirect(url_for("dashboard"))
            
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
        return render_template("question.html", step=step, title="What is your skin type?", description="This helps us with the right ingredients for your skin.", options=SKIN_TYPES, field="skin_type", next_step="concern", stage=0)
    if step == "concern":
        return render_template("question.html", step=step, title="Which of these describe your concern?", description="Select any one.", options=CONCERNS, field="concern", next_step="result", back_step="skin_type", stage=1)
    return redirect(url_for("result"))


@app.route("/result")
def result():
    concern = session.get("concern")
    skin_type = session.get("skin_type")
    texture = session.get("texture", "combination")
    image_name = session.get("scan_image")
    if not all([concern, skin_type, image_name]):
        return redirect(url_for("dashboard"))

    image_path = UPLOAD_FOLDER / image_name

    # Real AI prediction with confidence
    model_label, confidence = predict_image(image_path)

    profile = CONCERN_PROFILES[concern]

    # AI-confidence-driven skin score
    score = ai_skin_score(confidence, concern, skin_type, texture)

    # Tailored recommendations
    recommendations = build_recommendations(concern, skin_type, texture)

    # All 5 unique products for this concern
    concern_products = selected_products(concern)

    # Persist to history
    rows = read_history()
    record_id = f"{session['user_name']}-{image_name}"
    if not any(row.get("id") == record_id for row in rows):
        rows.append({
            "id": record_id,
            "userName": session["user_name"],
            "score": score,
            "confidence": f"{confidence:.1f}%",
            "disease": profile["diseaseName"],
            "aiDetected": model_label,
            "concern": profile["title"],
            "skinType": skin_type,
            "texture": texture,
            "createdAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        write_history(rows)

    return render_template(
        "result.html",
        profile=profile,
        score=score,
        confidence=confidence,
        model_label=model_label,
        image_name=image_name,
        recommendations=recommendations,
        products=concern_products,
        skin_type=skin_type,
        texture=texture,
    )


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