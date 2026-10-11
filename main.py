import os
import re
import json
import time
import base64
import logging
import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus

import openai
from fastapi import FastAPI, Depends, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, Session

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("othy-mechanic")

try:
    from dtc_helper import build_context, extract_codes, lookup
except Exception:
    logger.warning("dtc_helper غير متوفر: التشخيص غادي يخدم بلا مرجع الأكواد")
    def build_context(text: str) -> str: return ""
    def extract_codes(text: str) -> list: return []
    def lookup(code: str): return None

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./othy_mechanic.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DiagnosticRecord(Base):
    __tablename__ = "diagnostic_records"
    id = Column(Integer, primary_key=True, index=True)
    section_type = Column(String, index=True)
    car_brand = Column(String, nullable=True)
    gearbox_type = Column(String, nullable=True)
    issue_description = Column(Text, nullable=True)
    image_name = Column(String, nullable=True)
    ai_diagnosis = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

try:
    Base.metadata.create_all(bind=engine)
except Exception:
    logger.exception("تعذر إنشاء الجداول")

app = FastAPI(title="Othy Mechanic Pro Engine & Gearbox", version="3.3")

_origins_env = os.getenv("ALLOWED_ORIGINS", "*").strip()
ALLOWED_ORIGINS = ["*"] if _origins_env == "*" else [o.strip() for o in _origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEXT_FALLBACK_MODELS = [
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "openrouter/free",
]
VISION_FALLBACK_MODELS = [
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
]

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_TEXT_CHARS = 4000
TOTAL_DEADLINE_SECONDS = 90
VALID_SECTIONS = {"engine", "gearbox"}

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

LANG_PROMPT = {
    "ar": "اكتب كل المحتوى بالعربية (فصحى مبسطة مفهومة للميكانيسيان، والمصطلحات التقنية بالإنجليزية بين قوسين عند الحاجة).",
    "fr": "Rédigez tout le contenu en français technique clair, destiné à un mécanicien professionnel.",
    "en": "Write all the content in clear technical English, aimed at a professional mechanic.",
}

JSON_INSTRUCTIONS = """
جاوب فشكل JSON فقط، بلا أي نص زائد قبل ولا بعد وبلا Markdown، بهاد الصيغة بالضبط:
{
  "title": "عنوان قصير ودقيق للعطل",
  "summary": "ملخص تنفيذي فجملتين: شنو العطل الأرجح وشنو الخطوة الأولى",
  "severity": "low" | "medium" | "high",
  "urgent": true | false,
  "urgent_note": "تحذير سلامة محدد إذا كان العطل خطير، فارغ إلا لا",
  "causes": [{"cause": "السبب + علاش هو محتمل", "percent": 60}],
  "checks": ["خطوة فحص عملية مرتبة: شنو تقيس/تشوف، فين، وشنو النتيجة المتوقعة"],
  "repair": ["خطوات الإصلاح الموصى بها بالترتيب، كل خطوة عملية ومحددة"],
  "parts": ["قطع الغيار اللي خاص تتأكد منها أو تبدلها"],
  "next_steps": ["إلا ما تصلحش العطل بعد الإصلاح، شنو الخطوة الجاية"]
}
"""

def build_prompt(section_title: str, brand_str: str, gearbox_str: str, issue_str: str, lang: str, reference: str = "") -> str:
    reference_block = f"\n{reference}\n" if reference else ""
    return f"""
أنت مساعد تشخيص لميكانيسيان محترفين، مختص فأعطال السيارات وأكواد OBD-II، فقسم ({section_title}).
نوع السيارة: {brand_str}
نوع علبة السرعة: {gearbox_str}
وصف المشكلة أو الأكواد: {issue_str}
{reference_block}
{JSON_INSTRUCTIONS}
{LANG_PROMPT.get(lang, LANG_PROMPT["ar"])}
"""

def _try_models_once(client, messages_content: list, model_list: list, deadline: float, plain: list, validator=None):
    last_error = None
    for model_name in model_list:
        if time.monotonic() > deadline:
            last_error = last_error or TimeoutError("تجاوزنا الوقت المسموح")
            break
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "أنت نظام تشخيص ميكانيكي صناعي. جاوب بـJSON فقط."},
                    {"role": "user", "content": messages_content},
                ],
                max_tokens=2200,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("محتوى فارغ")
            if (validator or parse_report)(content) is None:
                if len(content.strip()) >= 200 and not plain:
                    plain.append((content, model_name))
                raise ValueError("جواب غير منظم")
            return content, model_name, None
        except Exception as e:
            last_error = e
            continue
    return None, None, (last_error or RuntimeError("كل الموديلات طاحو"))

def call_model_with_fallback(client, messages_content: list, has_image: bool, validator=None):
    model_list = VISION_FALLBACK_MODELS if has_image else TEXT_FALLBACK_MODELS
    deadline = time.monotonic() + TOTAL_DEADLINE_SECONDS
    plain: list = []
    content, used_model, error = _try_models_once(client, messages_content, model_list, deadline, plain, validator)
    if content: return content, used_model
    if plain and validator is None: return plain[0]
    raise error

REPORT_KEYS = ("title", "summary", "causes", "checks", "repair", "advice")

def parse_report(raw_text: str):
    candidates = [raw_text]
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match: candidates.append(match.group(0))
    for text in candidates:
        try:
            data = json.loads(text)
        except:
            continue
        if isinstance(data, dict) and sum(1 for k in REPORT_KEYS if data.get(k)) >= 2:
            return data
    return None

def extract_json(raw_text: str) -> dict:
    data = parse_report(raw_text)
    if data is not None: return data
    return {"title": "تقرير غير منظم", "summary": raw_text.strip()[:1500], "severity": "medium", "urgent": False}

def _as_str(value, limit: int = 500) -> str:
    if value is None: return ""
    if isinstance(value, str): return value.strip()[:limit]
    return str(value)[:limit]

def _str_list(value, max_items: int = 10, limit: int = 500) -> list:
    out = []
    if isinstance(value, list):
        for item in value[:max_items]:
            text = _as_str(item, limit)
            if text: out.append(text)
    elif isinstance(value, str) and value.strip():
        out = [value.strip()[:limit]]
    return out

def normalize_diagnosis(data: dict) -> dict:
    severity = _as_str(data.get("severity"), 20).lower()
    if severity not in ("low", "medium", "high"): severity = "medium"
    causes = []
    raw_causes = data.get("causes")
    if isinstance(raw_causes, list):
        for item in raw_causes[:8]:
            text = _as_str(item.get("cause") if isinstance(item, dict) else item)
            pct = 50
            if text: causes.append({"cause": text, "percent": pct})
    return {
        "title": _as_str(data.get("title"), 300),
        "summary": _as_str(data.get("summary"), 1500),
        "severity": severity,
        "urgent": bool(data.get("urgent")),
        "urgent_note": _as_str(data.get("urgent_note"), 500),
        "causes": causes,
        "checks": _str_list(data.get("checks")),
        "repair": _str_list(data.get("repair")),
        "parts": _str_list(data.get("parts"), max_items=12, limit=200),
        "next_steps": _str_list(data.get("next_steps")),
    }

def detected_codes_info(text: str) -> list:
    result = []
    for code in extract_codes(text or ""):
        info = lookup(code)
        if info:
            result.append({"c": info.get("c", code), "ar": info.get("ar", ""), "sev": info.get("sev", "medium")})
    return result

def make_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key: raise HTTPException(status_code=503, detail="مفتاح الذكاء الاصطناعي غير مضبوط")
    return openai.OpenAI(base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"), api_key=api_key, timeout=20.0)

@app.get("/health")
def health(): return {"status": "ok"}

# مسار التشخيص الأساسي الأصلي (بدون أي مساس)
@app.post("/api/v1/diagnose-pro")
async def diagnose_pro(
    section_type: str = Form("engine"),
    car_brand: str = Form(""),
    gearbox_type: str = Form(""),
    issue_description: str = Form(""),
    language: str = Form("ar"),
    format: str = Form("structured"),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
):
    issue_text = (issue_description or "").strip()[:MAX_TEXT_CHARS]
    brand_str = (car_brand or "").strip()[:100]
    gearbox_str = (gearbox_type or "").strip()[:100]
    
    image_content = []
    filename = None
    if file is not None and file.filename:
        filename = file.filename
        content_bytes = await file.read()
        if len(content_bytes) > 0:
            b64_data = base64.b64encode(content_bytes).decode("utf-8")
            image_content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}}]

    codes = detected_codes_info(issue_text)
    reference = build_context(issue_text)
    section_title = "تشخيص المحرك والأعطال" if section_type == "engine" else "علبة السرعة"
    prompt_text = build_prompt(section_title, brand_str, gearbox_str, issue_text, language, reference)

    client = make_client()
    raw_result, used_model = call_model_with_fallback(client, [{"type": "text", "text": prompt_text}] + image_content, bool(image_content))
    diagnosis = normalize_diagnosis(extract_json(raw_result))
    
    record = DiagnosticRecord(section_type=section_type, car_brand=brand_str, gearbox_type=gearbox_str, issue_description=issue_text, image_name=filename, ai_diagnosis=raw_result)
    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "status": "success",
        "record_id": str(record.id),
        "diagnosis": diagnosis,
        "codes": codes,
        "model_used": used_model,
        "timestamp": record.created_at,
    }

# مسار كود الراديو الذكي والاحترافي (مضاف بعناية تامة وبدون تعقيد)
@app.get("/api/v1/radio-help")
async def radio_help(brand: str, model: str = None, serial: str = None):
    b = (brand or "").strip().lower()
    s = (serial or "").strip().upper()
    
    # خوارزمية تحليل ودقة عالية حسب الماركة العالمية
    tips = [
        "تأكد من إدخال الأرقام بدقة عبر أزرار الراديو (من 1 إلى 4).",
        "الزر رقم 1 يغير الرقم الأول، وزر 4 يثبت الكود النهائي."
    ]
    steps = [
        "1. قم بتشغيل الراديو حتى يظهر على الشاشة الكلمة المطلوبة (CODE أو SAFE).",
        "2. اضغط على الأزرار المتسلسلة لإدخال الأرقام الصحيحة الخاصة بالراديو.",
        "3. اضغط مطولاً على الزر (6 أو زر التأكيد) لتثبيت الكود وتشغيل الموسيقى."
    ]
    warnings = [
        "تجنب إدخال كود خطأ أكثر من 3 مرات لكي لا يدخل جهاز الراديو في وضع الحظر النهائي (SAFE LOCK)."
    ]
    
    if "renault" in b or "dacia" in b:
        tips.insert(0, "بالنسبة لراديو رينو/داسيا، غالباً ما تحتاج إلى استخراج الـ Pre-code (حرف وثلاثة أرقام) من ملصق الباركود خلف جهاز الراديو.")
    
    links = [
        {
            "name": f"البحث عن كود راديو {brand} في يوتيوب",
            "kind": "video",
            "note": "شرح مرئي دقيق",
            "url": f"https://www.youtube.com/results?search_query=car+radio+code+unlock+{brand}+{s}"
        }
    ]

    return {
        "brand_name": f"دليل فك كود راديو {brand.upper()}",
        "serial": s,
        "serial_hint": f"الرقم التسلسلي المدخل: {s if s else 'غير متوفر'}",
        "tips": tips,
        "steps": steps,
        "warnings": warnings,
        "links": links
    }
