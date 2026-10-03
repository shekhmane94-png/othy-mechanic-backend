import os
import json
import base64
import logging
import datetime

import openai
from fastapi import FastAPI, Depends, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("othy-mechanic")

# 1. إعداد قاعدة البيانات
# ملاحظة: SQLite كيخدم فسيرفر طويل الأمد. إلا كنتم خدامين على Vercel Functions
# (serverless)، الملف كيتمسح فكل deploy. فهاد الحالة بدلو بـ Postgres مدار
# (Supabase, Neon, Railway...) عبر DATABASE_URL.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./othy_mechanic.db")
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 2. نموذج قاعدة البيانات
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

Base.metadata.create_all(bind=engine)

# 3. إعداد تطبيق FastAPI و CORS
app = FastAPI(title="Othy Mechanic Pro Engine & Gearbox AI", version="3.1")

# TODO: بدل "*" بالدومين الحقيقي ديال الفرونت إند قبل الإطلاق
# مثلا: allow_origins=["https://othy-mechanic.vercel.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# موديلات احتياطية (مجانية عبر OpenRouter، كيدعموا الصور)، كيجرب بالتتالي إلا طاح الأول
# ملاحظة: الموديلات ":free" ديال OpenRouter كيتبدلو من وقت لوقت. إلا طاحو هادو
# بزاف فالمستقبل، دخل لـ https://openrouter.ai/models?max_price=0 وشوف
# الموديلات اللي فيهم "Image" فالـ modalities، وبدل هاد اللائحة.
FALLBACK_MODELS = [
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
]

MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 8MB

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

LANG_PROMPT = {
    "ar": "أجب باللغة العربية بتقرير هندسي ميكانيكي محترف ودقيق.",
    "fr": "Répondez en français avec un rapport de diagnostic mécanique professionnel et précis.",
    "en": "Answer in English with a professional and precise mechanical diagnostic report.",
}

JSON_INSTRUCTIONS = """
جاوب فشكل JSON فقط، بلا أي نص زائد قبل ولا بعد، بهاد الصيغة بالضبط:
{
  "title": "عنوان قصير للتشخيص",
  "severity": "low" | "medium" | "high",
  "urgent": true | false,
  "urgent_note": "تحذير إذا كان المشكل خطير على السلامة، فارغ إلا لا",
  "causes": [{"cause": "وصف السبب", "percent": 70}],
  "advice": ["خطوة 1", "خطوة 2"]
}
رتب causes من الأعلى احتمال للأقل، ومجموع النسب يقارب 100.
"""

def build_prompt(section_title: str, brand_str: str, gearbox_str: str, issue_str: str, lang: str) -> str:
    return f"""
أنت خبير مهندس ميكانيك سيارات عالمي ومختص في نظام تشخيص وأعطال السيارات المتقدمة OBD-II وقسم المكونات ({section_title}).

نوع السيارة: {brand_str}
نوع علبة السرعة: {gearbox_str}
وصف المشكلة من الميكانيكي/الزبون: {issue_str}

قم بتحليل البيانات والصورة المرفقة (إن وجدت) بدقة تامة.
{JSON_INSTRUCTIONS}
{LANG_PROMPT.get(lang, LANG_PROMPT["ar"])}
"""

def call_model_with_fallback(client: "openai.OpenAI", messages_content: list) -> tuple[str, str]:
    """يجرب الموديلات واحد بواحد، كيرجع (نتيجة، اسم_الموديل)."""
    last_error = None
    for model_name in FALLBACK_MODELS:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "أنت نظام تشخيص ميكانيكي صناعي عالي الدقة. جاوب بـJSON فقط دايما.",
                    },
                    {"role": "user", "content": messages_content},
                ],
                max_tokens=1200,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content, model_name
        except Exception as e:
            logger.warning("Model %s failed: %s", model_name, e)
            last_error = e
            continue
    raise last_error if last_error else RuntimeError("كل الموديلات طاحو")

@app.post("/api/v1/diagnose-pro")
async def diagnose_pro(
    section_type: str = Form(...),
    car_brand: str = Form(None),
    gearbox_type: str = Form(None),
    issue_description: str = Form(None),
    language: str = Form("ar"),
    file: UploadFile = File(None),
    db: Session = Depends(get_db),
):
    if not issue_description and not file:
        raise HTTPException(status_code=400, detail="خاصك وصف المشكل أو صورة على الأقل.")

    client = openai.OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    image_content = []
    filename = None
    if file and file.filename:
        file_bytes = await file.read()
        if len(file_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=400, detail="الصورة كبيرة بزاف (الحد الأقصى 8MB).")
        filename = file.filename
        base64_image = base64.b64encode(file_bytes).decode("utf-8")
        image_content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]

    brand_str = car_brand or "غير محدد"
    gearbox_str = gearbox_type or "غير متاح"
    issue_str = issue_description or "غير متوفر"
    section_title = "تشخيص المحرك والأعطال القادمة" if section_type == "engine" else "علبة السرعة (Gearbox)"

    prompt_text = build_prompt(section_title, brand_str, gearbox_str, issue_str, language)
    messages_content = [{"type": "text", "text": prompt_text}] + image_content

    try:
        raw_result, used_model = call_model_with_fallback(client, messages_content)
        try:
            parsed = json.loads(raw_result)
        except json.JSONDecodeError:
            # الموديل ماجاوبش بـJSON صحيح، نرجعو نص خام فحقل title
            parsed = {"title": raw_result, "severity": "medium", "urgent": False,
                      "urgent_note": "", "causes": [], "advice": []}

        db_record = DiagnosticRecord(
            section_type=section_type,
            car_brand=car_brand,
            gearbox_type=gearbox_type,
            issue_description=issue_description,
            image_name=filename,
            ai_diagnosis=raw_result,
        )
        db.add(db_record)
        db.commit()
        db.refresh(db_record)

        return {
            "status": "success",
            "record_id": db_record.id,
            "diagnosis": parsed,
            "model_used": used_model,
            "timestamp": db_record.created_at,
        }
    except Exception as e:
        logger.exception("diagnose_pro failed")
        raise HTTPException(status_code=500, detail="وقع مشكل فالتحليل، عاود جرب من بعد شوية.")
