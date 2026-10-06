import os
import re
import json
import time
import base64
import logging
import datetime
from typing import Optional

import openai
from fastapi import FastAPI, Depends, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, Session

try:
    from sqlalchemy.orm import declarative_base
except ImportError:  # SQLAlchemy < 1.4
    from sqlalchemy.ext.declarative import declarative_base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("othy-mechanic")

# ملفات الداتا الجديدة (أكواد الأعطال + فك VIN). إلا ما لقاهمش السيرفر ما كيطيحش.
try:
    from dtc_helper import build_context, extract_codes, lookup
except Exception:  # pragma: no cover
    logger.warning("dtc_helper غير متوفر: التشخيص غادي يخدم بلا مرجع الأكواد")

    def build_context(text: str) -> str:
        return ""

    def extract_codes(text: str) -> list:
        return []

    def lookup(code: str):
        return None

try:
    from vin_helper import decode_vin
except Exception:  # pragma: no cover
    logger.warning("vin_helper غير متوفر: endpoint ديال VIN غادي يرجع خطأ")

    def decode_vin(vin: str) -> dict:
        raise RuntimeError("vin_helper missing")


# 1. إعداد قاعدة البيانات
# ملاحظة: SQLite كيخدم فسيرفر طويل الأمد. إلا كنتم خدامين على serverless،
# الملف كيتمسح فكل deploy. فهاد الحالة بدلو بـ Postgres عبر DATABASE_URL.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./othy_mechanic.db")
if DATABASE_URL.startswith("postgres://"):
    # بعض المزودين كيعطيو postgres:// وSQLAlchemy كيبغي postgresql://
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
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


try:
    Base.metadata.create_all(bind=engine)
except Exception:  # pragma: no cover
    logger.exception("تعذر إنشاء الجداول، التشخيص غادي يخدم بلا حفظ")


# 3. إعداد تطبيق FastAPI و CORS
app = FastAPI(title="Othy Mechanic Pro Engine & Gearbox", version="3.2")

# فالإنتاج دير متغير ALLOWED_ORIGINS فـ Environment بدومين الواجهة، مثلا:
# ALLOWED_ORIGINS=https://othy-mechanic.vercel.app
_origins_env = os.getenv("ALLOWED_ORIGINS", "*").strip()
ALLOWED_ORIGINS = ["*"] if _origins_env == "*" else [o.strip() for o in _origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,  # الواجهة ما كتستعملش cookies
    allow_methods=["*"],
    allow_headers=["*"],
)

# "openrouter/free" كيختار أحسن موديل مجاني متوفر فالوقت الحالي.
# باقي الموديلات احتياط. هاد الأسماء كتتبدل من وقت لوقت:
# https://openrouter.ai/models?max_price=0
TEXT_FALLBACK_MODELS = [
    "openrouter/free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
]
VISION_FALLBACK_MODELS = [
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
]

MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 8MB
MAX_TEXT_CHARS = 4000
TOTAL_DEADLINE_SECONDS = 90  # الواجهة كتوقف الطلب بعد 120 ثانية

VALID_SECTIONS = {"engine", "gearbox"}


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


LANG_PROMPT = {
    "ar": "أجب باللغة العربية بتقرير ميكانيكي احترافي ودقيق.",
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
  "checks": ["فحص أو قياس عملي يديرو الميكانيسيان لتأكيد السبب"],
  "advice": ["خطوة 1", "خطوة 2"]
}
رتب causes من الأعلى احتمال للأقل، ومجموع النسب يقارب 100.
checks: خطوات فحص عملية مرتبة (شنو يقيس، فين، وشنو القيمة المتوقعة إلا كنتي متأكد منها).
"""


def build_prompt(
    section_title: str,
    brand_str: str,
    gearbox_str: str,
    issue_str: str,
    lang: str,
    reference: str = "",
) -> str:
    reference_block = f"\n{reference}\n" if reference else ""
    return f"""
أنت مساعد تشخيص لميكانيسيان محترفين، مختص فأعطال السيارات وأكواد OBD-II، وفقسم ({section_title}).

نوع السيارة: {brand_str}
نوع علبة السرعة: {gearbox_str}
وصف المشكلة من الميكانيكي/الزبون: {issue_str}
{reference_block}
قواعد صارمة:
- حلل البيانات والصورة المرفقة (إن وجدت) بدقة.
- إلا كان كود عطل ما كاينش فالمرجع، ماتخمنش معناه: قل بصراحة أنه خاص بالماركة ويحتاج مرجع الصانع.
- ما تخترعش قيم أو أرقام تقنية ما كنتش متأكد منها. القرار النهائي للميكانيسيان.
{JSON_INSTRUCTIONS}
{LANG_PROMPT.get(lang, LANG_PROMPT["ar"])}
"""


def _try_models_once(client, messages_content: list, model_list: list, deadline: float):
    """يجرب لائحة الموديلات مرة وحدة: كيرجع (نتيجة, اسم_الموديل, None) ولا (None, None, آخر_خطأ)."""
    last_error = None
    for model_name in model_list:
        if time.monotonic() > deadline:
            last_error = last_error or TimeoutError("تجاوزنا الوقت المسموح")
            break
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "أنت نظام تشخيص ميكانيكي صناعي عالي الدقة. جاوب بـJSON فقط "
                            "دايما، بلا أي نص قبل أو بعد، وبلا Markdown fences."
                        ),
                    },
                    {"role": "user", "content": messages_content},
                ],
                max_tokens=1500,
                # ما كنستعملوش response_format لأن بعض الموديلات المجانية ما كتدعمهاش مزيان.
            )
            if not response or not response.choices:
                raise ValueError(f"جواب فارغ ولا ناقص من {model_name}")
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError(f"محتوى فارغ من {model_name}")
            return content, model_name, None
        except Exception as e:
            logger.warning("Model %s failed: %s", model_name, e)
            last_error = e
            continue
    return None, None, (last_error or RuntimeError("كل الموديلات طاحو"))


def call_model_with_fallback(client, messages_content: list, has_image: bool):
    """يجرب الموديلات، وإلا طاحو كاملين كيتسنى شوية ويعاود مرة وحدة، فحدود وقت إجمالي."""
    model_list = VISION_FALLBACK_MODELS if has_image else TEXT_FALLBACK_MODELS
    deadline = time.monotonic() + TOTAL_DEADLINE_SECONDS

    content, used_model, error = _try_models_once(client, messages_content, model_list, deadline)
    if content:
        return content, used_model

    if time.monotonic() < deadline:
        logger.warning("كل الموديلات طاحو فالجولة الأولى، كنعاودو...")
        time.sleep(1)
        content, used_model, error = _try_models_once(client, messages_content, model_list, deadline)
        if content:
            return content, used_model

    raise error


def extract_json(raw_text: str) -> dict:
    """يقرا JSON من جواب الموديل، حتى لو كان محاط بنص زائد أو Markdown fences."""
    candidates = [raw_text]
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for text in candidates:
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, TypeError):
            continue
    # فشلنا: نرجعو النص الخام كنصيحة وحدة باش ما يضيعش
    return {
        "title": "تقرير غير منظم",
        "severity": "medium",
        "urgent": False,
        "urgent_note": "",
        "causes": [],
        "checks": [],
        "advice": [raw_text.strip()[:1500]],
    }


def _as_str(value, limit: int = 500) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()[:limit]
    try:
        return json.dumps(value, ensure_ascii=False)[:limit]
    except Exception:
        return str(value)[:limit]


def _str_list(value, max_items: int = 10, limit: int = 400) -> list:
    out = []
    if isinstance(value, list):
        for item in value[:max_items]:
            text = _as_str(item, limit)
            if text:
                out.append(text)
    elif isinstance(value, str) and value.strip():
        out = [value.strip()[:limit]]
    return out


def normalize_diagnosis(data: dict) -> dict:
    """يضمن أن شكل التشخيص دايما سليم، حتى لو الموديل رجع حاجة غريبة."""
    severity = _as_str(data.get("severity"), 20).lower()
    if severity not in ("low", "medium", "high"):
        severity = "medium"

    causes = []
    raw_causes = data.get("causes")
    if isinstance(raw_causes, list):
        for item in raw_causes[:8]:
            if isinstance(item, dict):
                text = _as_str(item.get("cause") or item.get("description"))
                percent_raw = item.get("percent", 0)
            else:
                text = _as_str(item)
                percent_raw = 0
            try:
                percent = int(round(float(str(percent_raw).replace("%", "").strip())))
            except (ValueError, TypeError):
                percent = 0
            percent = max(0, min(100, percent))
            if text:
                causes.append({"cause": text, "percent": percent})

    urgent = data.get("urgent") in (True, "true", "True", "TRUE", 1)

    return {
        "title": _as_str(data.get("title"), 300),
        "severity": severity,
        "urgent": bool(urgent),
        "urgent_note": _as_str(data.get("urgent_note"), 500),
        "causes": causes,
        "checks": _str_list(data.get("checks")),
        "advice": _str_list(data.get("advice")),
    }


def detected_codes_info(text: str) -> list:
    """معلومات الأكواد اللي كاينين فالقاعدة (باش تبانو فالواجهة)."""
    result = []
    for code in extract_codes(text or ""):
        info = lookup(code)
        if info:
            result.append(
                {
                    "c": info.get("c", code),
                    "cat": info.get("cat", ""),
                    "en": info.get("en", ""),
                    "ar": info.get("ar", ""),
                    "sev": info.get("sev", "medium"),
                    "causes": info.get("causes", ""),
                }
            )
    return result


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/v1/vin/{vin}")
def vin_endpoint(vin: str):
    try:
        return decode_vin(vin)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("VIN decode failed")
        raise HTTPException(status_code=502, detail="تعذر فك رقم الشاسي حاليا")


# كنستعملو def (ماشي async def) باش الطلب للموديل ما يبلوكيش السيرفر كامل
@app.post("/api/v1/diagnose-pro")
def diagnose_pro(
    section_type: str = Form(...),
    car_brand: Optional[str] = Form(None),
    gearbox_type: Optional[str] = Form(None),
    issue_description: Optional[str] = Form(None),
    language: str = Form("ar"),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
):
    if section_type not in VALID_SECTIONS:
        raise HTTPException(status_code=400, detail="نوع الفحص غير صحيح.")

    issue_description = (issue_description or "").strip() or None
    car_brand = (car_brand or "").strip()[:200] or None
    gearbox_type = (gearbox_type or "").strip()[:200] or None

    has_file = bool(file and file.filename)
    if not issue_description and not has_file:
        raise HTTPException(status_code=400, detail="خاصك وصف المشكل أو صورة على الأقل.")
    if issue_description and len(issue_description) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=400, detail="الوصف طويل بزاف، اختصرو شوية.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY غير مضبوط")
        raise HTTPException(status_code=503, detail="السيرفر غير مهيأ بعد. تواصل مع الدعم.")

    client = openai.OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=api_key,
        max_retries=0,  # كنتحكمو حنا فالمحاولات بين الموديلات
        timeout=20.0,
    )

    image_content = []
    filename = None
    if has_file:
        file_bytes = file.file.read(MAX_IMAGE_BYTES + 1)
        if len(file_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=400, detail="الصورة كبيرة بزاف (الحد الأقصى 8MB).")
        filename = (file.filename or "")[:200]
        mime = file.content_type if (file.content_type or "").startswith("image/") else "image/jpeg"
        base64_image = base64.b64encode(file_bytes).decode("utf-8")
        image_content = [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64_image}"}}]

    brand_str = car_brand or "غير محدد"
    gearbox_str = gearbox_type or "غير متاح"
    issue_str = issue_description or "غير متوفر (راجع الصورة)"
    section_title = "تشخيص المحرك والأعطال" if section_type == "engine" else "علبة السرعة (Gearbox)"

    reference = build_context(issue_description or "")
    prompt_text = build_prompt(section_title, brand_str, gearbox_str, issue_str, language, reference)
    messages_content = [{"type": "text", "text": prompt_text}] + image_content

    try:
        raw_result, used_model = call_model_with_fallback(
            client, messages_content, has_image=bool(image_content)
        )
    except Exception:
        logger.exception("كل الموديلات فشلات")
        raise HTTPException(status_code=502, detail="الخدمة مشغولة دابا، عاود جرب من بعد شوية.")

    diagnosis = normalize_diagnosis(extract_json(raw_result))

    # الحفظ فقاعدة البيانات ماشي ضروري: إلا فشل، التشخيص ديما كيوصل للمستعمل
    record_id = None
    created_at = None
    try:
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
        record_id = db_record.id
        created_at = db_record.created_at
    except Exception:
        logger.exception("تعذر حفظ التشخيص فقاعدة البيانات")
        try:
            db.rollback()
        except Exception:
            pass

    return {
        "status": "success",
        "record_id": record_id,
        "diagnosis": diagnosis,
        "codes": detected_codes_info(issue_description or ""),
        "model_used": used_model,
        "timestamp": created_at,
    }
