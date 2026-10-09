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
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, Session

try:
    from sqlalchemy.orm import declarative_base
except ImportError:  # SQLAlchemy < 1.4
    from sqlalchemy.ext.declarative import declarative_base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("othy-mechanic")

# ملفات الداتا (أكواد الأعطال + فك VIN). إلا ما لقاهمش السيرفر ما كيطيحش.
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


# 1. قاعدة البيانات
# SQLite كيخدم فسيرفر طويل الأمد. فserverless الملف كيتمسح فكل deploy: بدلو بـ Postgres عبر DATABASE_URL.
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
except Exception:  # pragma: no cover
    logger.exception("تعذر إنشاء الجداول، التشخيص غادي يخدم بلا حفظ")


# 2. FastAPI و CORS
app = FastAPI(title="Othy Mechanic Pro Engine & Gearbox", version="3.3")

# فالإنتاج دير متغير ALLOWED_ORIGINS بدومين الواجهة، مثلا: https://othy-mechanic.vercel.app
_origins_env = os.getenv("ALLOWED_ORIGINS", "*").strip()
ALLOWED_ORIGINS = ["*"] if _origins_env == "*" else [o.strip() for o in _origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# "openrouter/free" كيختار أحسن موديل مجاني متوفر. الباقي احتياط (الأسماء كتتبدل:
# https://openrouter.ai/models?max_price=0)
TEXT_FALLBACK_MODELS = [
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "openrouter/free",
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
قواعد الجودة:
- causes مرتبة من الأعلى احتمال للأقل (3 إلى 5 أسباب)، ومجموع النسب يقارب 100.
- كل قائمة من 3 إلى 7 عناصر، وكل عنصر جملة وحدة واضحة بلا حشو.
- الفحوصات والإصلاح خاصهم يكونو قابلين للتنفيذ فالورشة (ترتيب منطقي من الأسهل والأرخص للأصعب).
- أعط القيم الرقمية فقط إلا كانت معروفة ومتفق عليها (بحال جهد البطارية)، وإلا قل "قارن مع قيمة الصانع لهاد الموديل".
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
أنت مساعد تشخيص لميكانيسيان محترفين، مختص فأعطال السيارات وأكواد OBD-II، فقسم ({section_title}).
المستخدم ميكانيسيان خبير: ما تشرحش البديهيات، ركز على التشخيص الدقيق والخطوات العملية.

نوع السيارة: {brand_str}
نوع علبة السرعة: {gearbox_str}
وصف المشكلة أو الأكواد: {issue_str}
{reference_block}
قواعد صارمة:
- حلل البيانات والصورة المرفقة (إن وجدت) بدقة.
- إلا كان كود عطل ما كاينش فالمرجع، ما تخمنش معناه: قل أنه خاص بالماركة ويحتاج مرجع الصانع.
- ما تخترعش قيم أو أرقام تقنية ما كنتش متأكد منها.
{JSON_INSTRUCTIONS}
{LANG_PROMPT.get(lang, LANG_PROMPT["ar"])}
"""


def build_followup_prompt(
    section_title: str,
    brand_str: str,
    gearbox_str: str,
    original_issue: str,
    previous_summary: str,
    earlier_findings: list,
    findings: str,
    lang: str,
    reference: str = "",
) -> str:
    earlier_block = ""
    if earlier_findings:
        earlier_block = "نتائج الفحوصات السابقة:\n" + "\n".join(f"- {f}" for f in earlier_findings) + "\n"
    reference_block = f"\n{reference}\n" if reference else ""
    return f"""
أنت مساعد تشخيص لميكانيسيان محترفين، فقسم ({section_title}). هادي متابعة لتشخيص سابق، ماشي تشخيص جديد.

نوع السيارة: {brand_str}
نوع علبة السرعة: {gearbox_str}
الشكاية الأصلية: {original_issue}

التشخيص السابق (آخر تقرير):
{previous_summary}

{earlier_block}النتيجة الجديدة اللي لقاها الميكانيسيان فالفحص:
{findings}
{reference_block}
المطلوب:
- استعمل النتيجة الجديدة باش تأكد أو تستبعد أسباب وتعدل النسب (السبب اللي تستبعد خاصو ينزل أو يتحيد).
- الفحوصات الجديدة خاصها تكون خطوات جديدة، ما تعاودش اللي تدارو.
- إلا النتيجة كتأكد سبب واضح، ركز الإصلاح عليه وعطي خطوات إصلاح محددة.
- ما تخترعش قيم أو أرقام ما كنتش متأكد منها.
{JSON_INSTRUCTIONS}
{LANG_PROMPT.get(lang, LANG_PROMPT["ar"])}
"""


def _try_models_once(client, messages_content: list, model_list: list, deadline: float, plain: list):
    """يجرب لائحة الموديلات مرة وحدة: (نتيجة, موديل, None) ولا (None, None, آخر_خطأ)."""
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
                max_tokens=2200,
            )
            if not response or not response.choices:
                raise ValueError(f"جواب فارغ ولا ناقص من {model_name}")
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError(f"محتوى فارغ من {model_name}")
            if parse_report(content) is None:
                # جواب ماشي تقرير: نحتفظو بيه كحل أخير إلا كان نص طويل، ونجربو موديل آخر
                if len(content.strip()) >= 200 and not plain:
                    plain.append((content, model_name))
                raise ValueError(f"جواب غير منظم من {model_name}: {content.strip()[:80]!r}")
            return content, model_name, None
        except Exception as e:
            logger.warning("Model %s failed: %s", model_name, e)
            last_error = e
            continue
    return None, None, (last_error or RuntimeError("كل الموديلات طاحو"))


def call_model_with_fallback(client, messages_content: list, has_image: bool):
    """يجرب الموديلات، وإلا طاحو كاملين كيعاود مرة وحدة، فحدود وقت إجمالي."""
    model_list = VISION_FALLBACK_MODELS if has_image else TEXT_FALLBACK_MODELS
    deadline = time.monotonic() + TOTAL_DEADLINE_SECONDS
    plain: list = []

    content, used_model, error = _try_models_once(client, messages_content, model_list, deadline, plain)
    if content:
        return content, used_model

    if time.monotonic() < deadline:
        logger.warning("كل الموديلات طاحو فالجولة الأولى، كنعاودو...")
        time.sleep(1)
        content, used_model, error = _try_models_once(client, messages_content, model_list, deadline, plain)
        if content:
            return content, used_model

    if plain:  # حل أخير: نص طويل ماشي JSON
        return plain[0]

    raise error


REPORT_KEYS = ("title", "summary", "causes", "checks", "repair", "advice")


def parse_report(raw_text: str):
    """يرجع dict إلا كان الجواب تقرير JSON حقيقي (فيه على الأقل مفتاحين من التقرير)، وإلا None.
    هادشي كيمنع جوابات بحال "User Safety: safe" (موديلات تصنيف/حماية) من أنها تعدّ تشخيص."""
    candidates = [raw_text]
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for text in candidates:
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict) and sum(1 for k in REPORT_KEYS if data.get(k)) >= 2:
            return data
    return None


def extract_json(raw_text: str) -> dict:
    """يقرا تقرير JSON من جواب الموديل (حتى لو محاط بنص أو fences)، وإلا كيحفظ النص كملخص."""
    data = parse_report(raw_text)
    if data is not None:
        return data
    return {
        "title": "تقرير غير منظم",
        "summary": raw_text.strip()[:1500],
        "severity": "medium",
        "urgent": False,
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


def _str_list(value, max_items: int = 10, limit: int = 500) -> list:
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
    repair = _str_list(data.get("repair")) or _str_list(data.get("advice"))

    return {
        "title": _as_str(data.get("title"), 300),
        "summary": _as_str(data.get("summary"), 1500),
        "severity": severity,
        "urgent": bool(urgent),
        "urgent_note": _as_str(data.get("urgent_note"), 500),
        "causes": causes,
        "checks": _str_list(data.get("checks")),
        "repair": repair,
        "parts": _str_list(data.get("parts"), max_items=12, limit=200),
        "next_steps": _str_list(data.get("next_steps")),
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


SEVERITY_AR = {"low": "منخفضة", "medium": "متوسطة", "high": "عالية"}


def format_report_text(d: dict, codes: list) -> str:
    """تقرير نصي مرتب فخانات: كيتعرض مزيان حتى فالواجهة القديمة (نص عادي)."""
    out = []
    if d.get("title"):
        out.append(f"🔧 {d['title']}")
    out.append(f"⚠️ الخطورة: {SEVERITY_AR.get(d.get('severity'), 'متوسطة')}")
    if d.get("urgent"):
        out.append(f"⛔ تحذير سلامة: {d.get('urgent_note') or 'هذا العطل قد يمس السلامة'}")
    if d.get("summary"):
        out.append(f"\n📌 الملخص\n{d['summary']}")
    if codes:
        out.append("\n🔎 الأكواد المقروءة")
        for c in codes:
            line = f"• {c.get('c', '')} — {c.get('ar', '')}"
            if c.get("en"):
                line += f" ({c['en']})"
            out.append(line)
    if d.get("causes"):
        out.append("\n🎯 الأسباب المحتملة")
        for i, c in enumerate(d["causes"], 1):
            pct = f" ({c['percent']}%)" if c.get("percent") else ""
            out.append(f"{i}. {c['cause']}{pct}")
    for icon, title, key in (
        ("🧪", "الفحوصات المقترحة", "checks"),
        ("🛠️", "خطوات الإصلاح", "repair"),
        ("🔩", "قطع الغيار للتحقق", "parts"),
        ("➡️", "إلا ما تصلحش العطل", "next_steps"),
    ):
        items = d.get(key) or []
        if items:
            out.append(f"\n{icon} {title}")
            for i, item in enumerate(items, 1):
                out.append(f"{i}. {item}")
    return "\n".join(out)


def build_result(structured: bool, record_id, diagnosis: dict, codes: list, used_model, created_at) -> dict:
    """الواجهة الجديدة كتطلب format=structured وكتاخد object. الواجهة القديمة كتاخد نص مرتب."""
    return {
        "status": "success",
        "record_id": record_id,
        "diagnosis": diagnosis if structured else format_report_text(diagnosis, codes),
        "diagnosis_data": diagnosis,
        "codes": codes,
        "model_used": used_model,
        "timestamp": created_at,
    }


def make_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY غير مضبوط")
        raise HTTPException(status_code=503, detail="السيرفر غير مهيأ بعد. تواصل مع الدعم.")
    return openai.OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=api_key,
        max_retries=0,  # كنتحكمو حنا فالمحاولات بين الموديلات
        timeout=20.0,
    )


def run_diagnosis(prompt_text: str, image_content: list):
    """يصيفط الطلب للموديل ويرجع (تشخيص_منظم, نص_خام, اسم_الموديل)."""
    client = make_client()
    messages_content = [{"type": "text", "text": prompt_text}] + image_content
    try:
        raw_result, used_model = call_model_with_fallback(
            client, messages_content, has_image=bool(image_content)
        )
    except Exception:
        logger.exception("كل الموديلات فشلات")
        raise HTTPException(status_code=502, detail="الخدمة مشغولة دابا، عاود جرب من بعد شوية.")
    return normalize_diagnosis(extract_json(raw_result)), raw_result, used_model


def save_record(db, **fields):
    """الحفظ فقاعدة البيانات ماشي ضروري: إلا فشل، التشخيص ديما كيوصل. كيرجع (id, created_at)."""
    try:
        record = DiagnosticRecord(**fields)
        db.add(record)
        db.commit()
        db.refresh(record)
        return record.id, record.created_at
    except Exception:
        logger.exception("تعذر حفظ التشخيص فقاعدة البيانات")
        try:
            db.rollback()
        except Exception:
            pass
        return None, None


def section_title_for(section_type: str) -> str:
    return "تشخيص المحرك والأعطال" if section_type == "engine" else "علبة السرعة (Gearbox)"


@app.get("/health")
def health():
    return {"status": "ok"}


_REFERENCES_FILE = Path(__file__).with_name("references.json")
_DEFAULT_REFERENCES = {
    "sources": [
        {"name": "شيما كهربائية (Google)", "kind": "diagram",
         "url": "https://www.google.com/search?q={q}", "query": "{base} wiring diagram"},
        {"name": "فيديو إصلاح (YouTube)", "kind": "video",
         "url": "https://www.youtube.com/results?search_query={q}", "query": "{base} repair fix"},
    ],
    "curated": [],
}


def load_references() -> dict:
    """يقرا references.json (المصادر + الروابط المؤكدة). إلا كان ناقص ولا خاطئ كيرجع للافتراضي."""
    try:
        with open(_REFERENCES_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("sources"), list):
            data.setdefault("curated", [])
            return data
    except Exception:
        logger.warning("references.json غير متوفر أو خاطئ، كنستعملو الافتراضي")
    return _DEFAULT_REFERENCES


def build_references(brand: str, codes_text: str, topic: str) -> dict:
    refs = load_references()
    brand = (brand or "").strip()[:100]
    topic = (topic or "").strip()[:100]
    code_list = extract_codes(codes_text or "")[:5]

    if code_list:
        base = " ".join([brand] + code_list).strip()
    else:
        base = f"{brand} {topic}".strip()
    if not base:
        return {"links": [], "curated": []}

    links = []
    for source in refs.get("sources", []):
        if not isinstance(source, dict):
            continue
        url_template = str(source.get("url", ""))
        if not url_template.startswith(("http://", "https://")):
            continue
        if source.get("direct"):
            final_url = url_template  # رابط مباشر للموقع (بلا بحث)
        elif "{q}" in url_template:
            query = str(source.get("query", "{base}")).replace("{base}", base)
            final_url = url_template.replace("{q}", quote_plus(query))
        else:
            continue
        links.append(
            {
                "name": str(source.get("name", "مرجع"))[:80],
                "kind": str(source.get("kind", "other"))[:20],
                "note": str(source.get("note", ""))[:120],
                "url": final_url,
            }
        )

    curated = []
    for item in refs.get("curated", []):
        if not isinstance(item, dict) or not item.get("verified") or not item.get("url"):
            continue
        match = item.get("match") if isinstance(item.get("match"), dict) else {}
        codes_ok = not match.get("codes") or any(c in code_list for c in match["codes"])
        brands = match.get("brand_contains")
        brand_ok = not brands or any(str(b).lower() in brand.lower() for b in brands)
        if codes_ok and brand_ok and str(item["url"]).startswith(("http://", "https://")):
            curated.append(
                {
                    "title": str(item.get("title", ""))[:150],
                    "url": str(item["url"]),
                    "type": str(item.get("type", "other"))[:20],
                }
            )
    return {"links": links, "curated": curated}


@app.get("/api/v1/references")
def references(brand: str = "", codes: str = "", topic: str = ""):
    return build_references(brand, codes, topic)


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
    output_format: Optional[str] = Form(None, alias="format"),
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

    prompt_text = build_prompt(
        section_title_for(section_type),
        car_brand or "غير محدد",
        gearbox_type or "غير متاح",
        issue_description or "غير متوفر (راجع الصورة)",
        language,
        build_context(issue_description or ""),
    )

    diagnosis, raw_result, used_model = run_diagnosis(prompt_text, image_content)

    record_id, created_at = save_record(
        db,
        section_type=section_type,
        car_brand=car_brand,
        gearbox_type=gearbox_type,
        issue_description=issue_description,
        image_name=filename,
        ai_diagnosis=raw_result,
    )

    return build_result(
        output_format == "structured",
        record_id,
        diagnosis,
        detected_codes_info(issue_description or ""),
        used_model,
        created_at,
    )


class FollowupRequest(BaseModel):
    section_type: str = "engine"
    car_brand: Optional[str] = None
    gearbox_type: Optional[str] = None
    language: str = "ar"
    original_issue: Optional[str] = None
    previous_report: Optional[dict] = None
    earlier_findings: Optional[list] = None
    format: Optional[str] = None
    findings: str


def _summarize_report(report: Optional[dict]) -> str:
    """يلخص التقرير السابق فنص قصير باش يتعطى للموديل كسياق."""
    if not isinstance(report, dict):
        return "غير متوفر"
    lines = []
    title = _as_str(report.get("title"), 200)
    if title:
        lines.append(f"العنوان: {title}")
    summary = _as_str(report.get("summary"), 600)
    if summary:
        lines.append(f"الملخص: {summary}")
    causes = report.get("causes")
    if isinstance(causes, list):
        for c in causes[:6]:
            if isinstance(c, dict):
                text = _as_str(c.get("cause"), 200)
                if text:
                    lines.append(f"- سبب محتمل ({_as_str(c.get('percent'), 10)}%): {text}")
    checks = _str_list(report.get("checks"), max_items=8, limit=200)
    if checks:
        lines.append("الفحوصات اللي اقترحناها:")
        lines.extend(f"- {c}" for c in checks)
    return "\n".join(lines) or "غير متوفر"


@app.post("/api/v1/diagnose-followup")
def diagnose_followup(payload: FollowupRequest, db: Session = Depends(get_db)):
    section_type = payload.section_type if payload.section_type in VALID_SECTIONS else "engine"
    findings = (payload.findings or "").strip()
    if not findings:
        raise HTTPException(status_code=400, detail="كتب شنو لقيتي فالفحص باش نتابعو.")
    if len(findings) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=400, detail="النص طويل بزاف، اختصرو شوية.")

    car_brand = (payload.car_brand or "").strip()[:200] or None
    gearbox_type = (payload.gearbox_type or "").strip()[:200] or None
    original_issue = (payload.original_issue or "").strip()[:MAX_TEXT_CHARS]
    earlier = [
        _as_str(f, 600)
        for f in (payload.earlier_findings or [])[:10]
        if _as_str(f, 600)
    ]

    prompt_text = build_followup_prompt(
        section_title_for(section_type),
        car_brand or "غير محدد",
        gearbox_type or "غير متاح",
        original_issue or "غير متوفر",
        _summarize_report(payload.previous_report),
        earlier,
        findings,
        payload.language,
        build_context(f"{original_issue}\n{findings}"),
    )

    diagnosis, raw_result, used_model = run_diagnosis(prompt_text, [])

    record_id, created_at = save_record(
        db,
        section_type=section_type,
        car_brand=car_brand,
        gearbox_type=gearbox_type,
        issue_description=f"[FOLLOW-UP] {findings}"[:MAX_TEXT_CHARS],
        image_name=None,
        ai_diagnosis=raw_result,
    )

    return build_result(
        payload.format == "structured",
        record_id,
        diagnosis,
        detected_codes_info(f"{original_issue}\n{findings}"),
        used_model,
        created_at,
    )
