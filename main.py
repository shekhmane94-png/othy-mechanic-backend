import os
import base64
import datetime
import openai
from fastapi import FastAPI, Depends, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# 1. إعداد قاعدة البيانات
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
app = FastAPI(title="Othy Mechanic Pro Engine & Gearbox AI", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. الحصول على جلسة قاعدة البيانات
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 5. مسار التشخيص الاحترافي
@app.post("/api/v1/diagnose-pro")
async def diagnose_pro(
    section_type: str = Form(...),
    car_brand: str = Form(None),
    gearbox_type: str = Form(None),
    issue_description: str = Form(None),
    language: str = Form("ar"),
    file: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    client = openai.OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENAI_API_KEY")
    )

    try:
        image_content = []
        filename = None

        if file and file.filename:
            filename = file.filename
            file_bytes = await file.read()
            base64_image = base64.b64encode(file_bytes).decode("utf-8")
            image_content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]

        lang_prompt = {
            "ar": "أجب باللغة العربية بتقرير هندسي ميكانيكي محترف ودقيق.",
            "fr": "Répondez en français avec un rapport de diagnostic mécanique professionnel et précis.",
            "en": "Answer in English with a professional and precise mechanical diagnostic report."
        }.get(language, "أجب باللغة العربية.")

        brand_str = car_brand if car_brand else "غير محدد"
        gearbox_str = gearbox_type if gearbox_type else "غير متاح"
        issue_str = issue_description if issue_description else "غير متوفر"

        section_title = "تشخيص المحرك والأعطال القادمة" if section_type == "engine" else "علبة السرعة (Gearbox)"

        prompt_text = f"""
أنت خبير مهندس ميكانيك سيارات عالمي ومختص في نظام تشخيص وأعطال السيارات المتقدمة OBD-II وقسم المكونات ({section_title}).
نوع السيارة: {brand_str}
نوع علبة السرعة: {gearbox_str}
وصف المشكلة من الميكانيكي/الزبون: {issue_str}

قم بتحليل البيانات والصورة المرفقة (إن وجدت) بدقة تامة واعطني:
1. التحليل التقني الدقيق للمشكلة.
2. الأسباب المحتملة مرتبة حسب الأولوية.
3. الحل العملي المباشر والإصلاح الهندسي المطلوب.
{lang_prompt}
        """

        messages_content = [{"type": "text", "text": prompt_text}] + image_content

        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-lite-001:free",
            messages=[
                {
                    "role": "system",
                    "content": "أنت نظام تشخيص ميكانيكي صناعي عالي الدقة مبني على آلاف البيانات العالمية للسيارات"
                },
                {
                    "role": "user",
                    "content": messages_content
                }
            ],
            max_tokens=1200,
            extra_body={
                "models": [
                    "google/gemini-2.0-flash-lite-001:free",
                    "qwen/qwen-2.5-72b-instruct:free",
                    "meta-llama/llama-3.1-8b-instruct:free"
                ],
                "route": "fallback"
            }
        )

        analysis_result = response.choices[0].message.content

        db_record = DiagnosticRecord(
            section_type=section_type,
            car_brand=car_brand,
            gearbox_type=gearbox_type,
            issue_description=issue_description,
            image_name=filename,
            ai_diagnosis=analysis_result
        )
        db.add(db_record)
        db.commit()
        db.refresh(db_record)

        return {
            "status": "success",
            "record_id": db_record.id,
            "diagnosis": analysis_result,
            "timestamp": db_record.created_at
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في المعالجة الذكية: {str(e)}")
