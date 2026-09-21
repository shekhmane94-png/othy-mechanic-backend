import os
from fastapi import FastAPI, Depends, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import openai
import datetime
import base64

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./othy_mechanic.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
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

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Othy Mechanic Pro Engine & Gearbox AI", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = openai.OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://omniroute.online/v1"),
    api_key=os.getenv("OPENAI_API_KEY")
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/api/v1/diagnose-pro")
async def diagnose_pro(
    section_type: str = Form(...), # "engine" أو "gearbox"
    car_brand: str = Form(None),
    gearbox_type: str = Form(None),
    issue_description: str = Form(None),
    language: str = Form("ar"), # ar, fr, en
    file: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    """
    محرك التشخيص الاحترافي المتكامل للسيارات (محرك + علبة السرعة) مع دعم Vision AI OCR وقاعدة البيانات.
    """
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
                        "url": f"data:{file.content_type};base64,{base64_image}"
                    }
                }
            ]

        # بناء التوجيه الاحترافي للذكاء الاصطناعي بناءً على القسم والحقول المدخلة
        lang_prompt = {
            "ar": "أجب باللغة العربية بتقرير هندسي ميكانيكي محترف ودقيق.",
            "fr": "Répondez en français avec un rapport de diagnostic mécanique professionnel et précis.",
            "en": "Answer in English with a professional and precise mechanical diagnostic report."
        }.get(language, "أجب باللغة العربية.")

        brand_str = car_brand if car_brand else "غير محدد"
        gearbox_str = gearbox_type if gearbox_type else "غير متاح"
        issue_str = issue_description if issue_description else "غير متوفر"

        prompt_text = f"""
        أنت خبير مهندس ميكانيك سيارات محترف عالمياً ومختص في نظام تشخيص OBD-II وأعطال السيارات المتقدمة.
        القسم المستهدف: {'تشخيص المحرك والأعطال العامة' if section_type == 'engine' else 'تشخيص علبة السرعة (Gearbox)'}
        نوع السيارة: {brand_str}
        نوع علبة السرعة: {gearbox_str}
        وصف المشكل من الميكانيكي/الزبون: {issue_str}
        
        قم بتحليل البيانات والصورة المرفقة (إن وجدت) بدقة تامة، وعطني:
        1. التحليل التقني الدقيق للمشكلة.
        2. الأسباب المحتملة مرتبة حسب الأولوية.
        3. الحل العملي المباشر والإصلاح الهندسي المطلوب.
        {lang_prompt}
        """

        messages_content = [{"type": "text", "text": prompt_text}] + image_content

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "أنت نظام تشخيص ميكانيكي صناعي عالي الدقة مبني على آلاف البيانات العالمية للسيارات."
                },
                {
                    "role": "user",
                    "content": messages_content
                }
            ],
            max_tokens=1200
        )

        analysis_result = response.choices[0].message.content

        # حفظ العملية في قاعدة البيانات الحقيقية
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