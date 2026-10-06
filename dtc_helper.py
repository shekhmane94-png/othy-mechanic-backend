"""البحث فقاعدة أكواد الأعطال وتجهيز سياق مرجعي للتشخيص."""
import json
import re
from pathlib import Path

_DATA_FILE = Path(__file__).with_name("dtc_codes.json")

with open(_DATA_FILE, encoding="utf-8") as f:
    CODES = {item["c"]: item for item in json.load(f)["codes"]}

_CODE_RE = re.compile(r"\b([PBCU][0-3][0-9A-F]{3})\b", re.IGNORECASE)

_PREFIX_AR = {
    "P": "محرك/علبة سرعة/انبعاثات",
    "B": "هيكل (أيربيج، تكييف، راحة)",
    "C": "شاسيه (ABS، توجيه، تعليق)",
    "U": "شبكة اتصال بين الكمبيوترات",
}


def extract_codes(text: str) -> list[str]:
    """يستخرج أكواد الأعطال من النص بدون تكرار وبنفس الترتيب."""
    seen, result = set(), []
    for match in _CODE_RE.findall(text or ""):
        code = match.upper()
        if code not in seen:
            seen.add(code)
            result.append(code)
    return result


def lookup(code: str) -> dict | None:
    return CODES.get(code.upper())


def build_context(text: str) -> str:
    """يرجع نص مرجعي للأكواد الموجودة فالنص (فارغ إذا ما كاين حتى كود)."""
    lines = []
    for code in extract_codes(text):
        info = lookup(code)
        if info:
            lines.append(
                f"- {code} | {info['en']} | {info['ar']} | الخطورة: {info['sev']} | "
                f"أسباب شائعة: {info['causes']}"
            )
        else:
            kind = "خاص بالماركة" if code[1] in "123" and code[0] == "P" and code[1] != "2" else "غير موجود فالقاعدة"
            lines.append(
                f"- {code} | {kind} ({_PREFIX_AR.get(code[0], '')}). "
                "ما كاينش عندنا وصفه الرسمي: لا تخمن معناه، وقل أنه يحتاج مرجع الماركة."
            )
    if not lines:
        return ""
    return "مرجع أكواد الأعطال (داتا موثوقة، اعتمد عليها ولا تخترع أوصاف):\n" + "\n".join(lines)