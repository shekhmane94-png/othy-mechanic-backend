"""فك رقم الشاسي VIN عبر API مجاني ديال NHTSA (vPIC)."""
import json
import re
import urllib.request

_VIN_RE = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")  # بلا I و O و Q

_FIELDS = {
    "Make": "make",
    "Model": "model",
    "ModelYear": "year",
    "BodyClass": "body",
    "EngineModel": "engine_model",
    "DisplacementL": "displacement_l",
    "EngineCylinders": "cylinders",
    "FuelTypePrimary": "fuel",
    "TransmissionStyle": "transmission",
    "DriveType": "drive",
    "PlantCountry": "plant_country",
}


def decode_vin(vin: str) -> dict:
    vin = (vin or "").strip().upper()
    if not _VIN_RE.match(vin):
        raise ValueError("رقم الشاسي خاصو يكون 17 حرف/رقم (بلا I و O و Q)")

    url = f"https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/{vin}?format=json"
    with urllib.request.urlopen(url, timeout=8) as resp:
        raw = json.loads(resp.read().decode("utf-8"))

    row = (raw.get("Results") or [{}])[0]
    result = {"vin": vin}
    for src, dst in _FIELDS.items():
        value = (row.get(src) or "").strip()
        if value:
            result[dst] = value
    result["complete"] = bool(result.get("make") and result.get("model") and result.get("year"))
    return result