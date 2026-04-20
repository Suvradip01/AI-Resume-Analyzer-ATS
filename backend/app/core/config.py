import os

_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


class Settings:
    PROJECT_NAME: str = "InSightATS API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Fine-tuned weights (see backend/models/)
    NER_MODEL_DIR: str = os.path.join(_BASE, "models", "ner_model")
    MATCHER_MODEL_DIR: str = os.path.join(_BASE, "models", "matcher_model")
    COMPLEXITY_MODEL_DIR: str = os.path.join(_BASE, "models", "complexity_model")

    # SHAP explainability is very slow (~60s+ per request); off by default
    ENABLE_SHAP: bool = _env_bool("ENABLE_SHAP", False)


settings = Settings()
