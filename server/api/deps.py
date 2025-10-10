from functools import lru_cache
from core.config import settings


@lru_cache()
def get_settings():
    """
    FastAPI dependency for settings.
    Allows override in tests while keeping a single instantiation in production.
    """
    return settings
