import os
from typing import Optional

try:
    from pydantic_settings import BaseSettings
    _settings_backend = "pydantic-settings"
except ImportError:
    # Fallback for environments without pydantic-settings installed
    from dotenv import load_dotenv
    load_dotenv()

    class BaseSettings:  # type: ignore
        """Minimal fallback to avoid runtime failure; does not include validators."""

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    _settings_backend = "fallback"


class Settings(BaseSettings):
    """
    중앙 집중식 환경 설정.

    - .env를 자동으로 읽되 모듈 전역에서 load_dotenv를 중복 호출하지 않도록 함
    - FastAPI Depends나 직접 import로 재사용 가능
    """

    # AWS / Bedrock
    BEDROCK_REGION: str = "us-east-1"
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str = "linkbig-ht-06-f4"  # 실제 배포 시 override 필요
    USE_AWS_S3: bool = False
    ENABLE_MOCK_API: bool = False

    # LLM
    BEDROCK_MODEL_ID: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    OPENAI_API_KEY: Optional[str] = None

    # Database
    DATABASE_URL: str = "postgresql://username:password@localhost:5432/dbname"
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 3600
    DB_ECHO: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @classmethod
    def load(cls):
        """
        Instantiate Settings. If pydantic-settings가 없는 환경에서는 .env를 이미 로드했으므로
        os.environ 기반으로 수동 초기화한다.
        """
        if _settings_backend == "fallback":
            return cls(
                BEDROCK_REGION=os.getenv("BEDROCK_REGION", "us-east-1"),
                AWS_REGION=os.getenv("AWS_REGION", "us-east-1"),
                S3_BUCKET_NAME=os.getenv("S3_BUCKET_NAME", "linkbig-ht-06-f4"),
                USE_AWS_S3=os.getenv("USE_AWS_S3", "false").lower() == "true",
                ENABLE_MOCK_API=os.getenv("ENABLE_MOCK_API", "false").lower() == "true",
                BEDROCK_MODEL_ID=os.getenv(
                    "BEDROCK_MODEL_ID",
                    "anthropic.claude-3-sonnet-20240229-v1:0",
                ),
                OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"),
                DATABASE_URL=os.getenv(
                    "DATABASE_URL",
                    "postgresql://username:password@localhost:5432/dbname",
                ),
                DB_POOL_SIZE=int(os.getenv("DB_POOL_SIZE", "5")),
                DB_MAX_OVERFLOW=int(os.getenv("DB_MAX_OVERFLOW", "10")),
                DB_POOL_TIMEOUT=int(os.getenv("DB_POOL_TIMEOUT", "30")),
                DB_POOL_RECYCLE=int(os.getenv("DB_POOL_RECYCLE", "3600")),
                DB_ECHO=os.getenv("DB_ECHO", "False").lower() == "true",
            )
        return cls()


settings = Settings.load()

# 하위 호환성을 위해 기존 상수명 유지
BEDROCK_REGION = settings.BEDROCK_REGION
AWS_REGION = settings.AWS_REGION
S3_BUCKET_NAME = settings.S3_BUCKET_NAME
USE_AWS_S3 = settings.USE_AWS_S3
ENABLE_MOCK_API = settings.ENABLE_MOCK_API

BEDROCK_MODEL_ID = settings.BEDROCK_MODEL_ID
OPENAI_API_KEY = settings.OPENAI_API_KEY

DATABASE_URL = settings.DATABASE_URL
DB_POOL_SIZE = settings.DB_POOL_SIZE
DB_MAX_OVERFLOW = settings.DB_MAX_OVERFLOW
DB_POOL_TIMEOUT = settings.DB_POOL_TIMEOUT
DB_POOL_RECYCLE = settings.DB_POOL_RECYCLE
DB_ECHO = settings.DB_ECHO
