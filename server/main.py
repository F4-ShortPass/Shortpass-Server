import os
import sys
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from api import interview, evaluation, job, applicant, company, persona, interview_report, jd_persona
from api import interview, jd_persona, job, evaluation_db, jd_parser, evaluation_mock, company, applicant, evaluation_stream, evaluation_result, agent_logs, feedback
import json
import logging # Import logging
from pathlib import Path
from typing import Dict, Any, Optional
import time
from fastapi import Request
from core.config import settings

# 로거 설정
logger = logging.getLogger("uvicorn") # Re-insert logger definition

# 🔴 DEPRECATED: 전역 캐시 (더 이상 사용하지 않음)
# 동적 Persona가 Job 테이블에 저장되므로 하드코딩된 캐시는 불필요
PERSONA_DATA_CACHE: Optional[Dict[str, Any]] = None


def load_persona_data() -> Optional[Dict[str, Any]]:
    """
    ⚠️  DEPRECATED: 더 이상 사용하지 않음

    동적 Persona가 Job 테이블에 저장되므로 하드코딩된 persona_data.json은 불필요합니다.
    하위 호환성을 위해 함수는 남겨두되, None을 반환합니다.
    """
    logger.warning("⚠️  load_persona_data() is deprecated. Use Job.persona fields instead.")
    return None


def load_persona_data_legacy() -> Optional[Dict[str, Any]]:
    """
    서버 시작 시 사전 처리된 JD 페르소나 데이터를 로드

    Returns:
        Dict: 페르소나 데이터 또는 None (파일이 없을 경우)
    """
    persona_file = Path(__file__).parent / "assets" / "persona_data.json"

    try:
        if not persona_file.exists():
            logger.warning(f"⚠️  경고: 페르소나 데이터 파일이 없습니다: {persona_file}")
            logger.warning(f"   전처리 스크립트를 실행하세요: python preprocess_jd.py")
            return None

        with open(persona_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"✅ 페르소나 데이터 로드 완료: {persona_file}")
        logger.info(f"   - 회사: {data.get('company_name', 'Unknown')}")
        logger.info(f"   - 직무: {data.get('job_title', 'Unknown')}")
        logger.info(f"   - 핵심 역량: {len(data.get('core_competencies', []))}개")

        return data

    except json.JSONDecodeError as e:
        logger.error(f"❌ 에러: 페르소나 데이터 JSON 파싱 실패: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ 에러: 페르소나 데이터 로드 실패: {e}")
        return None


app = FastAPI(
    title="AWS_FLEX",
    description="AI 기반 면접 및 채용 매칭 서비스의 API입니다.",
    version="1.0.0",
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Request: {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.4f}s")
    return response


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실행되는 이벤트"""
    global PERSONA_DATA_CACHE

    logger.info("="*60)
    logger.info("AWS_FLEX 서버 시작 중...")
    logger.info("="*60)

    # 🔴 DEPRECATED: 페르소나 데이터 로드 (더 이상 사용하지 않음)
    # 동적 Persona가 Job 테이블에 저장되므로 하드코딩된 캐시 불필요
    # PERSONA_DATA_CACHE = load_persona_data()

    logger.info("✅ 서버 초기화 완료. 동적 Persona 사용 중 (Job 테이블)")
    logger.info("✅ 모든 초기화 완료. 서버 준비 완료!")

# CORS 설정 - 프론트엔드와 통신 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite 개발 서버
        "http://localhost:5174",  # Vite 개발 서버 (추가 포트)
        "http://localhost:3000",  # React 개발 서버
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:3000",
        #"*",  # 개발 중에는 모든 origin 허용 (프로덕션에서는 제거)
    ],
    allow_credentials=True,
    allow_methods=["*"],  # GET, POST, PUT, DELETE 등 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# API 라우터 포함
app.include_router(interview.router, prefix="/api/v1", tags=["Interview"])
app.include_router(jd_persona.router, prefix="/api/v1", tags=["JD Persona"])
app.include_router(job.router, prefix="/api/v1", tags=["Job"])
app.include_router(evaluation_db.router, prefix="/api/v1", tags=["Evaluation"])
app.include_router(jd_parser.router, prefix="/api/v1", tags=["JD Parser"])
app.include_router(applicant.router, prefix="/api/v1", tags=["Applicant"])
app.include_router(company.router, prefix="/api/v1", tags=["Company"])
app.include_router(evaluation_stream.router, prefix="/api/v1", tags=["Evaluation Stream"])
app.include_router(evaluation_result.router, prefix="/api/v1", tags=["Evaluation Result"])
app.include_router(agent_logs.router, prefix="/api/v1", tags=["Agent Logs"])
app.include_router(feedback.router, tags=["Feedback"])  # 🆕 Feedback Loop API

if settings.ENABLE_MOCK_API:
    logger.warning("Mock API routes enabled by configuration (ENABLE_MOCK_API=True)")
    app.include_router(evaluation_mock.router, prefix="/api/v1", tags=["Mock Evaluations"])
else:
    logger.info("Mock API routes disabled (ENABLE_MOCK_API=False)")
# app.include_router(interview_report.router, prefix="/api/v1", tags=["Interview Report"])
# app.include_router(persona.router, prefix="/api/v1/personas", tags=["Persona"])
# app.include_router(evaluation.router, prefix= "/api/v1", tags=["Evaluation"]) 

@app.get("/", tags=["Root"])
async def read_root():
    """
    루트 엔드포인트. API 서버가 실행 중인지 확인합니다.
    """
    return {"message": "Welcome to the Interview & Matching Service API!"}

@app.get("/health", tags=["Health"])
async def health_check():
    """
    헬스 체크 엔드포인트
    """
    return {"status": "healthy", "service": "AWS_FLEX"}
