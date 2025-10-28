"""
평가 API 엔드포인트 (DB 없이 메모리 저장)
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Optional, List
from datetime import datetime
import logging
import json
from services.evaluation.evaluation_service import EvaluationService
from db.database import get_db
from models.interview import InterviewSession
from sqlalchemy.orm import Session

router = APIRouter(prefix="/evaluations")
logger = logging.getLogger("uvicorn")

# 메모리 저장소 (DB 대신)
evaluations_store = {}
evaluation_counter = 0

# 서비스 초기화
evaluation_service = EvaluationService()


# ==================== Schemas ====================

class TranscriptSegment(BaseModel):
    """Transcript Segment"""
    segment_id: int
    segment_order: int
    question_text: str
    answer_text: str
    answer_duration_sec: int
    char_index_start: int
    char_index_end: int


class Transcript(BaseModel):
    """면접 전사"""
    metadata: Dict
    segments: List[TranscriptSegment]


class EvaluationRequest(BaseModel):
    """평가 요청"""
    interview_id: int
    applicant_id: int
    job_id: int
    job_weights: Optional[Dict[str, float]] = None
    common_weights: Optional[Dict[str, float]] = None


class EvaluationStatusResponse(BaseModel):
    """평가 상태 응답"""
    evaluation_id: int
    status: str
    message: Optional[str] = None
    job_score: Optional[float] = None
    common_score: Optional[float] = None
    created_at: str
    completed_at: Optional[str] = None


# ==================== 1. 평가 생성 및 실행 ====================

@router.post("/", response_model=dict)
async def create_evaluation(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    평가 생성 및 백그라운드 실행
    
    Returns:
        {
            "evaluation_id": 1,
            "status": "processing",
            "message": "평가가 시작되었습니다"
        }
    """
    logger.info(f"Creating evaluation for interview ID: {request.interview_id}")
    global evaluation_counter
    
    try:
        # 면접 세션에서 transcript_s3_url 조회
        session = db.query(InterviewSession).filter(InterviewSession.id == request.interview_id).first()
        if not session:
            raise HTTPException(status_code=404, detail=f"Interview session with id {request.interview_id} not found.")

        # [테스트 모드] transcript_jiwon_101.json 직접 사용
        # TODO: 나중에 실제 transcript 사용하도록 변경
        transcript_s3_url = "LOCAL:transcript_jiwon_101"  # 로컬 파일 사용 플래그

        # 평가 ID 생성
        evaluation_counter += 1
        evaluation_id = evaluation_counter
        
        # 메모리에 저장
        evaluations_store[evaluation_id] = {
            "evaluation_id": evaluation_id,
            "interview_id": request.interview_id,
            "applicant_id": request.applicant_id,
            "job_id": request.job_id,
            "status": "processing",
            "job_score": None,
            "common_score": None,
            "job_aggregation_result": None,
            "common_aggregation_result": None,
            "validation_result": None,
            "error_message": None,
            "created_at": datetime.now().isoformat(),
            "completed_at": None
        }
        
        # 기본 가중치
        job_weights = request.job_weights or {
            "structured_thinking": 0.25,
            "business_documentation": 0.20,
            "financial_literacy": 0.20,
            "industry_learning": 0.20,
            "stakeholder_management": 0.15
        }
        
        common_weights = request.common_weights or {
            "problem_solving": 0.25,
            "organizational_fit": 0.20,
            "growth_potential": 0.20,
            "interpersonal_skills": 0.20,
            "achievement_motivation": 0.15
        }
        
        # 백그라운드 작업 등록
        background_tasks.add_task(
            run_evaluation_background,
            evaluation_id=evaluation_id,
            interview_id=request.interview_id,
            applicant_id=request.applicant_id,
            job_id=request.job_id,
            transcript_s3_url=transcript_s3_url,
            job_weights=job_weights,
            common_weights=common_weights
        )
        
        return {
            "evaluation_id": evaluation_id,
            "status": "processing",
            "message": "평가가 시작되었습니다. 약 3-5초 소요됩니다."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"평가 생성 실패: {str(e)}")


# ==================== 2. 평가 상태 조회 ====================

@router.get("/{evaluation_id}", response_model=EvaluationStatusResponse)
async def get_evaluation_status(evaluation_id: int):
    """평가 상태 조회"""
    logger.info(f"Getting evaluation status for ID: {evaluation_id}")
    if evaluation_id not in evaluations_store:
        raise HTTPException(status_code=404, detail="평가를 찾을 수 없습니다")
    
    data = evaluations_store[evaluation_id]
    
    return EvaluationStatusResponse(
        evaluation_id=data["evaluation_id"],
        status=data["status"],
        message=data["error_message"] if data["status"] == "failed" else None,
        job_score=data["job_score"],
        common_score=data["common_score"],
        created_at=data["created_at"],
        completed_at=data["completed_at"]
    )


# ==================== 3. 평가 상세 결과 조회 ====================

@router.get("/{evaluation_id}/detail")
async def get_evaluation_detail(evaluation_id: int):
    """평가 상세 결과 조회"""
    logger.info(f"Getting evaluation detail for ID: {evaluation_id}")
    if evaluation_id not in evaluations_store:
        raise HTTPException(status_code=404, detail="평가를 찾을 수 없습니다")
    
    data = evaluations_store[evaluation_id]
    
    if data["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"평가가 완료되지 않았습니다 (상태: {data['status']})"
        )
    
    return {
        "evaluation_id": data["evaluation_id"],
        "status": data["status"],
        "job_aggregation": data["job_aggregation_result"],
        "common_aggregation": data["common_aggregation_result"],
        "validation": data["validation_result"],
        "analysis_summary": data.get("analysis_summary"),
        "post_processing": data.get("post_processing"),
        "created_at": data["created_at"],
        "completed_at": data["completed_at"]
    }


# ==================== 4. 전체 평가 목록 ====================

@router.get("/")
async def get_all_evaluations():
    """전체 평가 목록 조회 (디버깅용)"""
    logger.info("Getting all evaluations")
    return {
        "total": len(evaluations_store),
        "evaluations": [
            {
                "evaluation_id": data["evaluation_id"],
                "status": data["status"],
                "job_score": data["job_score"],
                "common_score": data["common_score"],
                "created_at": data["created_at"]
            }
            for data in evaluations_store.values()
        ]
    }


# ==================== 백그라운드 작업 ====================

async def run_evaluation_background(
    evaluation_id: int,
    interview_id: int,
    applicant_id: int,
    job_id: int,
    transcript_s3_url: str,
    job_weights: Dict[str, float],
    common_weights: Dict[str, float]
):
    """백그라운드에서 실행되는 평가 함수"""

    try:
        from services.storage.s3_service import S3Service
        from pathlib import Path
        from core.config import settings

        # [테스트 모드] 로컬 파일 사용
        if transcript_s3_url.startswith("LOCAL:"):
            local_file = transcript_s3_url.split("LOCAL:")[1]
            transcript_path = Path(__file__).resolve().parent.parent / "test_data" / f"{local_file}.json"
            print(f"[테스트 모드] 로컬 transcript 사용: {transcript_path}")

            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript = json.load(f)
        else:
            # S3에서 transcript 다운로드
            s3_service = S3Service(bucket_name=settings.S3_BUCKET_NAME, region_name=settings.AWS_REGION)
            s3_key = transcript_s3_url.split(f"s3://{settings.S3_BUCKET_NAME}/")[1]
            transcript = s3_service.download_json(s3_key)

        if not transcript:
            raise Exception(f"Failed to load transcript: {transcript_s3_url}")

        print(f"\n[백그라운드] Evaluation #{evaluation_id} 시작")
        
        # 평가 실행
        result = await evaluation_service.evaluate_interview(
            interview_id=interview_id,
            applicant_id=applicant_id,
            job_id=job_id,
            transcript=transcript,
            job_weights=job_weights,
            common_weights=common_weights
        )
        
        # 메모리 업데이트
        evaluations_store[evaluation_id].update({
            "status": "completed",
            "job_score": result["job_aggregation"]["overall_job_score"],
            "common_score": result["common_aggregation"]["overall_common_score"],
            "job_aggregation_result": result["job_aggregation"],
            "common_aggregation_result": result["common_aggregation"],
            "validation_result": result["validation"],
            "analysis_summary": result.get("analysis_summary"),
            "post_processing": result.get("post_processing"),
            "completed_at": datetime.now().isoformat()
        })
        
        print(f"[백그라운드] Evaluation #{evaluation_id} 완료")
        print(f"  - Job 점수: {evaluations_store[evaluation_id]['job_score']}")
        print(f"  - Common 점수: {evaluations_store[evaluation_id]['common_score']}")
        
    except Exception as e:
        print(f"[백그라운드 오류] Evaluation #{evaluation_id}: {e}")
        
        # 실패 상태 업데이트
        evaluations_store[evaluation_id].update({
            "status": "failed",
            "error_message": str(e)
        })
