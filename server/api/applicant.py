# server/api/applicant.py
"""
Applicant 관련 API 엔드포인트
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import Optional, List
import logging

from db.database import get_db
from services.applicant_service import ApplicantService
from schemas.applicant import (
    ApplicantCreate,
    ApplicantResponse,
    ApplicantDetailResponse,
    ApplicantUpdate
)
from schemas.evaluation import EvaluationDetailResponse
from models.evaluation import Evaluation
from models.interview import Applicant
from models.job import Job


router = APIRouter(prefix="/applicants", tags=["applicants"])
logger = logging.getLogger("uvicorn")


def build_evaluation_detail_from_db(eval_obj, applicant, job):
    """DB 평가 결과를 프론트엔드 형식으로 변환"""
    # 레이더 차트 데이터 구성
    radar_labels = []
    radar_scores = []
    competency_details = []

    # job_aggregation에서 역량 점수 추출
    if eval_obj.job_aggregation:
        for name, data in eval_obj.job_aggregation.items():
            score = data.get("score", 0) if isinstance(data, dict) else data
            radar_labels.append(name)
            radar_scores.append(score)

            # 상세 정보 구성
            detail = {
                "name": name,
                "score": score,
                "positive_feedback": data.get("positive_feedback", "") if isinstance(data, dict) else None,
                "negative_feedback": data.get("negative_feedback") if isinstance(data, dict) else None,
                "evidence_transcript_id": data.get("evidence_id") if isinstance(data, dict) else None
            }
            competency_details.append(detail)

    # AI 요약 추출
    ai_summary = ""
    if eval_obj.fit_analysis and isinstance(eval_obj.fit_analysis, dict):
        ai_summary = eval_obj.fit_analysis.get("summary", "") or eval_obj.fit_analysis.get("overall_assessment", "")

    # 등급 산정
    score = eval_obj.match_score or 0
    if score >= 90:
        grade = "S"
    elif score >= 80:
        grade = "A"
    elif score >= 70:
        grade = "B"
    elif score >= 60:
        grade = "C"
    else:
        grade = "D"

    # Deep dive 분석 추출
    deep_dive_analysis = []
    if eval_obj.reasoning_log and isinstance(eval_obj.reasoning_log, list):
        for item in eval_obj.reasoning_log:
            if isinstance(item, dict):
                deep_dive_analysis.append({
                    "question_topic": item.get("topic", ""),
                    "trigger_reason": item.get("trigger_reason", ""),
                    "initial_question": item.get("initial_question", ""),
                    "candidate_initial_response": item.get("initial_response", ""),
                    "follow_up_question": item.get("follow_up_question", ""),
                    "candidate_response_summary": item.get("response_summary", ""),
                    "agent_evaluation": item.get("evaluation", ""),
                    "score_impact": item.get("score_impact", "0"),
                    "transcript_segment_id": item.get("segment_id", "")
                })

    return {
        "candidate_id": f"CAND_{applicant.id:03d}" if applicant else f"CAND_{eval_obj.applicant_id:03d}",
        "name": applicant.name if applicant else "Unknown",
        "job_title": job.title if job else "미정",
        "total_score": round(score),
        "grade": grade,
        "ai_summary": ai_summary,
        "radar_chart_data": {
            "labels": radar_labels if radar_labels else ["종합"],
            "scores": radar_scores if radar_scores else [round(score)]
        },
        "competency_details": competency_details if competency_details else [{
            "name": "종합 평가",
            "score": round(score),
            "positive_feedback": None,
            "negative_feedback": None,
            "evidence_transcript_id": None
        }],
        "deep_dive_analysis": deep_dive_analysis,
        "feedback_loop": {
            "is_reviewed": False,
            "hr_comment": "",
            "adjusted_score": None
        },
        "interview_date": eval_obj.created_at.strftime("%Y-%m-%d") if eval_obj.created_at else "",
        "priority_review": score >= 85,
        "rank": 0
    }


@router.get("/{applicant_id}/evaluation-details", response_model=EvaluationDetailResponse)
async def get_applicant_evaluation_details(
    applicant_id: int,
    db: Session = Depends(get_db)
):
    """
    지원자의 상세 평가 리포트 조회

    Args:
        applicant_id: 지원자 ID

    Returns:
        EvaluationDetailResponse: 지원자의 상세 평가 정보

    Raises:
        404: 평가 결과를 찾을 수 없는 경우
    """
    logger.info(f"Getting evaluation details for applicant ID: {applicant_id}")

    # 1. Applicant 확인
    applicant = db.query(Applicant).filter(Applicant.id == applicant_id).first()
    if not applicant:
        raise HTTPException(status_code=404, detail=f"Applicant {applicant_id} not found")

    # 2. DB에서 최신 평가 결과 조회
    eval_obj = db.query(Evaluation).filter(
        Evaluation.applicant_id == applicant_id
    ).order_by(Evaluation.created_at.desc()).first()

    if not eval_obj:
        raise HTTPException(
            status_code=404,
            detail=f"No evaluation found for applicant {applicant_id}. Please complete the interview and evaluation first."
        )

    logger.info(f"Found evaluation (ID: {eval_obj.id}) for applicant_id={applicant_id}")

    # 3. Job 조회
    job = db.query(Job).filter(Job.id == eval_obj.job_id).first() if eval_obj.job_id else None

    # 4. DB 결과를 프론트엔드 형식으로 변환
    return build_evaluation_detail_from_db(eval_obj, applicant, job)



@router.post("/", response_model=ApplicantResponse)
async def create_applicant(
    name: str = Form(..., description="이름"),
    email: str = Form(..., description="이메일"),
    gender: Optional[str] = Form(None, description="성별"),
    education: Optional[str] = Form(None, description="학력"),
    birthdate: Optional[str] = Form(None, description="생년월일 (YYYY-MM-DD)"),
    portfolio_file: Optional[UploadFile] = File(None, description="포트폴리오 PDF"),
    db: Session = Depends(get_db)
):
    """
    지원자 생성 또는 업데이트 (Upsert)

    동일한 이메일이 이미 존재하면 기존 지원자 정보를 업데이트합니다.

    전체 플로우:
    1. 이메일로 기존 지원자 확인
    2. 존재하면 업데이트, 없으면 새로 생성
    3. 포트폴리오 PDF가 있으면 S3에 업로드
    4. DB에 파일 경로 저장

    Args:
        name: 이름
        email: 이메일
        gender: 성별
        education: 학력
        birthdate: 생년월일
        portfolio_file: 포트폴리오 PDF
        db: Database session

    Returns:
        ApplicantResponse: 생성 또는 업데이트된 지원자 정보
    """
    logger.info(f"Creating or updating applicant with email: {email}")
    try:
        applicant_service = ApplicantService()

        # 지원자 생성
        applicant = applicant_service.create_applicant(
            db=db,
            name=name,
            email=email,
            gender=gender,
            education=education,
            birthdate=birthdate,
        )

        # 포트폴리오 업로드 (옵션) - RAG 파이프라인 포함
        if portfolio_file and portfolio_file.filename:
            # PDF 파일 검증
            if not portfolio_file.filename.endswith('.pdf'):
                raise HTTPException(
                    status_code=400,
                    detail="Only PDF files are allowed for portfolio"
                )

            # 파일 크기 제한 (10MB)
            max_size = 10 * 1024 * 1024
            file_content = await portfolio_file.read()

            if len(file_content) > max_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"File size exceeds maximum limit of {max_size / (1024*1024)}MB"
                )

            # S3 업로드 + RAG 파이프라인 (파싱, 청킹, 임베딩 생성)
            await applicant_service.upload_portfolio(
                db=db,
                applicant_id=applicant.id,
                file_content=file_content,
                file_name=portfolio_file.filename
            )

        return ApplicantResponse.model_validate(applicant)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create applicant: {str(e)}"
        )


@router.get("/{applicant_id}", response_model=ApplicantDetailResponse)
async def get_applicant(
    applicant_id: int,
    db: Session = Depends(get_db)
):
    """
    지원자 상세 정보 조회

    Args:
        applicant_id: 지원자 ID
        db: Database session

    Returns:
        ApplicantDetailResponse: 지원자 상세 정보
    """
    logger.info(f"Getting applicant with ID: {applicant_id}")
    applicant_service = ApplicantService()
    applicant = applicant_service.get_applicant(db, applicant_id)

    if not applicant:
        raise HTTPException(status_code=404, detail="Applicant not found")

    return ApplicantDetailResponse.model_validate(applicant)


@router.get("/", response_model=List[ApplicantResponse])
async def get_applicants(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    지원자 목록 조회

    Args:
        skip: 건너뛸 개수
        limit: 조회할 개수
        db: Database session

    Returns:
        List[ApplicantResponse]: 지원자 목록
    """
    logger.info(f"Getting applicants with skip: {skip}, limit: {limit}")
    applicant_service = ApplicantService()
    applicants = applicant_service.get_applicants(db, skip=skip, limit=limit)

    return [ApplicantResponse.model_validate(a) for a in applicants]


@router.patch("/{applicant_id}", response_model=ApplicantResponse)
async def update_applicant(
    applicant_id: int,
    applicant_update: ApplicantUpdate,
    db: Session = Depends(get_db)
):
    """
    지원자 정보 수정

    Args:
        applicant_id: 지원자 ID
        applicant_update: 수정할 정보
        db: Database session

    Returns:
        ApplicantResponse: 수정된 지원자 정보
    """
    logger.info(f"Updating applicant with ID: {applicant_id}")
    applicant_service = ApplicantService()

    update_data = applicant_update.model_dump(exclude_unset=True)
    applicant = applicant_service.update_applicant(
        db=db,
        applicant_id=applicant_id,
        **update_data
    )

    if not applicant:
        raise HTTPException(status_code=404, detail="Applicant not found")

    return ApplicantResponse.model_validate(applicant)


@router.delete("/{applicant_id}")
async def delete_applicant(
    applicant_id: int,
    db: Session = Depends(get_db)
):
    """
    지원자 삭제

    Args:
        applicant_id: 지원자 ID
        db: Database session

    Returns:
        dict: 삭제 결과
    """
    logger.info(f"Deleting applicant with ID: {applicant_id}")
    applicant_service = ApplicantService()
    success = applicant_service.delete_applicant(db, applicant_id)

    if not success:
        raise HTTPException(status_code=404, detail="Applicant not found")

    return {"message": f"Applicant {applicant_id} deleted successfully"}
