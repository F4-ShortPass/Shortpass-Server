"""
Evaluation List API - 지원자 목록 및 상세 조회
DB 데이터 있으면 사용, 없으면 mock fallback
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, text
from db.database import get_db
from models.evaluation import Evaluation
from models.interview import Applicant
from models.job import Job
from models.company import Company
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging

router = APIRouter(prefix="/evaluations")
logger = logging.getLogger("uvicorn")


def get_mock_applicants_list():
    """Mock 지원자 목록 데이터 반환"""
    mock_applicants = [
        {"rank": 1, "applicant_id": 1, "applicant_name": "이서준", "track": "HRD 컨설턴트", "interview_date": "2025-11-05", "total_score": 91, "strengths": "논리적 사고, 고객 대응", "weaknesses": "비즈니스 감각", "ai_summary_comment": "구조적인 사고와 응대 능력이 우수합니다.", "status": "추천", "competency_scores": [{"name": "Data Insight", "score": 95}, {"name": "Strategic Solving", "score": 90}, {"name": "Value Chain", "score": 85}, {"name": "Marketing", "score": 88}, {"name": "Stakeholder", "score": 80}]},
        {"rank": 2, "applicant_id": 2, "applicant_name": "김다은", "track": "HRD 컨설턴트", "interview_date": "2025-11-08", "total_score": 88, "strengths": "커뮤니케이션, 문제 해결력", "weaknesses": "고객 니즈 분석", "ai_summary_comment": "논리 전개가 깔끔합니다.", "status": "추천", "competency_scores": [{"name": "Data Insight", "score": 85}, {"name": "Strategic Solving", "score": 92}, {"name": "Value Chain", "score": 80}, {"name": "Marketing", "score": 90}, {"name": "Stakeholder", "score": 85}]},
        {"rank": 3, "applicant_id": 3, "applicant_name": "송유진", "track": "일본사업 매니저", "interview_date": "2025-11-03", "total_score": 86, "strengths": "외국어 능력, 커뮤니케이션", "weaknesses": "세부 실행 전략", "ai_summary_comment": "다문화 이해도가 돋보입니다.", "status": "추천", "competency_scores": [{"name": "Data Insight", "score": 80}, {"name": "Strategic Solving", "score": 85}, {"name": "Value Chain", "score": 88}, {"name": "Marketing", "score": 82}, {"name": "Stakeholder", "score": 90}]},
        {"rank": 4, "applicant_id": 4, "applicant_name": "박현우", "track": "HRD 컨설턴트", "interview_date": "2025-11-07", "total_score": 83, "strengths": "데이터 활용, 실행력", "weaknesses": "프레젠테이션", "ai_summary_comment": "실행 방안이 명확합니다.", "status": "보류", "competency_scores": [{"name": "Data Insight", "score": 90}, {"name": "Strategic Solving", "score": 78}, {"name": "Value Chain", "score": 85}, {"name": "Marketing", "score": 75}, {"name": "Stakeholder", "score": 80}]},
        {"rank": 5, "applicant_id": 5, "applicant_name": "김하린", "track": "교수운영팀", "interview_date": "2025-11-04", "total_score": 79, "strengths": "프로세스 이해, 문제 분석", "weaknesses": "리더십", "ai_summary_comment": "업무 프로세스 이해도가 높습니다.", "status": "보류", "competency_scores": [{"name": "Data Insight", "score": 82}, {"name": "Strategic Solving", "score": 80}, {"name": "Value Chain", "score": 75}, {"name": "Marketing", "score": 78}, {"name": "Stakeholder", "score": 70}]},
    ]

    return {
        "company_name": "캐럿글로벌 (CARROT Global)",
        "job_title": "HRD 컨설턴트 (신입/경력)",
        "total_applicants": len(mock_applicants),
        "completed_evaluations": len(mock_applicants),
        "average_score": 85.4,
        "applicants": mock_applicants
    }


def get_status_by_score(score: float) -> str:
    """점수 기반 상태 반환"""
    if score >= 85:
        return "추천"
    elif score >= 70:
        return "보류"
    else:
        return "검토 필요"


def extract_strengths_weaknesses(eval_obj):
    """평가 결과에서 강점/약점 추출"""
    strengths = []
    weaknesses = []

    # key_insights에서 추출
    if eval_obj.key_insights:
        strengths = eval_obj.key_insights.get("strengths", [])[:2]
        weaknesses = eval_obj.key_insights.get("weaknesses", [])[:2]

    # fit_analysis에서도 추출 시도
    if not strengths and eval_obj.fit_analysis:
        strengths = eval_obj.fit_analysis.get("strengths", [])[:2]
    if not weaknesses and eval_obj.fit_analysis:
        weaknesses = eval_obj.fit_analysis.get("areas_for_improvement", [])[:2]

    return ", ".join(strengths) if strengths else "분석 중", ", ".join(weaknesses) if weaknesses else "분석 중"


def extract_competency_scores(eval_obj):
    """평가 결과에서 역량별 점수 추출"""
    scores = []

    # job_aggregation에서 추출
    if eval_obj.job_aggregation:
        for name, data in eval_obj.job_aggregation.items():
            if isinstance(data, dict):
                scores.append({"name": name, "score": data.get("score", 0)})
            else:
                scores.append({"name": name, "score": data})

    # 기본 역량 필드에서 추출
    if not scores:
        competency_fields = [
            ("job_expertise", "직무 전문성"),
            ("analytical", "분석력"),
            ("execution", "실행력"),
            ("relationship_building", "관계 구축"),
            ("resilience", "회복탄력성"),
        ]
        for field, label in competency_fields:
            data = getattr(eval_obj, field, None)
            if data and isinstance(data, dict):
                scores.append({"name": label, "score": data.get("score", 0)})

    return scores if scores else [{"name": "종합", "score": eval_obj.match_score or 0}]


# ------------------ Swagger Schemas ------------------

class AnalysisSummary(BaseModel):
    aggregator_summary: Optional[str] = Field(None, description="지민 에이전트가 생성한 심사평/요약")
    overall_applicant_summary: Optional[str] = Field(None, description="후처리 레이어가 생성한 전체 지원자 요약")
    positive_keywords: List[str] = Field(default_factory=list, description="핵심 긍정 키워드")
    negative_keywords: List[str] = Field(default_factory=list, description="핵심 보완 키워드")
    recommended_questions: List[str] = Field(default_factory=list, description="추가로 물어볼 추천 질문 리스트")


class PostProcessingMeta(BaseModel):
    version: Optional[str] = Field(None, description="후처리 버전")
    source: Optional[str] = Field(None, description="rules / llm 등 생성 소스")
    llm_used: Optional[bool] = Field(None, description="LLM 사용 여부")


class AggregatedEvaluationPayload(BaseModel):
    final_result: Optional[Dict[str, Any]] = Field(None, description="최종 점수/역량별 결과 블록")
    analysis_summary: Optional[AnalysisSummary] = Field(None, description="요약/키워드/추천질문 블록")
    post_processing: Optional[PostProcessingMeta] = Field(None, description="후처리 메타 정보")


class ApplicantEvaluationDetail(BaseModel):
    evaluation_id: int
    job_id: Optional[int] = None
    applicant_id: int
    applicant_name: str
    job_title: Optional[str] = None
    company_id: Optional[int] = None
    overall_score: float
    normalized_score: Optional[float] = None
    confidence_score: Optional[float] = None
    competency_scores: Optional[Dict[str, Any]] = None
    job_aggregation: Optional[Dict[str, Any]] = None
    common_aggregation: Optional[Dict[str, Any]] = None
    fit_analysis: Optional[Dict[str, Any]] = None
    key_insights: Optional[Dict[str, Any]] = None
    reasoning_log: Optional[Dict[str, Any]] = None
    aggregated_evaluation: Optional[AggregatedEvaluationPayload] = None
    analysis_summary: Optional[AnalysisSummary] = None
    post_processing: Optional[PostProcessingMeta] = None
    match_result: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None


@router.get("/jobs/{job_id}/applicants")
async def get_applicants_list(
    job_id: int,
    db: Session = Depends(get_db)
):
    """지원자 목록 조회 - DB 우선, 없으면 mock"""
    logger.info(f"Getting applicants list for job ID: {job_id}")

    # DB에서 평가 결과 조회
    evaluations = db.query(Evaluation).filter(
        Evaluation.job_id == job_id
    ).order_by(Evaluation.match_score.desc()).all()

    # 데이터 없으면 mock 반환
    if not evaluations:
        logger.info(f"No evaluations found for job_id={job_id}, returning mock data")
        return get_mock_applicants_list()

    # Job, Company 정보 조회
    job = db.query(Job).filter(Job.id == job_id).first()
    company = None
    if job:
        company = db.query(Company).filter(Company.id == job.company_id).first()

    # 응답 데이터 구성
    applicants = []
    total_score = 0

    for idx, eval_obj in enumerate(evaluations):
        applicant = db.query(Applicant).filter(Applicant.id == eval_obj.applicant_id).first()
        if not applicant:
            continue

        strengths, weaknesses = extract_strengths_weaknesses(eval_obj)
        competency_scores = extract_competency_scores(eval_obj)

        # AI 요약 추출
        ai_summary = ""
        if eval_obj.fit_analysis and isinstance(eval_obj.fit_analysis, dict):
            ai_summary = eval_obj.fit_analysis.get("summary", "") or eval_obj.fit_analysis.get("overall_assessment", "")

        score = eval_obj.match_score or 0
        total_score += score

        applicants.append({
            "rank": idx + 1,
            "applicant_id": eval_obj.applicant_id,
            "applicant_name": applicant.name,
            "track": job.title if job else "미정",
            "interview_date": eval_obj.created_at.strftime("%Y-%m-%d") if eval_obj.created_at else "",
            "total_score": round(score),
            "strengths": strengths,
            "weaknesses": weaknesses,
            "ai_summary_comment": ai_summary[:100] + "..." if len(ai_summary) > 100 else ai_summary,
            "status": get_status_by_score(score),
            "competency_scores": competency_scores
        })

    avg_score = total_score / len(applicants) if applicants else 0

    return {
        "company_name": company.name if company else "회사명 미정",
        "job_title": job.title if job else "직무명 미정",
        "total_applicants": len(applicants),
        "completed_evaluations": len([a for a in applicants if a["status"] != "검토 필요"]),
        "average_score": round(avg_score, 1),
        "applicants": applicants
    }


@router.get("/jobs/{job_id}/applicants/{applicant_id}/result", response_model=ApplicantEvaluationDetail)
async def get_applicant_detail(
    job_id: int,
    applicant_id: int,
    db: Session = Depends(get_db)
):
    """지원자 상세 평가 결과"""
    logger.info(f"Getting applicant detail for job ID: {job_id}, applicant ID: {applicant_id}")

    eval_obj = db.query(Evaluation).filter(
        Evaluation.job_id == job_id,
        Evaluation.applicant_id == applicant_id
    ).first()

    if not eval_obj:
        raise HTTPException(
            status_code=404,
            detail=f"Evaluation not found for job_id={job_id}, applicant_id={applicant_id}"
        )

    applicant = db.query(Applicant).filter(Applicant.id == applicant_id).first()
    aggregated_eval = eval_obj.aggregated_evaluation or {}
    job = db.query(Job).filter(Job.id == eval_obj.job_id).first() if eval_obj.job_id else None

    return {
        "evaluation_id": eval_obj.id,
        "job_id": eval_obj.job_id,
        "applicant_id": eval_obj.applicant_id,
        "applicant_name": applicant.name if applicant else "Unknown",
        "job_title": job.title if job else None,
        "company_id": job.company_id if job else None,
        "overall_score": round(eval_obj.match_score, 1) if eval_obj.match_score else 0,
        "normalized_score": round(eval_obj.normalized_score, 1) if eval_obj.normalized_score else None,
        "confidence_score": round(eval_obj.confidence_score, 2) if eval_obj.confidence_score else None,
        "competency_scores": eval_obj.competency_scores,
        "job_aggregation": eval_obj.job_aggregation,
        "common_aggregation": eval_obj.common_aggregation,
        "fit_analysis": eval_obj.fit_analysis,
        "key_insights": eval_obj.key_insights,
        "reasoning_log": eval_obj.reasoning_log,
        "aggregated_evaluation": aggregated_eval,
        "analysis_summary": aggregated_eval.get("analysis_summary"),
        "post_processing": aggregated_eval.get("post_processing"),
        "match_result": eval_obj.match_result,
        "created_at": eval_obj.created_at.isoformat() if eval_obj.created_at else None
    }


@router.get("/jobs/{job_id}/statistics")
async def get_job_statistics(
    job_id: int,
    db: Session = Depends(get_db)
):
    """Job별 통계"""
    logger.info(f"Getting job statistics for job ID: {job_id}")

    evaluations = db.query(Evaluation).filter(Evaluation.job_id == job_id).all()

    if not evaluations:
        return {
            "job_id": job_id,
            "total_evaluations": 0,
            "average_score": 0,
            "min_score": 0,
            "max_score": 0
        }

    scores = [e.match_score for e in evaluations if e.match_score]

    return {
        "job_id": job_id,
        "total_evaluations": len(evaluations),
        "average_score": round(sum(scores) / len(scores), 2) if scores else 0,
        "min_score": round(min(scores), 1) if scores else 0,
        "max_score": round(max(scores), 1) if scores else 0
    }
