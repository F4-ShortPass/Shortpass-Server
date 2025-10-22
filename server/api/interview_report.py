# server/api/interview_report.py
"""
면접 결과 리포트 API
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from db.database import get_db
from models.interview import (
    InterviewSession, SessionPersona, PersonaInstance,
    SessionScore, SessionExplanation, SessionTranscript
)
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging


router = APIRouter()
logger = logging.getLogger("uvicorn")


# ==================== Schemas ====================

class PersonaScoreDetail(BaseModel):
    """페르소나별 점수 상세"""
    persona_instance_id: int
    persona_name: str
    criterion_key: str
    score: float
    explanation: Optional[str] = None
    log_json: Optional[Dict[str, Any]] = None                                                


class PersonaSummary(BaseModel):
    """페르소나별 요약"""
    persona_instance_id: int
    persona_name: str
    average_score: float
    scores: Dict[str, float]
    explanations: Dict[str, str]


class InterviewReportResponse(BaseModel):
    """면접 리포트 응답"""
    interview_id: int
    applicant_id: int
    company_id: int
    status: str

    # 종합 점수
    overall_score: float
    overall_grade: str

    # 페르소나별 요약
    persona_summaries: List[PersonaSummary]

    # 페르소나별 상세 점수
    detailed_scores: List[PersonaScoreDetail]

    # 대화 기록 요약
    total_turns: int
    transcript_sample: List[str]


# ==================== Endpoints ====================

@router.get("/interviews/{interview_id}/report", response_model=InterviewReportResponse)
def get_interview_report(
    interview_id: int,
    db: Session = Depends(get_db)
):
    """
    면접 결과 리포트 조회

    - 종합 점수 + 페르소나별 분해
    - 각 페르소나의 평가 기준별 점수 및 근거
    - 대화 기록 요약
    """
    logger.info(f"Getting interview report for interview ID: {interview_id}")
    # 1. 면접 세션 조회
    session = db.query(InterviewSession).filter(InterviewSession.id == interview_id).first()
    if not session:
        raise HTTPException(status_code=404, detail=f"면접 세션 {interview_id}를 찾을 수 없습니다.")

    # 2. 세션 페르소나 조회
    session_personas = (
        db.query(SessionPersona)
        .filter(SessionPersona.session_id == interview_id)
        .order_by(SessionPersona.order)
        .all()
    )

    if not session_personas:
        raise HTTPException(status_code=404, detail="세션에 페르소나가 없습니다.")

    # 3. 페르소나 인스턴스 조회
    persona_instances = {}
    for sp in session_personas:
        pi = db.query(PersonaInstance).filter(PersonaInstance.id == sp.persona_instance_id).first()
        if pi:
            persona_instances[pi.id] = pi

    # 4. 점수 조회
    scores = (
        db.query(SessionScore)
        .filter(SessionScore.session_id == interview_id)
        .all()
    )

    # 5. 설명 조회
    explanations = (
        db.query(SessionExplanation)
        .filter(SessionExplanation.session_id == interview_id)
        .all()
    )

    # 6. 대화 기록 조회
    transcripts = (
        db.query(SessionTranscript)
        .filter(SessionTranscript.session_id == interview_id)
        .order_by(SessionTranscript.turn)
        .all()
    )

    # 7. 데이터 가공
    # 페르소나별 점수 집계
    persona_scores_map: Dict[int, Dict[str, float]] = {}
    for score in scores:
        if score.persona_instance_id not in persona_scores_map:
            persona_scores_map[score.persona_instance_id] = {}
        persona_scores_map[score.persona_instance_id][score.criterion_key] = score.score

    # 페르소나별 설명 집계
    persona_explanations_map: Dict[int, Dict[str, str]] = {}
    for expl in explanations:
        if expl.persona_instance_id not in persona_explanations_map:
            persona_explanations_map[expl.persona_instance_id] = {}
        persona_explanations_map[expl.persona_instance_id][expl.criterion_key] = expl.explanation or ""

    # 페르소나별 요약 생성
    persona_summaries = []
    total_score_sum = 0.0
    total_score_count = 0

    for persona_id, persona_instance in persona_instances.items():
        scores_dict = persona_scores_map.get(persona_id, {})
        explanations_dict = persona_explanations_map.get(persona_id, {})

        if scores_dict:
            avg_score = sum(scores_dict.values()) / len(scores_dict)
        else:
            avg_score = 0.0

        persona_summaries.append(PersonaSummary(
            persona_instance_id=persona_id,
            persona_name=persona_instance.instance_name,
            average_score=round(avg_score, 2),
            scores=scores_dict,
            explanations=explanations_dict
        ))

        total_score_sum += avg_score
        total_score_count += 1

    # 전체 평균 점수
    overall_score = round(total_score_sum / total_score_count, 2) if total_score_count > 0 else 0.0

    # 등급 계산
    if overall_score >= 90:
        grade = "S"
    elif overall_score >= 80:
        grade = "A"
    elif overall_score >= 70:
        grade = "B"
    elif overall_score >= 60:
        grade = "C"
    else:
        grade = "D"

    # 상세 점수 리스트
    detailed_scores = []
    for score in scores:
        persona_instance = persona_instances.get(score.persona_instance_id)
        expl = next((e for e in explanations if e.persona_instance_id == score.persona_instance_id and e.criterion_key == score.criterion_key), None)

        detailed_scores.append(PersonaScoreDetail(
            persona_instance_id=score.persona_instance_id,
            persona_name=persona_instance.instance_name if persona_instance else "Unknown",
            criterion_key=score.criterion_key,
            score=score.score,
            explanation=expl.explanation if expl else None,
            log_json=expl.log_json if expl else None
        ))

    # 대화 기록 샘플
    transcript_sample = [t.text for t in transcripts[:10]]

    return InterviewReportResponse(
        interview_id=interview_id,
        applicant_id=session.applicant_id,
        company_id=session.company_id,
        status=session.status.value,
        overall_score=overall_score,
        overall_grade=grade,
        persona_summaries=persona_summaries,
        detailed_scores=detailed_scores,
        total_turns=len(transcripts),
        transcript_sample=transcript_sample
    )
