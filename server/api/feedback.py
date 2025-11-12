# server/api/feedback.py
"""
Feedback API - 오답노트 피드백 루프
HR이 평가 결과를 수정하면 이를 저장하고, 다음 평가 시 참고할 수 있도록 함
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from sqlalchemy.orm import Session
from openai import AsyncOpenAI

from db.database import get_db
from services.feedback.feedback_manager import FeedbackManager
from core.config import settings

router = APIRouter(prefix="/api/v1/feedback", tags=["feedback"])

# OpenAI 클라이언트 초기화
openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


# ==================== Schemas ====================

class FeedbackCreateRequest(BaseModel):
    """피드백 저장 요청"""
    evaluation_id: int = Field(..., description="평가 ID")
    applicant_id: Optional[int] = Field(None, description="지원자 ID")
    job_category: str = Field(..., description="직무 카테고리 (예: Sales, Engineering)")
    competency_name: str = Field(..., description="역량 이름 (예: problem_solving)")

    # AI 판단
    ai_score: int = Field(..., ge=0, le=100, description="AI가 매긴 점수")
    ai_reasoning: str = Field(..., description="AI의 판단 근거")

    # HR 수정
    human_score: int = Field(..., ge=0, le=100, description="HR이 수정한 점수")
    human_reasoning: str = Field(..., description="HR의 수정 사유")


class FeedbackResponse(BaseModel):
    """피드백 응답"""
    id: int
    job_category: str
    competency_name: str
    mistake_summary: str
    correction_guideline: str
    ai_score: int
    human_score: int
    created_at: str


class RelevantFeedbackRequest(BaseModel):
    """관련 피드백 조회 요청"""
    job_category: str
    competency_name: str
    current_context: str = Field(..., description="현재 평가 컨텍스트 (트랜스크립트 요약)")
    top_k: int = Field(3, ge=1, le=10, description="반환할 최대 개수")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="최소 유사도")


class RelevantFeedbackItem(BaseModel):
    """관련 피드백 항목"""
    id: int
    mistake_summary: str
    correction_guideline: str
    ai_score: int
    human_score: int
    similarity: float


# ==================== Endpoints ====================

@router.post("/", response_model=FeedbackResponse)
async def create_feedback(
    request: FeedbackCreateRequest,
    db: Session = Depends(get_db)
):
    """
    HR의 평가 수정을 피드백으로 저장

    **사용 시나리오:**
    1. HR이 CandidateEvaluation 페이지에서 AI 평가 점수를 수정
    2. 수정 사유를 입력하고 저장
    3. 이 API가 호출되어 오답노트에 기록됨

    **Example:**
    ```json
    {
      "evaluation_id": 123,
      "job_category": "Sales",
      "competency_name": "interpersonal_skill",
      "ai_score": 70,
      "ai_reasoning": "지원자가 공격적인 말투를 사용하여 대인관계 역량이 부족해 보임",
      "human_score": 90,
      "human_reasoning": "영업 직무에서는 공격성이 아니라 적극성으로 해석해야 함. 고객 설득력이 뛰어남"
    }
    ```
    """
    try:
        # FeedbackManager 초기화
        feedback_manager = FeedbackManager(db=db, openai_client=openai_client)

        # 피드백 저장
        feedback = await feedback_manager.save_feedback(
            job_category=request.job_category,
            competency_name=request.competency_name,
            ai_score=request.ai_score,
            ai_reasoning=request.ai_reasoning,
            human_score=request.human_score,
            human_reasoning=request.human_reasoning,
            evaluation_id=request.evaluation_id,
            applicant_id=request.applicant_id
        )

        return FeedbackResponse(
            id=feedback.id,
            job_category=feedback.job_category,
            competency_name=feedback.competency_name,
            mistake_summary=feedback.mistake_summary,
            correction_guideline=feedback.correction_guideline,
            ai_score=feedback.ai_score,
            human_score=feedback.human_score,
            created_at=feedback.created_at.isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")


@router.post("/search", response_model=List[RelevantFeedbackItem])
async def search_relevant_feedback(
    request: RelevantFeedbackRequest,
    db: Session = Depends(get_db)
):
    """
    현재 평가 상황과 유사한 과거 피드백 검색 (RAG)

    **사용 시나리오:**
    - CompetencyAgent가 평가 시작 전에 호출
    - 유사한 과거 실수 사례를 찾아서 프롬프트에 주입

    **Example:**
    ```json
    {
      "job_category": "Sales",
      "competency_name": "interpersonal_skill",
      "current_context": "지원자가 적극적이고 설득력 있는 말투로 고객 응대 사례를 설명함",
      "top_k": 3,
      "similarity_threshold": 0.7
    }
    ```
    """
    try:
        # FeedbackManager 초기화
        feedback_manager = FeedbackManager(db=db, openai_client=openai_client)

        # 관련 피드백 검색
        feedbacks = await feedback_manager.get_relevant_feedback(
            job_category=request.job_category,
            competency_name=request.competency_name,
            current_context=request.current_context,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )

        return [
            RelevantFeedbackItem(
                id=fb["id"],
                mistake_summary=fb["mistake_summary"],
                correction_guideline=fb["correction_guideline"],
                ai_score=fb["ai_score"],
                human_score=fb["human_score"],
                similarity=fb["similarity"]
            )
            for fb in feedbacks
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search feedback: {str(e)}")


@router.get("/stats")
async def get_feedback_stats(
    db: Session = Depends(get_db)
):
    """
    피드백 통계 조회

    Returns:
        {
          "total_feedbacks": 150,
          "by_competency": {
            "problem_solving": 25,
            "interpersonal_skill": 30,
            ...
          },
          "by_job_category": {
            "Sales": 50,
            "Engineering": 60,
            ...
          }
        }
    """
    from sqlalchemy import func
    from models.feedback_memory import FeedbackMemory

    try:
        # 전체 개수
        total = db.query(func.count(FeedbackMemory.id)).scalar()

        # 역량별 개수
        by_competency = db.query(
            FeedbackMemory.competency_name,
            func.count(FeedbackMemory.id).label('count')
        ).group_by(FeedbackMemory.competency_name).all()

        # 직무별 개수
        by_job_category = db.query(
            FeedbackMemory.job_category,
            func.count(FeedbackMemory.id).label('count')
        ).group_by(FeedbackMemory.job_category).all()

        return {
            "total_feedbacks": total,
            "by_competency": {item.competency_name: item.count for item in by_competency},
            "by_job_category": {item.job_category: item.count for item in by_job_category}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")
