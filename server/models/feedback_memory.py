# server/models/feedback_memory.py
"""
Feedback Memory 모델 - 오답노트 피드백 루프
HR이 수정한 평가 결과를 저장하여 AI가 같은 실수를 반복하지 않도록 함
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Index
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from db.database import Base


class FeedbackMemory(Base):
    """
    피드백 메모리 테이블
    AI의 평가 실수와 HR의 교정을 저장하여 RAG로 재사용
    """
    __tablename__ = "feedback_memory"

    id = Column(Integer, primary_key=True, index=True)

    # 역량 정보
    job_category = Column(String(100), nullable=False, index=True)  # 예: 'Sales', 'Developer', 'Marketing'
    competency_name = Column(String(100), nullable=False, index=True)  # 예: 'problem_solving', 'communication'

    # AI의 실수 요약 (RAG 검색 대상)
    mistake_summary = Column(Text, nullable=False)  # 예: "공격적인 말투를 부정적으로 평가함"

    # AI의 원본 판단
    ai_score = Column(Integer, nullable=False)  # AI가 매긴 점수
    ai_reasoning = Column(Text, nullable=False)  # AI의 판단 근거

    # HR의 교정
    human_score = Column(Integer, nullable=False)  # HR이 수정한 점수
    correction_guideline = Column(Text, nullable=False)  # 교정 지침

    # 벡터 임베딩 (pgvector)
    embedding = Column(Vector(1536), nullable=False)  # mistake_summary의 임베딩

    # 메타데이터
    evaluation_id = Column(Integer, nullable=True, index=True)  # 원본 평가 ID (참조용)
    applicant_id = Column(Integer, nullable=True)  # 지원자 ID

    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # 복합 인덱스
    __table_args__ = (
        Index('ix_feedback_job_competency', 'job_category', 'competency_name'),
        Index('ix_feedback_embedding_ivfflat', 'embedding', postgresql_using='ivfflat', postgresql_ops={'embedding': 'vector_cosine_ops'}),
    )
