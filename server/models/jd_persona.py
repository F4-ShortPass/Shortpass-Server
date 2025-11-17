# server/models/jd_persona.py
"""
JD 기반 페르소나 데이터 모델
"""
from sqlalchemy import Column, Integer, String, Text, JSON, DateTime, ForeignKey, Boolean
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from typing import List, Dict, Any
import json

from db.database import Base


class JDPersona(Base):
    """
    JD 기반으로 생성된 페르소나 정보
    """
    __tablename__ = "jd_personas"

    id = Column(Integer, primary_key=True, index=True)

    # Job 정보
    job_id = Column(Integer, index=True, nullable=False)  # Job과 연결
    company_id = Column(Integer, index=True, nullable=False)

    # 생성된 페르소나 데이터
    company_name = Column(String(255), nullable=True)

    # 역량 정보 (JSON 형태로 저장)
    common_competencies = Column(JSON, nullable=False)  # 공통 역량 6개
    job_competencies = Column(JSON, nullable=False)     # 직무 역량 6개

    # 기업 질문
    core_questions = Column(JSON, nullable=False)       # 기업에서 입력한 3개 질문

    # 페르소나 정보
    persona_summary = Column(JSON, nullable=False)      # 생성된 페르소나들 정보

    # 분석 결과
    analysis_summary = Column(Text, nullable=True)      # JD 분석 요약

    # 시각화 데이터
    visualization_config = Column(JSON, nullable=True)  # 육각형 그래프 설정

    # 메타데이터
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # 사용 여부 (페르소나 활성화/비활성화)
    is_active = Column(Boolean, default=True)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 형태로 변환"""
        return {
            "id": self.id,
            "job_id": self.job_id,
            "company_id": self.company_id,
            "company_name": self.company_name,
            "common_competencies": self.common_competencies,
            "job_competencies": self.job_competencies,
            "core_questions": self.core_questions,
            "persona_summary": self.persona_summary,
            "analysis_summary": self.analysis_summary,
            "visualization_config": self.visualization_config,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "is_active": self.is_active
        }

    @classmethod
    def create_from_generation_result(
        cls,
        job_id: int,
        company_id: int,
        generation_result: Dict[str, Any],
        visualization_data: Dict[str, Any] = None
    ):
        """페르소나 생성 결과로부터 객체 생성"""
        return cls(
            job_id=job_id,
            company_id=company_id,
            company_name=generation_result.get("company", "Unknown Company"),
            common_competencies=generation_result.get("common_competencies", []),
            job_competencies=generation_result.get("job_competencies", []),
            core_questions=generation_result.get("core_questions", []),
            persona_summary=generation_result.get("persona_summary", []),
            analysis_summary=generation_result.get("analysis_summary", ""),
            visualization_config=visualization_data,
            is_active=True
        )

    def get_persona_count(self) -> int:
        """생성된 페르소나 개수"""
        return len(self.persona_summary) if self.persona_summary else 0

    def get_persona_types(self) -> List[str]:
        """페르소나 타입들"""
        if not self.persona_summary:
            return []
        return [persona.get("type", "Unknown") for persona in self.persona_summary]

    def get_competency_mapping(self) -> Dict[str, List[str]]:
        """역량 매핑 정보"""
        return {
            "common": self.common_competencies or [],
            "job_specific": self.job_competencies or []
        }


class JDPersonaQuestion(Base):
    """
    페르소나별 추가 질문 (확장 가능)
    """
    __tablename__ = "jd_persona_questions"

    id = Column(Integer, primary_key=True, index=True)
    persona_id = Column(Integer, ForeignKey("jd_personas.id"), index=True)

    # 질문 정보
    persona_type = Column(String(100), nullable=False)  # 페르소나 타입
    question_text = Column(Text, nullable=False)        # 질문 내용
    question_category = Column(String(50), nullable=True)  # 질문 카테고리
    target_competencies = Column(JSON, nullable=True)   # 평가 대상 역량

    # 메타데이터
    created_at = Column(DateTime, server_default=func.now())
    is_active = Column(Boolean, default=True)

    # 관계
    persona = relationship("JDPersona", back_populates="questions")

# 관계 설정
JDPersona.questions = relationship("JDPersonaQuestion", back_populates="persona", cascade="all, delete-orphan")