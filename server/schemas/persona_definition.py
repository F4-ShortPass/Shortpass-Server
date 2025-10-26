from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# Reusing CompetencyItem from jd_definition.py
from .jd_definition import CompetencyItem

class PersonaMeta(BaseModel):
    """페르소나 에이전트의 메타 정보 (정체성, 말투 등)"""
    identity: str = Field(..., description="페르소나의 정체성 (예: 삼성물산 패션부문 15년차 시니어 채용 담당자)")
    name: str = Field(..., description="페르소나의 이름 (예: 김삼성 책임)")
    tone_and_manner: List[str] = Field(..., description="페르소나의 말투 및 태도 특성")
    system_instruction: str = Field(..., description="LLM에 전달될 시스템 지시 프롬프트")

class JobInfo(BaseModel):
    """직무에 대한 요약 정보"""
    company_name: str = Field(..., description="회사명")
    job_title: str = Field(..., description="채용 직무 제목")
    roles: List[str] = Field(..., description="직무의 주요 역할 리스트")
    description_summary: str = Field(..., description="직무에 대한 요약 설명")

class PersonaConfig(BaseModel):
    """
    면접 페르소나 및 평가 설정을 위한 통합 스키마
    JD 분석 결과를 기반으로 에이전트 및 프론트엔드가 사용
    """
    job_info: JobInfo = Field(..., description="직무에 대한 요약 정보")
    persona_meta: PersonaMeta = Field(..., description="면접관 페르소나의 메타 정보")
    
    common_competencies: List[CompetencyItem] = Field(
        ...,
        description="모든 직무에 공통적으로 적용되는 핵심 역량 리스트"
    )
    job_competencies: List[CompetencyItem] = Field(
        ...,
        description="해당 직무에 특화된 핵심 역량 리스트"
    )
    
    initial_questions: List[str] = Field(
        ...,
        description="면접 시작 시 AI 면접관이 던질 초기 질문 리스트"
    )
    
    weights: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="평가 가중치 정보 (예: {'category': {'common': 0.4, 'job': 0.6}})"
    )
    
    meta_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="설정 파일 자체에 대한 메타데이터 (버전, 생성일 등)"
    )