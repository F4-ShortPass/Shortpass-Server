# server/schemas/company.py
"""
Company related request/response schemas
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime


class CompanyCreate(BaseModel):
    """기업 생성 요청"""
    name: str = Field(..., min_length=1, max_length=200, description="회사명")
    company_values_text: Optional[str] = Field(None, description="회사 가치관/인재상")
    company_culture_desc: Optional[str] = Field(None, description="조직 문화 설명")
    blind_mode: bool = Field(default=False, description="블라인드 채용 여부")

    # JD PDF는 별도로 multipart/form-data로 받음


class CompanyResponse(BaseModel):
    """기업 응답"""
    id: int
    name: str
    company_values_text: Optional[str] = None
    company_culture_desc: Optional[str] = None
    blind_mode: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CompanyUpdate(BaseModel):
    """기업 정보 수정"""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    company_values_text: Optional[str] = None
    company_culture_desc: Optional[str] = None
    competencies_weights: Optional[Dict[str, float]] = None
    priority_weights: Optional[Dict[str, float]] = None
    blind_mode: Optional[bool] = None


class CompanyDetailResponse(BaseModel):
    """기업 상세 정보"""
    id: int
    name: str
    company_values_text: Optional[str] = None
    company_culture_desc: Optional[str] = None
    core_values: Optional[List[str]] = None
    competencies_weights: Optional[Dict[str, float]] = None
    priority_weights: Optional[Dict[str, float]] = None
    blind_mode: bool
    created_at: datetime
    updated_at: datetime
    # 관련 Job 개수
    total_jobs: int = 0

    class Config:
        from_attributes = True

class CompanyProfile(BaseModel):
    """
    매칭 계산에 사용될 기업 및 직무 프로필
    Company 정보와 Job 정보를 결합
    """
    # Company 정보
    id: int
    name: str
    core_values: Optional[List[str]] = []
    competencies_weights: Optional[Dict[str, float]] = {}
    priority_weights: Optional[Dict[str, Dict[str, float]]] = {}

    # Job 정보 (JD 파싱 결과)
    required_skills: Optional[List[str]] = []
    preferred_skills: Optional[List[str]] = []
    min_years_experience: Optional[int] = 0
    preferred_domains: Optional[List[str]] = []
    preferred_special_experience: Optional[List[str]] = []

    class Config:
        from_attributes = True

class QuestionCreate(BaseModel):
    """추가 질문 생성 요청"""
    question_text: str = Field(..., min_length=1, description="질문 내용")
    question_type: str = Field(default="custom", description="질문 타입")
    job_id: Optional[int] = Field(None, description="특정 Job에 연결 (선택)")


class QuestionResponse(BaseModel):
    """질문 응답"""
    id: int
    question_text: str
    question_type: str
    job_id: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True