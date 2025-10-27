# server/schemas/applicant.py
"""
Applicant related request/response schemas
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime


class ApplicantCreate(BaseModel):
    """지원자 생성 요청"""
    name: str = Field(..., min_length=1, max_length=100, description="이름")
    email: EmailStr = Field(..., description="이메일")
    gender: Optional[str] = Field(None, max_length=20, description="성별")
    education: Optional[str] = Field(None, max_length=200, description="학력")
    birthdate: Optional[str] = Field(None, description="생년월일 (YYYY-MM-DD)")

    # 파일 경로는 별도로 처리
    # portfolio_file은 multipart/form-data로 받음


class ApplicantResponse(BaseModel):
    """지원자 응답"""
    id: int
    name: str
    email: str
    age: Optional[int] = None
    education: Optional[str] = None
    gender: Optional[str] = None
    skills: Optional[List[str]] = None
    total_experience_years: Optional[int] = None
    resume_file_path: Optional[str] = None
    portfolio_file_path: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ApplicantUpdate(BaseModel):
    """지원자 정보 수정"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    email: Optional[EmailStr] = None
    gender: Optional[str] = None
    education: Optional[str] = None
    age: Optional[int] = None


class ApplicantDetailResponse(BaseModel):
    """지원자 상세 정보 (블라인드 모드 고려)"""
    id: int
    name: str
    email: str
    # 블라인드 모드에서는 제외되는 필드들
    age: Optional[int] = None
    education: Optional[str] = None
    gender: Optional[str] = None
    # 항상 포함되는 필드들
    skills: Optional[List[str]] = None
    total_experience_years: Optional[int] = None
    domain_experience: Optional[List[str]] = None
    resume_file_path: Optional[str] = None
    portfolio_file_path: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

class ApplicantProfile(BaseModel):
    """
    매칭 계산에 사용될 지원자 프로필 (이력서 파싱 결과)
    """
    id: int
    skills: Optional[List[str]] = []
    total_experience_years: Optional[int] = 0
    domain_experience: Optional[List[str]] = []
    special_experience: Optional[List[str]] = []

    class Config:
        from_attributes = True