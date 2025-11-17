# server/schemas/persona.py
"""
페르소나 관련 Pydantic 스키마
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class PersonaBase(BaseModel):
    """페르소나 기본 스키마"""
    persona_name: str = Field(..., description="페르소나 이름")
    archetype: str = Field(..., description="아키타입 (analytical, supportive, stress_tester)")
    description: Optional[str] = Field(None, description="페르소나 설명")
    system_prompt: Optional[str] = Field(None, description="LLM 시스템 프롬프트")
    welcome_message: Optional[str] = Field(None, description="첫인사 메시지")
    style_description: Optional[str] = Field(None, description="스타일 설명")
    focus_keywords: Optional[List[str]] = Field(None, description="집중 키워드")
    focus_areas: Optional[List[str]] = Field(None, description="집중 영역")


class PersonaCreate(PersonaBase):
    """페르소나 생성 요청"""
    company_id: int = Field(..., description="회사 ID")


class PersonaResponse(PersonaBase):
    """페르소나 응답"""
    id: int
    company_id: int
    pdf_file_path: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class QuestionBase(BaseModel):
    """질문 기본 스키마"""
    question_text: str = Field(..., description="질문 내용")
    question_type: str = Field(..., description="질문 유형")
    expected_keywords: Optional[List[str]] = Field(None, description="기대되는 키워드")
    evaluation_criteria: Optional[List[str]] = Field(None, description="평가 기준")
    difficulty_level: int = Field(3, ge=1, le=5, description="난이도 (1-5)")


class QuestionCreate(QuestionBase):
    """질문 생성 요청"""
    persona_id: Optional[int] = Field(None, description="페르소나 ID")
    job_id: Optional[int] = Field(None, description="Job ID")


class QuestionResponse(QuestionBase):
    """질문 응답"""
    id: int
    persona_id: Optional[int] = None
    job_id: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True


class PersonaUploadResponse(BaseModel):
    """페르소나 업로드 응답"""
    persona: PersonaResponse
    questions: List[QuestionResponse]
    message: str


class PersonaListResponse(BaseModel):
    """페르소나 리스트 응답"""
    personas: List[PersonaResponse]
    total: int
