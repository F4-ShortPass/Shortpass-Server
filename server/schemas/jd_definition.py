from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class CompetencyItem(BaseModel):
    """JD에서 추출된 개별 역량 항목 스키마"""
    id: str = Field(..., description="역량 식별자 (예: COMP_01)")
    name: str = Field(..., description="역량 명칭 (JD에 명시된 키워드)")
    definition: str = Field(..., description="역량의 정의 (JD 문맥에 따른 구체적 설명)")
    evaluation_criteria: List[str] = Field(..., description="구체적으로 확인해야 할 행동 지표 리스트")
    interview_question_guide: Optional[str] = Field(None, description="이 역량을 검증하기 위한 질문 가이드")

class JDParseResult(BaseModel):
    """
    JD PDF 파싱 및 역량 분석 결과의 표준 JSON 형식 스키마
    """
    jd_id: int = Field(..., description="Job Description ID (DB Job ID)")
    company_id: int = Field(..., description="회사 ID")
    company_name: str = Field(..., description="회사명")
    job_title: str = Field(..., description="채용 직무 제목")
    raw_text: str = Field(..., description="JD PDF에서 추출된 전체 텍스트")
    company_description_summary: Optional[str] = Field(None, description="회사 소개 요약 텍스트")
    job_description_summary: Optional[str] = Field(None, description="직무 내용 요약 텍스트")

    common_competencies: List[CompetencyItem] = Field(
        ...,
        description="JD에서 추출된 공통 핵심 역량 리스트 (5개 이내)"
    )
    job_competencies: List[CompetencyItem] = Field(
        ...,
        description="JD에서 추출된 직무 특화 핵심 역량 리스트 (5개 이내)"
    )
    weights: Dict[str, Dict[str, float]] = Field(
        ...,
        description="카테고리별/역량별 가중치 정보. (예: {'category': {'job_competencies': 0.6, 'common_competencies': 0.4}, 'competency': {'COMP_01': 0.2, ...}})"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="파싱 과정에서 생성된 추가 메타데이터 (예: LLM 모델명, 프롬프트 버전)"
    )
