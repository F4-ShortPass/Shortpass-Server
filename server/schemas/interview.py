# app/schemas/interview.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class SegmentMetadata(BaseModel):
    """세그먼트 메타데이터"""
    speaking_speed_wpm: int = Field(..., description="분당 단어 수")
    pause_count: int = Field(..., description="멈춤 횟수")
    filler_words: List[str] = Field(default_factory=list)

class TranscriptSegment(BaseModel):
    """인터뷰 세그먼트 (질문-답변 단위)"""
    segment_id: int
    segment_order: int
    segment_type: str = Field(..., description="common or job_specific")
    
    # 질문
    question_id: int
    question_text: str
    question_type: str
    question_language: str = Field(default="ko")
    
    # 답변
    answer_text: str = Field(..., description="STT 변환된 전체 답변")
    answer_duration_sec: int
    answer_timestamp_start: str
    answer_timestamp_end: str
    answer_language: str = Field(default="ko")
    
    # 오디오
    audio_file_path: Optional[str] = None
    
    # 평가 대상 역량
    target_competencies: Dict[str, List[str]] = Field(
        ...,
        description="평가할 역량 {'common': [...], 'job_specific': [...]}"
    )
    
    # STT 메타
    stt_confidence: float = Field(ge=0, le=1)
    stt_provider: str = Field(default="AWS Transcribe")
    
    metadata: Optional[SegmentMetadata] = None


class InterviewTranscript(BaseModel):
    """전체 인터뷰 Transcript"""
    interview_id: int
    applicant_id: int
    job_id: int
    company_id: int
    
    total_duration_sec: int
    started_at: str
    completed_at: str
    
    segments: List[TranscriptSegment]
    
    full_transcript: str = Field(..., description="전체 대화 내용 (검색용)")
    
    interview_metadata: Dict = Field(
        default_factory=dict,
        description="전체 인터뷰 통계"
    )


# ========== Request/Response ==========

class PrepareInterviewRequest(BaseModel):
    """면접 준비 요청 - 단일 회사 + 순차 페르소나"""
    candidateId: str = Field(..., description="지원자 ID")
    companyId: str = Field(..., description="선택된 회사 ID")
    personaInstanceIds: List[str] = Field(..., description="페르소나 인스턴스 ID 리스트 (순차 패널)")


class PrepareInterviewResponse(BaseModel):
    """면접 준비 응답"""
    interviewId: int = Field(..., description="생성된 면접 세션 ID")
    applicantId: int = Field(..., description="지원자 ID")
    companyId: int = Field(..., description="선택된 회사 ID")
    personaInstanceIds: List[int] = Field(..., description="페르소나 인스턴스 ID 리스트")
    status: str = Field(default="pending", description="면접 상태")
    message: str = Field(default="면접이 준비되었습니다.", description="응답 메시지")
    websocketUrl: str


class EvaluateAnswerRequest(BaseModel):
    """답변 평가 요청"""
    interview_id: int
    question_id: int
    answer_text: str = Field(..., min_length=1)
    is_common: bool = False
    job_id: Optional[int] = None