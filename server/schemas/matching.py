# server/schemas/matching.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


class MatchScore(BaseModel):
    """매칭 점수"""
    total_score: float = Field(..., ge=0, le=100)
    
    job_expertise: float = Field(..., ge=0, le=100)
    problem_solving: float = Field(..., ge=0, le=100)
    organizational_fit: float = Field(..., ge=0, le=100)
    growth_potential: float = Field(..., ge=0, le=100)
    interpersonal_skill: float = Field(..., ge=0, le=100)
    achievement_motivation: float = Field(..., ge=0, le=100)
    
    strengths: List[str] = []
    weaknesses: List[str] = []
    
    def get_all_competency_scores(self) -> Dict[str, float]:
        """6개 역량 점수를 dict로 반환"""
        return {
            "job_expertise": self.job_expertise,
            "problem_solving": self.problem_solving,
            "organizational_fit": self.organizational_fit,
            "growth_potential": self.growth_potential,
            "interpersonal_skill": self.interpersonal_skill,
            "achievement_motivation": self.achievement_motivation
        }


class CompanyMatch(BaseModel):
    """회사 매칭 정보"""
    company_id: int
    job_id: int
    company_name: str
    job_title: str = "개발자"
    match_score: MatchScore


class CompanyMatchResult(BaseModel):
    """지원자가 보는 매칭 결과"""
    applicant_id: int
    applicant_name: str
    interview_session_id: int
    interview_completed_at: datetime
    
    match: CompanyMatch


class ApplicantMatch(BaseModel):
    """지원자 매칭 정보"""
    rank: int
    applicant_id: int
    applicant_name: str
    
    age: Optional[int] = None
    education: Optional[str] = None
    gender: Optional[str] = None
    
    match_score: MatchScore
    interview_summary: str = ""
    highlights: List[str] = []


class ApplicantMatchResult(BaseModel):
    """기업이 보는 매칭 결과"""
    company_id: int
    company_name: str
    job_id: int
    
    applicants: List[ApplicantMatch] = []
    total_applicants: int = 0


class CalculateMatchRequest(BaseModel):
    """매칭 계산 요청"""
    interview_id: int