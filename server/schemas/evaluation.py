from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# Schemas for Detailed Applicant View (/applicants/{applicant_id}/evaluation-details)
# 결과 화면 

class RadarChartData(BaseModel):
    labels: List[str]
    scores: List[int]

class CompetencyDetail(BaseModel):
    name: str
    score: int
    positive_feedback: Optional[str] = None
    negative_feedback: Optional[str] = None
    evidence_transcript_id: str

class DeepDiveAnalysisItem(BaseModel):
    question_topic: str
    trigger_reason: str
    initial_question: str
    candidate_initial_response: str
    follow_up_question: str
    candidate_response_summary: str
    agent_evaluation: str
    score_impact: str
    transcript_segment_id: str

class FeedbackLoop(BaseModel):
    is_reviewed: bool
    hr_comment: Optional[str] = None
    adjusted_score: Optional[int] = None

class EvaluationDetailResponse(BaseModel):
    candidate_id: str
    name: str
    job_title: str
    total_score: int
    grade: str
    ai_summary: str
    radar_chart_data: RadarChartData
    competency_details: List[CompetencyDetail]
    deep_dive_analysis: List[DeepDiveAnalysisItem]
    feedback_loop: FeedbackLoop
    interview_date: str
    priority_review: bool
    rank: int

# Schemas for Applicant List View (/jobs/{job_id}/applicants)

class ApplicantSummary(BaseModel):
    applicant_id: str
    job_id: str
    rank: int
    applicant_name: str
    track: str
    total_score: int
    strengths: str
    weaknesses: str
    ai_summary_comment: str
    status: str
    competency_scores: List[Dict[str, Any]] # e.g. [{"name": "Data Insight", "score": 95}, ...]

class ApplicantListResponse(BaseModel):
    company_name: str
    job_title: str
    total_applicants: int
    completed_evaluations: int
    average_score: float
    applicants: List[ApplicantSummary]
