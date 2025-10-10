"""
LangGraph State 정의
모든 노드가 공유하는 상태
"""

import operator
from typing import Dict, List, Optional, TypedDict, Any, Annotated
from datetime import datetime


class EvaluationState(TypedDict):
    """평가 프로세스 전체 상태"""  
    
    # 입력 데이터
    interview_id: int
    applicant_id: int
    job_id: int
    transcript_s3_url: str
    transcript: Dict
    transcript_content: Dict
    resume_data: Optional[Dict]
    openai_client: Any
    prompts: Dict[str, str]

    # 가중치 (10개 역량)
    competency_weights: Dict[str, float]

    # JD에서 추출한 직무 역량 (동적)
    job_competencies: Optional[List[Dict[str, Any]]]
    session_id: Optional[int]
    
    
    
    # Stage 1: 10개 Agent 평가 결과
    # Common Competencies (5개)
    achievement_motivation_result: Optional[Dict]
    growth_potential_result: Optional[Dict]
    interpersonal_skill_result: Optional[Dict]
    organizational_fit_result: Optional[Dict]
    problem_solving_result: Optional[Dict]
    
    # Job Competencies (5개)
    customer_journey_marketing_result: Optional[Dict]
    md_data_analysis_result: Optional[Dict]
    seasonal_strategy_kpi_result: Optional[Dict]
    stakeholder_collaboration_result: Optional[Dict]
    value_chain_optimization_result: Optional[Dict]
    
    
    
    # Stage 2: Aggregator 중간 결과
    # Sub-step 2.1: Resume Verification 결과
    segment_evaluations_with_resume: List[Dict] 
    
    # Sub-step 2.2: Confidence V2 계산 완료 여부
    confidence_v2_calculated: bool 
    
    # Sub-step 2.3: Segment Overlap 조정 결과
    segment_overlap_adjustments: List[Dict] 
    
    # Sub-step 2.4: Cross-Competency 검증 플래그
    cross_competency_flags: List[Dict]
    
    # Aggregator 최종 출력
    aggregated_competencies: Dict[str, Dict] 
    
    
    
    # Stage 2 (구) - 문제 탐지 (Deprecated, 통합됨) 
    # evidence_conflicts: List[Dict]
    low_confidence_list: List[Dict] 
    requires_collaboration: bool
    
    
    
    # Stage 3 (구) - 협업 결과 (필요 시 유지)
    collaboration_results: List[Dict]
    collaboration_count: int
    
    
    
    # Stage 3: Final Report
    final_score: Optional[float] 
    avg_confidence: Optional[float] 
    final_reliability: Optional[str] 
    reliability_note: Optional[str]
    
    final_result: Optional[Dict]
    presentation_result: Optional[Dict]
    
    # 메타 정보  
    started_at: datetime
    completed_at: Optional[datetime]
    errors: Annotated[List[str], operator.add] 
    execution_logs: Annotated[List[Dict[str, Any]], operator.add]  