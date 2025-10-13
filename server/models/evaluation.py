# server/models/evaluation.py
from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey, Index, JSON, Text, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from db.database import Base


class Evaluation(Base):
    """
    Evaluation 테이블 - Multiagent 평가 결과 (1:1 매칭, 근거 저장)
    """
    __tablename__ = "evaluations"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Foreign Keys
    applicant_id = Column(Integer, ForeignKey("applicants.id", ondelete="CASCADE"), nullable=False, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    interview_id = Column(Integer, ForeignKey("interview_sessions.id", ondelete="SET NULL"), nullable=True, index=True)
    
    # ===== 핵심 점수 =====
    match_score = Column(Float, nullable=False, index=True)  # 원점수 (0-100)
    normalized_score = Column(Float, nullable=True, index=True)  # 표준화 점수 (편향 방지)
    weighted_score = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=False)
    evaluation_status = Column(String(50), nullable=False, default="completed", server_default="completed")
    

    # ===== 6개 고정 역량 점수 (JSON in RDS) =====
    job_expertise = Column(JSON, nullable=True)  # 직무 전문성
    analytical = Column(JSON, nullable=True)  # 분석력
    execution = Column(JSON, nullable=True)  # 실행력
    relationship_building = Column("relationship", JSON, nullable=True)  # 관계 구축 (고객 집착) - DB 컬럼명: relationship
    resilience = Column(JSON, nullable=True)  # 회복탄력성 (학습 태도)
    influence = Column(JSON, nullable=True)  # 영향력

    # ===== 기타 역량 점수 (JSON) =====
    problem_solving = Column(JSON, nullable=True)
    organizational_fit = Column(JSON, nullable=True)
    growth_potential = Column(JSON, nullable=True)
    interpersonal_skills = Column(JSON, nullable=True)
    achievement_motivation = Column(JSON, nullable=True)
    structured_thinking = Column(JSON, nullable=True)
    business_documentation = Column(JSON, nullable=True)
    financial_literacy = Column(JSON, nullable=True)
    industry_learning = Column(JSON, nullable=True)
    stakeholder_management = Column(JSON, nullable=True)

    # ===== 통합 점수 (JSON) =====
    job_aggregation = Column(JSON, nullable=True)
    common_aggregation = Column(JSON, nullable=True)

    # ===== Fit 분석 =====
    job_requirement_fit_score = Column(Float, nullable=True)
    fit_analysis = Column(JSON, nullable=True)
    expected_onboarding_duration = Column(Text, nullable=True)
    onboarding_support_needed = Column(JSON, nullable=True)
    key_insights = Column(JSON, nullable=True)


    # ===== 상세 결과 (JSON) =====

    competency_scores = Column(JSON, nullable=True)
    individual_evaluations = Column(JSON, nullable=True)
    aggregated_evaluation = Column(JSON, nullable=True)
    match_result = Column(JSON, nullable=True)
    validation_result = Column(JSON, nullable=True)  # 신뢰도 검증 결과

    # ===== S3 참조 (Claim Check Pattern) =====
    transcript_s3_url = Column(Text, nullable=True) # S3 transcript 경로
    agent_logs_s3_url = Column(Text, nullable=True)  # S3 execution logs 경로
    evidence_s3_url = Column(Text, nullable=True) # S3 filtered evidence 경로

    # ===== 근거 저장 (레거시) =====
    reasoning_log = Column(JSON, nullable=True)  # AI 추론 과정 로그 (레거시)

    """
    {
        "job_expertise": {
            "reasoning": "Python 3년 경험, AWS 실무 프로젝트 2개...",
            "evidence_keywords": ["Python", "AWS", "Docker"],
            "confidence": 0.85
        },
        ...
    }
    """
    
    rubric_scores = Column(JSON, nullable=True)  # 루브릭 기반 평가 점수
    """
    {
        "논리성": {"score": 90, "evidence": "..."},
        "명확성": {"score": 85, "evidence": "..."},
        "협업력": {"score": 75, "evidence": "..."}
    }
    """
    
    # ===== 정규화 정보 (편향 방지) =====
    normalization_metadata = Column(JSON, nullable=True)
    """
    {
        "job_avg": 75.3,
        "job_std": 8.2,
        "company_avg": 78.5,
        "normalized_at": "2025-11-10T12:00:00Z"
    }
    """
    
    # ===== 메타데이터 =====
    evaluation_metadata = Column(JSON, nullable=True)
    
    # ===== 타임스탬프 =====
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # ===== Relationships =====
    applicant = relationship("Applicant", backref="evaluations")
    job = relationship("Job", backref="evaluations")
    interview = relationship("InterviewSession", backref="evaluations")
    
    # ===== Indexes =====
    __table_args__ = (
        Index('ix_evaluations_job_match_score', 'job_id', 'match_score'),
        Index('ix_evaluations_job_normalized_score', 'job_id', 'normalized_score'),
        Index('ix_evaluations_applicant_created', 'applicant_id', 'created_at'),
    )
