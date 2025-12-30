# server/models/interview.py
# @@
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, Index, Enum, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from db.database import Base
import enum


class InterviewStatus(enum.Enum):
    """면접 상태"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

# ==================== Company & Applicant ====================

class Company(Base):
    """
    Company 테이블 - 회사 정보

    Attributes:
        id: 회사 ID (PK)
        name: 회사명
        company_url: 회사 소개 URL (핵심 가치 분석용)
        company_values_text: 인재상/핵심 가치 텍스트
        company_culture_desc: 조직 문화 설명
        core_values: 핵심 가치 리스트 (JSONB)
        category_weights: 4대 카테고리 가중치 (JSONB)
        priority_weights: 세부 우선순위 가중치 (JSONB)
        blind_mode: 블라인드 채용 모드
        created_at: 생성 시각
        updated_at: 수정 시각
    """
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False, index=True)
    company_url = Column(String(500), nullable=True)  # 회사 소개 URL

    # 회사 정보
    company_values_text = Column(Text, nullable=True)
    company_culture_desc = Column(Text, nullable=True)
    
    # 파싱 결과
    core_values = Column(JSONB, nullable=True)
    
    # 가중치
    category_weights = Column(JSONB, nullable=True)
    priority_weights = Column(JSONB, nullable=True)
    
    # 설정
    blind_mode = Column(Boolean, default=False, nullable=False)
    
    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class Applicant(Base):
    """
    Applicant 테이블 - 지원자 정보
    
    Attributes:
        id: 지원자 ID (PK)
        name: 이름
        email: 이메일
        age: 나이 (Blind 대상)
        education: 학력 (Blind 대상)
        gender: 성별 (Blind 대상)
        skills: 보유 기술 리스트 (JSONB)
        total_experience_years: 총 경력 연수
        domain_experience: 도메인 경험 리스트 (JSONB)
        special_experience: 특수 경험 리스트 (JSONB)
        resume_parsed_data: 이력서 파싱 결과 (JSONB)
        portfolio_parsed_data: 포트폴리오 파싱 결과 (JSONB)
        resume_file_path: 이력서 파일 경로
        portfolio_file_path: 포트폴리오 파일 경로
        created_at: 생성 시각
        updated_at: 수정 시각
    """
    __tablename__ = "applicants"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(200), nullable=True, unique=True, index=True)
    
    # 개인정보 (Blind 대상)
    age = Column(Integer, nullable=True)
    education = Column(String(200), nullable=True)
    gender = Column(String(20), nullable=True)
    
    # 이력서/포트폴리오 파싱 결과
    skills = Column(JSONB, nullable=True)
    total_experience_years = Column(Integer, default=0)
    domain_experience = Column(JSONB, nullable=True)
    special_experience = Column(JSONB, nullable=True)
    
    # 원본 파싱 데이터
    resume_parsed_data = Column(JSONB, nullable=True)
    portfolio_parsed_data = Column(JSONB, nullable=True)
    
    # 파일 경로
    resume_file_path = Column(String(500), nullable=True)
    portfolio_file_path = Column(String(500), nullable=True)
    
    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    resume_chunks = relationship("ResumeChunk", back_populates="applicant", cascade="all, delete-orphan")


class ResumeChunk(Base):
    """
    Resume Chunks 테이블 - 이력서/포트폴리오를 청크 단위로 분할하여 저장

    pgvector를 사용하여 1024차원의 임베딩 벡터를 저장합니다.

    Attributes:
        id: 청크 고유 ID (PK)
        applicant_id: 지원자 ID (FK)
        chunk_text: 청크 텍스트 내용
        embedding: 1024차원 벡터 임베딩 (Amazon Titan Text Embeddings V2)
        chunk_index: 청크 순서 (0부터 시작)
        source_type: 소스 타입 ('resume' or 'portfolio')
        created_at: 생성 시각

    Example:
        # 유사도 검색 (코사인 유사도)
        from pgvector.sqlalchemy import cosine_distance

        query_embedding = [0.1, 0.2, ...]  # 1024차원
        results = db.query(ResumeChunk).filter(
            ResumeChunk.applicant_id == applicant_id
        ).order_by(
            cosine_distance(ResumeChunk.embedding, query_embedding)
        ).limit(5).all()
    """
    __tablename__ = "resume_chunks"

    id = Column(Integer, primary_key=True, index=True)
    applicant_id = Column(Integer, ForeignKey("applicants.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(1024), nullable=True)  # Amazon Titan Text Embeddings V2 dimension
    chunk_index = Column(Integer, nullable=False, default=0)
    source_type = Column(String(20), nullable=False, default='resume')  # 'resume' or 'portfolio'
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationship to applicant
    applicant = relationship("Applicant", back_populates="resume_chunks")

    # Index for vector similarity search
    __table_args__ = (
        Index('ix_resume_chunks_applicant_id_chunk_index', 'applicant_id', 'chunk_index'),
    )


# ==================== Interview ====================

class InterviewSession(Base):
    """
    Interview Session 테이블 - 면접 세션 정보 (1:1 구조)

    Attributes:
        id: 면접 세션 ID (PK)
        applicant_id: 지원자 ID
        company_id: 회사 ID (FK) - 단일 회사
        job_id: 채용 공고 ID (FK)
        status: 면접 상태 (pending, in_progress, completed, failed)
        current_question_index: 현재 질문 인덱스
        current_persona_index: 현재 페르소나 인덱스 (순차 진행)
        started_at: 면접 시작 시각
        completed_at: 면접 완료 시각
        created_at: 생성 시각
    """
    __tablename__ = "interview_sessions"

    id = Column(Integer, primary_key=True, index=True)
    applicant_id = Column(Integer, nullable=False, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=True, index=True)
    status = Column(Enum(InterviewStatus), default=InterviewStatus.PENDING, nullable=False)
    current_question_index = Column(Integer, default=0, nullable=False)
    current_persona_index = Column(Integer, default=0, nullable=False)  # 현재 진행 중인 페르소나 순서
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    evaluation_completed = Column(Boolean, default=False, nullable=False)
    transcript_s3_url = Column(Text, nullable=True)  # S3 transcript 경로

    # Relationships
    results = relationship("InterviewResult", back_populates="session", cascade="all, delete-orphan")
    session_personas = relationship("SessionPersona", back_populates="session", cascade="all, delete-orphan")
    transcripts = relationship("SessionTranscript", back_populates="session", cascade="all, delete-orphan")
    scores = relationship("SessionScore", back_populates="session", cascade="all, delete-orphan")
    explanations = relationship("SessionExplanation", back_populates="session", cascade="all, delete-orphan")


class InterviewResult(Base):
    """
    Interview Results 테이블 - 면접 질문별 답변 및 평가 결과

    면접 진행 중 질문 1개당 1개 row가 실시간으로 저장됨.

    Attributes:
        id: 결과 ID (PK)
        interview_id: 면접 세션 ID (FK)
        question_id: 질문 ID
        question_text: 질문 내용
        question_type: 질문 유형 (technical, behavioral, situational, etc.)
        
        is_common: 공통 질문 여부 (True: 공통, False: 기업별)
        job_id: 기업별 질문일 경우 해당 job_id (공통이면 NULL)
        
        stt_full_text: STT로 변환된 전체 답변 텍스트
        
        scores: 차원별 평가 점수 (JSONB)
            예: {"python": 85, "system_design": 90, "collaboration": 88, ...}
        overall_score: 종합 점수 (평균값, 인덱싱/정렬용)
        
        keywords: 추출된 키워드 (JSONB)
            예: {"matched": ["Python", "FastAPI"], "missing": ["Docker"]}
        strengths: 강점 리스트 (JSONB array)
        weaknesses: 약점 리스트 (JSONB array)
        ai_feedback: AI 피드백 텍스트
        metadata: 추가 메타데이터 (JSONB)
        
        created_at: 생성 시각
        updated_at: 수정 시각
        
    Example:
        # 답변 평가 결과 저장
        result = InterviewResult(
            interview_id=1,
            question_id=5,
            question_text="Python의 GIL에 대해 설명해주세요",
            question_type="technical",
            is_common=False,
            job_id=101,
            stt_full_text="GIL은 Global Interpreter Lock의 약자로...",
            scores={
                "python": 90,
                "technical_depth": 85,
                "communication": 88
            },
            overall_score=87.67,
            keywords={
                "matched": ["GIL", "threading", "multiprocessing"],
                "missing": ["asyncio"]
            },
            strengths=[
                "GIL의 개념을 정확히 이해하고 있음",
                "대안 방법을 제시할 수 있음"
            ],
            weaknesses=[
                "asyncio와의 비교 설명 부족"
            ],
            ai_feedback="Python의 GIL에 대해 깊이 있게 이해하고 있으며..."
        )
    """
    __tablename__ = "interview_results"

    id = Column(Integer, primary_key=True, index=True)
    interview_id = Column(
        Integer,
        ForeignKey("interview_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    question_id = Column(Integer, nullable=True, index=True)
    question_text = Column(Text, nullable=True)
    question_type = Column(String(50), nullable=True)
    
    # 공통/기업별 구분
    is_common = Column(Boolean, default=False, nullable=False, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=True, index=True)
    
    # STT 결과
    stt_full_text = Column(Text, nullable=False)

    # 평가 점수
    scores = Column(JSONB, nullable=True)
    # {
    #   "python": 85,
    #   "system_design": 90,
    #   "collaboration": 88,
    #   "problem_solving": 87
    # }
    
    overall_score = Column(Float, nullable=True, index=True)  # 평균값 (정렬용)

    # 분석 결과 (JSONB)
    keywords = Column(JSONB, nullable=True)
    
    strengths = Column(JSONB, nullable=True)
    
    weaknesses = Column(JSONB, nullable=True)
    
    ai_feedback = Column(Text, nullable=True)
    interview_metadata = Column(JSONB, nullable=True)

    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationship
    session = relationship("InterviewSession", back_populates="results")

    __table_args__ = (
        Index('ix_interview_results_interview_question', 'interview_id', 'question_id'),
    )


class PersonaInstance(Base):
    """
    PersonaInstance 테이블 - 회사별 커스텀 페르소나 인스턴스

    Attributes:
        id: 인스턴스 ID (PK)
        company_id: 회사 ID (FK)
        persona_template_id: 페르소나 템플릿 ID (FK)
        instance_name: 인스턴스명 (예: "기술형", "논리형", "컬처핏형")
        custom_weights: 커스텀 가중치 (JSONB)
        question_tone: 질문 톤 커스텀 (Text)
        created_at: 생성 시각
    """
    __tablename__ = "persona_instances"

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False, index=True)
    persona_template_id = Column(Integer, ForeignKey("personas.id"), nullable=False, index=True)
    instance_name = Column(String(100), nullable=False)
    custom_weights = Column(JSONB, nullable=True)
    question_tone = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    company = relationship("Company")
    persona_template = relationship("PersonaDB")


class SessionPersona(Base):
    """
    SessionPersona 테이블 - 세션별 페르소나 순서 및 역할

    Attributes:
        id: ID (PK)
        session_id: 세션 ID (FK)
        persona_instance_id: 페르소나 인스턴스 ID (FK)
        order: 라운드 순서 (0부터 시작)
        role: 역할 (primary/reviewer)
        created_at: 생성 시각
    """
    __tablename__ = "session_personas"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("interview_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    persona_instance_id = Column(Integer, ForeignKey("persona_instances.id"), nullable=False, index=True)
    order = Column(Integer, nullable=False)
    role = Column(String(50), default="primary", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    session = relationship("InterviewSession", back_populates="session_personas")
    persona_instance = relationship("PersonaInstance")

    __table_args__ = (
        Index('ix_session_personas_session_order', 'session_id', 'order'),
    )


class SessionTranscript(Base):
    """
    SessionTranscript 테이블 - 세션 대화 기록

    Attributes:
        id: ID (PK)
        session_id: 세션 ID (FK)
        persona_instance_id: 페르소나 인스턴스 ID (FK, nullable)
        turn: 대화 순서
        text: 대화 내용
        embedding: 텍스트 임베딩 벡터 (1536차원, RAG용)
        meta_json: 메타데이터 (JSONB)
        created_at: 생성 시각
    """
    __tablename__ = "session_transcripts"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("interview_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    persona_instance_id = Column(Integer, ForeignKey("persona_instances.id"), nullable=True, index=True)
    turn = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=True)
    meta_json = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    session = relationship("InterviewSession", back_populates="transcripts")
    persona_instance = relationship("PersonaInstance")


class SessionScore(Base):
    """
    SessionScore 테이블 - 세션별 페르소나별 평가 점수

    Attributes:
        id: ID (PK)
        session_id: 세션 ID (FK)
        persona_instance_id: 페르소나 인스턴스 ID (FK)
        criterion_key: 평가 기준 키 (예: "technical_depth")
        score: 점수 (0-100)
        created_at: 생성 시각
    """
    __tablename__ = "session_scores"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("interview_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    persona_instance_id = Column(Integer, ForeignKey("persona_instances.id"), nullable=False, index=True)
    criterion_key = Column(String(100), nullable=False, index=True)
    score = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    session = relationship("InterviewSession", back_populates="scores")
    persona_instance = relationship("PersonaInstance")

    __table_args__ = (
        Index('ix_session_scores_session_persona_criterion', 'session_id', 'persona_instance_id', 'criterion_key'),
    )


class SessionExplanation(Base):
    """
    SessionExplanation 테이블 - 세션별 페르소나별 평가 근거

    Attributes:
        id: ID (PK)
        session_id: 세션 ID (FK)
        persona_instance_id: 페르소나 인스턴스 ID (FK)
        criterion_key: 평가 기준 키
        explanation: 설명 텍스트
        log_json: 상세 로그 (JSONB, 예: coherence:0.91, evidence_match:0.84)
        created_at: 생성 시각
    """
    __tablename__ = "session_explanations"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("interview_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    persona_instance_id = Column(Integer, ForeignKey("persona_instances.id"), nullable=False, index=True)
    criterion_key = Column(String(100), nullable=False, index=True)
    explanation = Column(Text, nullable=True)
    log_json = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    session = relationship("InterviewSession", back_populates="explanations")
    persona_instance = relationship("PersonaInstance")

    __table_args__ = (
        Index('ix_session_explanations_session_persona_criterion', 'session_id', 'persona_instance_id', 'criterion_key'),
    )

class Question(Base):
    """
    Questions 테이블 - 면접 질문 템플릿

    Attributes:
        id: 질문 ID (PK)
        job_id: 채용 공고 ID (FK, nullable)
        persona_id: 페르소나 ID (FK, nullable)
        question_type: 질문 유형 (technical, behavioral, situational, etc.)
        question_text: 질문 내용
        expected_keywords: 기대되는 키워드 (JSONB)
        evaluation_criteria: 평가 기준 (JSONB)
        difficulty_level: 난이도 (1-5)
        created_at: 생성 시각
    """
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=True, index=True)
    persona_id = Column(Integer, ForeignKey("personas.id"), nullable=True, index=True)
    question_type = Column(String(50), nullable=False, index=True)
    question_text = Column(Text, nullable=False)
    expected_keywords = Column(JSONB, nullable=True)
    evaluation_criteria = Column(JSONB, nullable=True)
    difficulty_level = Column(Integer, default=3, nullable=False)
    evaluation_dimensions = Column(JSONB, nullable=True)
    dimension_weights = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class PersonaDB(Base):
    """
    Personas 테이블 - 면접관 페르소나 정보

    Attributes:
        id: 페르소나 ID (PK)
        company_id: 회사 ID (FK)
        persona_name: 페르소나 이름
        archetype: 아키타입 (analytical, supportive, stress_tester)
        description: 페르소나 설명
        system_prompt: LLM 시스템 프롬프트
        welcome_message: 첫인사 메시지
        style_description: 스타일 설명 (UI용)
        focus_keywords: 집중 키워드 (JSONB)
        focus_areas: 집중 영역 (JSONB)
        pdf_file_path: 원본 PDF 파일 경로
        parsed_data: 파싱된 원본 데이터 (JSONB)
        created_at: 생성 시각
        updated_at: 수정 시각
    """
    __tablename__ = "personas"

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False, index=True)

    # 페르소나 기본 정보
    persona_name = Column(String(200), nullable=False)
    archetype = Column(String(50), nullable=False, index=True)  # analytical, supportive, stress_tester
    description = Column(Text, nullable=True)

    # LLM 프롬프트
    system_prompt = Column(Text, nullable=True)
    welcome_message = Column(Text, nullable=True)
    style_description = Column(Text, nullable=True)

    # 집중 영역
    focus_keywords = Column(JSONB, nullable=True)  # ["키워드1", "키워드2"]
    focus_areas = Column(JSONB, nullable=True)     # ["영역1", "영역2"]

    # 원본 데이터
    pdf_file_path = Column(String(500), nullable=True)
    parsed_data = Column(JSONB, nullable=True)  # 전체 파싱 결과

    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    company = relationship("Company", backref="personas")
