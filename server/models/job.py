# server/models/job.py
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Index, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from db.database import Base


class Job(Base):
    """
    Job (채용 공고) 테이블

    Attributes:
        id: 채용 공고 고유 ID (PK)
        company_id: 회사 ID
        title: 채용 공고 제목
        description: 원본 채용 공고 전체 내용

        # RAG Agent 파싱 결과
        required_skills: 필수 기술 리스트 (JSON)
        preferred_skills: 우대 기술 리스트 (JSON)
        domain_requirements: 도메인 요구사항 (JSON)
        dynamic_evaluation_criteria: 동적 평가 기준 (JSON, 5개)
        competency_weights: 역량별 가중치 (JSON)
        position_type: 포지션 타입 (backend, frontend 등)
        seniority_level: 시니어리티 (junior, mid, senior 등)
        main_responsibilities: 주요 업무 (JSON)

        created_at: 생성 시각
        updated_at: 수정 시각
    """
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, nullable=False, index=True)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    company_url = Column(String(500), nullable=True)  # 기업 웹사이트 URL (향후 파싱용)

    # RAG Agent 파싱 결과
    required_skills = Column(JSON, nullable=True)
    preferred_skills = Column(JSON, nullable=True)
    domain_requirements = Column(JSON, nullable=True)
    dynamic_evaluation_criteria = Column(JSON, nullable=True)
    competency_weights = Column(JSON, nullable=True)
    weights_reasoning = Column(JSON, nullable=True)
    position_type = Column(String(100), nullable=True)
    seniority_level = Column(String(50), nullable=True)
    main_responsibilities = Column(JSON, nullable=True)

    # 🆕 동적 Persona 정보 (JD 기반 생성)
    company_name = Column(String(200), nullable=True)  # JD에서 추출한 회사명
    job_title = Column(String(200), nullable=True)  # JD에서 추출한 직무명
    interviewer_identity = Column(String(500), nullable=True)  # 면접관 정체성 (예: "15년 차 시니어 채용 담당자")
    interviewer_name = Column(String(100), nullable=True)  # 면접관 이름 (예: "김OO 책임")
    interviewer_tone = Column(JSON, nullable=True)  # 면접관 말투/스타일 (리스트)
    initial_questions = Column(JSON, nullable=True)  # 면접 시작 질문들 (3-5개)
    system_instruction = Column(Text, nullable=True)  # 시스템 지시사항

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationship to job_chunks
    chunks = relationship("JobChunk", back_populates="job", cascade="all, delete-orphan")


class JobChunk(Base):
    """
    Job Chunks 테이블 - 채용 공고를 청크 단위로 분할하여 저장

    pgvector를 사용하여 1024차원의 임베딩 벡터를 저장합니다.

    Attributes:
        id: 청크 고유 ID (PK)
        job_id: 채용 공고 ID (FK)
        chunk_text: 청크 텍스트 내용
        embedding: 1024차원 벡터 임베딩 (Amazon Titan Text Embeddings V2)
        chunk_index: 청크 순서 (0부터 시작)
        created_at: 생성 시각

    Example:
        # 유사도 검색 (코사인 유사도)
        from pgvector.sqlalchemy import cosine_distance

        query_embedding = [0.1, 0.2, ...]  # 1024차원
        results = db.query(JobChunk).order_by(
            cosine_distance(JobChunk.embedding, query_embedding)
        ).limit(5).all()
    """
    __tablename__ = "job_chunks"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(1024), nullable=True)  # Amazon Titan Text Embeddings V2 dimension
    chunk_index = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationship to job
    job = relationship("Job", back_populates="chunks")

    # Index for vector similarity search (HNSW or IVFFlat)
    # Note: Create this index manually after table creation for better performance:
    # CREATE INDEX ON job_chunks USING hnsw (embedding vector_cosine_ops);
    __table_args__ = (
        Index('ix_job_chunks_job_id_chunk_index', 'job_id', 'chunk_index'),
    )
