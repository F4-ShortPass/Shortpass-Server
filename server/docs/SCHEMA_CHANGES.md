# 데이터베이스 스키마 변경사항 및 팀원 작업 통합

## 개요

팀원 B와의 협업을 통해 RDS PostgreSQL 스키마를 최종 확정했습니다.

## 주요 변경사항

### 1. 새로운 테이블 추가

#### Companies 테이블
회사 정보 및 채용 가중치를 관리합니다.

```sql
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,

    -- 회사 정보
    company_values_text TEXT,
    company_culture_desc TEXT,
    core_values JSONB,

    -- 가중치 설정
    category_weights JSONB,    -- 4대 카테고리 가중치
    priority_weights JSONB,    -- 세부 우선순위 가중치

    -- 설정
    blind_mode BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE
);
```

**사용 예시**:
```python
company = Company(
    name="ABC Corp",
    company_values_text="혁신, 협업, 도전",
    category_weights={
        "technical": 0.4,
        "cultural": 0.3,
        "experience": 0.2,
        "soft_skills": 0.1
    },
    blind_mode=True
)
```

#### Applicants 테이블
지원자 정보 및 이력서 파싱 결과를 저장합니다.

```sql
CREATE TABLE applicants (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(200) UNIQUE,

    -- 개인정보 (Blind 대상)
    age INTEGER,
    education VARCHAR(200),
    gender VARCHAR(20),

    -- 파싱 결과
    skills JSONB,
    total_experience_years INTEGER,
    domain_experience JSONB,
    special_experience JSONB,
    resume_parsed_data JSONB,
    portfolio_parsed_data JSONB,

    -- 파일 경로
    resume_file_path VARCHAR(500),
    portfolio_file_path VARCHAR(500),

    created_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE
);
```

**사용 예시**:
```python
applicant = Applicant(
    name="김개발",
    email="kim@example.com",
    skills=["Python", "FastAPI", "PostgreSQL"],
    total_experience_years=5,
    domain_experience=["backend", "api_development"],
    resume_file_path="s3://bucket/resumes/kim.pdf"
)
```

### 2. 기존 테이블 수정

#### Interview Sessions 테이블 변경

**변경 전**:
```sql
job_id INTEGER NOT NULL REFERENCES jobs(id)
```

**변경 후**:
```sql
applicant_id INTEGER NOT NULL REFERENCES applicants(id)
job_ids JSONB NOT NULL  -- 1~3개 회사 지원 가능
```

**이유**: 한 번의 면접으로 1~3개 회사에 동시 지원 가능하도록 변경

**사용 예시**:
```python
session = InterviewSession(
    applicant_id=123,
    job_ids=[101, 102, 103],  # 3개 회사 지원
    status=InterviewStatus.IN_PROGRESS
)
```

#### Interview Results 테이블 변경

**주요 변경사항**:
1. **공통/기업별 질문 구분 추가**
   ```python
   is_common = Column(Boolean, default=False, nullable=False)
   job_id = Column(Integer, ForeignKey("jobs.id"), nullable=True)
   ```

2. **점수 구조 변경**
   - 변경 전: `logic_score`, `relevance_score`, `technical_score`, `communication_score` 개별 컬럼
   - 변경 후: `scores` (JSONB) + `overall_score` (Float)

   ```python
   scores = {
       "python": 85,
       "system_design": 90,
       "collaboration": 88,
       "problem_solving": 87
   }
   overall_score = 87.5  # 평균값
   ```

3. **Keywords 구조 개선**
   ```python
   keywords = {
       "matched": ["Python", "FastAPI", "PostgreSQL"],
       "missing": ["Docker", "Kubernetes"]
   }
   ```

## 테이블 관계도

```
Companies (1) ─────< (N) Jobs (1) ─────< (N) Job_Chunks
                                            [embedding vector(1024)]

Applicants (1) ─────< (N) Interview_Sessions
                              │
                              │ (1)
                              │
                              ▼
                          (N) Interview_Results
                              │
                              └─> (N) Jobs (공통 질문이면 NULL)
```

## 중요 필드 설명

### Job_Chunks 테이블

| 필드 | 타입 | 설명 |
|------|------|------|
| chunk_text | TEXT | 청크 텍스트 (필수) |
| embedding | vector(1024) | Amazon Titan Embeddings V2 |
| chunk_index | INTEGER | 청크 순서 (0부터 시작) |

**벡터 검색 예시**:
```python
from pgvector.sqlalchemy import cosine_distance

results = db.query(JobChunk).order_by(
    cosine_distance(JobChunk.embedding, query_embedding)
).limit(5).all()
```

### Interview_Results 테이블

| 필드 | 타입 | 설명 |
|------|------|------|
| stt_full_text | TEXT | STT 변환된 답변 (필수) |
| scores | JSONB | 차원별 점수 |
| overall_score | FLOAT | 종합 점수 (정렬/인덱싱용) |
| keywords | JSONB | 매칭/누락 키워드 |
| strengths | JSONB | 강점 목록 |
| weaknesses | JSONB | 약점 목록 |
| ai_feedback | TEXT | AI 피드백 |

## 인덱스 전략

### 1. 벡터 인덱스 (HNSW)
```sql
CREATE INDEX idx_job_chunks_embedding_hnsw
ON job_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

### 2. 복합 인덱스
```sql
-- Interview Results
CREATE INDEX ix_interview_results_interview_question
ON interview_results(interview_id, question_id);

-- Job Chunks
CREATE INDEX ix_job_chunks_job_id_chunk_index
ON job_chunks(job_id, chunk_index);
```

### 3. 조건 검색 인덱스
```sql
CREATE INDEX ix_interview_results_is_common ON interview_results(is_common);
CREATE INDEX ix_interview_results_job_id ON interview_results(job_id);
CREATE INDEX ix_interview_results_overall_score ON interview_results(overall_score);
```

## 마이그레이션 가이드

### 1. 환경 설정

```bash
# .env 파일 설정
DATABASE_URL=postgresql://admin:password@f4-db.czwmu86ms4yl.us-east-1.rds.amazonaws.com:5432/flexdb
```

### 2. pgvector 확장 활성화

RDS PostgreSQL에 접속:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
\dx vector  -- 확인
```

### 3. 테이블 생성

```bash
cd /home/ec2-user/flex/server
source ../venv1/bin/activate
python init_db.py
```

### 4. 검증

```python
from db import check_db_connection, SessionLocal
from models import Company, Applicant, JobChunk, InterviewResult

# 연결 확인
assert check_db_connection()

# 테이블 확인
db = SessionLocal()
count = db.query(Company).count()
print(f"Companies: {count}")
```

## 팀원 작업 통합 상태

### ✅ 완료된 작업

1. **Database Layer**
   - ✅ SQLAlchemy 모델 정의 (Company, Applicant 추가)
   - ✅ Database connection pool 설정
   - ✅ pgvector 통합

2. **Schema Definition**
   - ✅ Companies 테이블 설계
   - ✅ Applicants 테이블 설계
   - ✅ Interview Sessions job_ids 변경
   - ✅ Interview Results 점수 구조 개선

3. **API Layer**
   - ✅ FastAPI WebSocket 엔드포인트 (api/interview.py)
   - ✅ Pydantic 스키마 정의 (schemas/interview.py)

4. **Service Layer**
   - ✅ InterviewEvaluationService 구현
   - ⚠️ Import 경로 수정 필요 (app.* → server.* 또는 상대 경로)

### ⚠️ 주의사항

1. **Import 경로 불일치**
   - 현재: `from app.models.interview import ...`
   - 수정 필요: `from models.interview import ...` 또는 `from server.models.interview import ...`

2. **Main App 파일 누락**
   - `main.py` 또는 `app.py` 생성 필요
   - FastAPI 앱 초기화 및 라우터 등록 필요

## 다음 단계

### 1. 팀원 B와 확인 필요
- [ ] RDS 접속 정보 공유
- [ ] .env 파일 비밀번호 설정
- [ ] pgvector 확장 활성화 권한 확인

### 2. 통합 테스트
- [ ] 각 테이블 CRUD 테스트
- [ ] 벡터 유사도 검색 테스트
- [ ] FK 관계 무결성 테스트

### 3. API 개발
- [ ] main.py 생성 및 FastAPI 앱 초기화
- [ ] 라우터 등록 및 Dependency Injection 설정
- [ ] Import 경로 통일

## 참고 자료

- [DATABASE_SETUP.md](./DATABASE_SETUP.md) - 상세 설정 가이드
- [schema.sql](./schema.sql) - SQL 스키마 참조
- [pgvector Documentation](https://github.com/pgvector/pgvector)

## 문의

스키마 관련 질문이나 변경이 필요한 경우 팀원 B와 상의해주세요.
