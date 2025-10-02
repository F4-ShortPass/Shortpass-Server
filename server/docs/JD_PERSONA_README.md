# JD 페르소나 생성 시스템

##  개요

이 시스템은 채용공고(JD) PDF를 업로드하면 자동으로 역량을 분석하고, 기업의 필수 질문을 바탕으로 AI 페르소나를 생성하는 백엔드 서비스입니다.

## 🏗️ 시스템 아키텍처

```
PDF Upload → Text Extraction → LLM Analysis → Competency Classification → Persona Generation → Database Storage
```

## ✨ 주요 기능

### 1. JD 분석 및 역량 분류
- **PDF 업로드**: JD PDF 파일 자동 텍스트 추출
- **공통 역량**: 6개 고정 값 (고객지향, 도전정신, 협동, 팀워크, 목표지향, 책임감)
- **직무 역량**: JD에서 LLM이 추출하는 6개 역량
- **시각화 데이터**: 육각형 그래프용 데이터 생성

### 2. 페르소나 생성
- **기업 질문**: 3개 필수 입력
- **AI 페르소나**: AWS Bedrock Claude 3 Sonnet 기반 생성
- **구조화된 출력**: JSON 형태의 페르소나 정보

### 3. 데이터 저장 및 관리
- **PostgreSQL**: 메인 데이터베이스
- **pgvector**: 벡터 검색 지원
- **S3**: PDF 파일 저장

##  새로 추가된 파일들

### Core Services
- `server/services/competency_service.py` - 역량 분석 및 페르소나 생성 서비스
- `server/services/jd_persona_service.py` - JD 페르소나 DB 관리 서비스

### API Endpoints
- `server/api/jd_persona.py` - JD 페르소나 REST API
  - `POST /api/v1/jd-persona/upload` - JD PDF 업로드 및 역량 분석
  - `POST /api/v1/jd-persona/generate-persona` - 페르소나 생성
  - `GET /api/v1/jd-persona/analysis/{job_id}` - 역량 분석 조회
  - `GET /api/v1/jd-persona/jobs/{job_id}/basic-info` - Job 정보 조회

### Database Models
- `server/models/jd_persona.py` - JD 페르소나 데이터 모델
  - `JDPersona` - 메인 페르소나 정보 테이블
  - `JDPersonaQuestion` - 추가 질문 테이블

### Testing
- `server/test_main.py` - JD 페르소나 기능 테스트 서버 (포트 8002)
- `server/bedrock_test.py` - AWS Bedrock 연결 테스트

## 🔧 설치 및 실행

### 1. 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정
```bash
# AWS Bedrock 설정
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0

# Database 설정
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
```

### 3. 데이터베이스 마이그레이션
```bash
# JDPersona 테이블 생성 필요
# alembic revision --autogenerate -m "Add JD persona tables"
# alembic upgrade head
```

### 4. 서버 실행
```bash
# 메인 서버 (포트 8000)
cd server
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 테스트 서버 (포트 8002)
cd server
python test_main.py
```

## 🧪 테스트

### Bedrock 연결 테스트
```bash
cd server
python bedrock_test.py
```

### API 테스트
```bash
# 건강 체크
curl -X GET "http://localhost:8002/health"

# CompetencyService 테스트
curl -X GET "http://localhost:8002/api/v1/test/competency-service"

# JD 업로드 테스트
curl -X POST "http://localhost:8002/api/v1/jd-persona/upload" \
  -F "pdf_file=@test.pdf" \
  -F "company_id=1" \
  -F "title=데이터 분석가"
```

##  API 응답 예시

### 역량 분석 응답
```json
{
  "job_id": 1,
  "common_competencies": ["고객지향", "도전정신", "협동", "팀워크", "목표지향", "책임감"],
  "job_competencies": ["데이터분석", "문제해결력", "창의적 사고", "기술적 이해", "리더십", "커뮤니케이션"],
  "analysis_summary": "데이터 분석가 역량 분석 결과...",
  "visualization_data": {
    "common_competencies": {
      "title": "공통 역량 (고정값)",
      "items": ["고객지향", "도전정신", "협동", "팀워크", "목표지향", "책임감"],
      "color": "#3B82F6"
    },
    "job_competencies": {
      "title": "직무 역량 (JD 추출)",
      "items": ["데이터분석", "문제해결력", "창의적 사고", "기술적 이해", "리더십", "커뮤니케이션"],
      "color": "#10B981"
    },
    "chart_config": {
      "type": "hexagon",
      "max_value": 5,
      "grid_lines": 5
    }
  }
}
```

## ⚠️ 알려진 이슈

### 1. 로컬 개발 환경
- AWS Bedrock 자격 증명이 없으면 LLM 호출 실패
- 실패 시 기본 역량으로 fallback 처리됨

### 2. 데이터베이스 테이블 충돌
- `models/jd_persona.py`에서 기존 테이블과 충돌 가능
- 해결: `from db.database import Base` 사용

### 3. Company 모델 의존성
- `Company` 모델이 없어도 동작하도록 방어 코드 포함

## 🔄 다음 단계

1. **프론트엔드 연동**: React/Vue 컴포넌트 개발
2. **AWS 배포**: ECS/EC2에 배포 설정
3. **DB 마이그레이션**: Alembic 스크립트 생성
4. **테스트 케이스**: 단위 테스트 및 통합 테스트 추가

## 📝 기술 스택

- **Backend**: FastAPI, Python 3.9+
- **Database**: PostgreSQL + pgvector
- **AI**: AWS Bedrock (Claude 3 Sonnet)
- **File Storage**: AWS S3
- **PDF Processing**: pdfplumber
- **ORM**: SQLAlchemy 2.0

---

## 🎯 구현 완료 사항

✅ **Core Services**: CompetencyService, JDPersonaService
✅ **API Endpoints**: 완전한 REST API
✅ **Database Models**: JDPersona 테이블 설계
✅ **LLM Integration**: AWS Bedrock 연동
✅ **Error Handling**: 견고한 에러 처리
✅ **Testing**: 기본 테스트 환경 구축

시스템이 **SSH 환경에서 AWS 자격 증명과 함께 배포될 준비**가 완료되었습니다!