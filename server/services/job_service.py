# server/services/job_service.py
"""
Job 처리 서비스 - JD PDF 업로드 및 벡터화
"""
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field

from models.job import Job, JobChunk
try:
    from models.company import Company
except ImportError:
    Company = None  # fallback if company model doesn't exist
from services.s3_service import S3Service
from services.embedding_service import EmbeddingService
from ai.parsers.jd_parser import JDParser
from ai.utils.llm_client import LLMClient


# ==================== Pydantic Models for Persona Generation ====================

class InterviewerPersona(BaseModel):
    """면접관 페르소나 정보"""
    company_name: str = Field(description="회사명 (JD에서 추출)")
    job_title: str = Field(description="직무명 (JD에서 추출)")
    interviewer_identity: str = Field(description="면접관 정체성 (예: 'OO부문 15년 차 시니어 채용 담당자')")
    interviewer_name: str = Field(description="면접관 이름 (예: '김OO 책임')")
    interviewer_tone: List[str] = Field(description="면접관 말투/스타일 (3-5개)", min_length=3)
    initial_questions: List[str] = Field(description="면접 시작 질문 (3-5개)", min_length=3, max_length=5)
    system_instruction: str = Field(description="면접관 시스템 지시사항")


class JobService:
    """
    채용 공고 처리 서비스

    전체 플로우:
    1. PDF 업로드 → S3 저장
    2. PDF 다운로드 → 텍스트 추출
    3. 텍스트 청크 분할
    4. 청크별 임베딩 생성 (Bedrock Titan)
    5. DB 저장 (jobs, job_chunks)
    """

    def __init__(self):
        self.s3_service = S3Service()
        self.embedding_service = EmbeddingService()
        self.jd_parser = JDParser(chunk_size=1000, chunk_overlap=200)
        # self.prompt_builder = ParsingPromptBuilder()  # 임시 비활성화
        self.llm_client = LLMClient()

    async def process_jd_pdf(
        self,
        db: Session,
        pdf_content: bytes,
        file_name: str,
        company_id: int,
        title: str,
        company_url: str = None
    ) -> Job:
        """
        JD PDF 전체 처리 플로우

        Args:
            db: 데이터베이스 세션
            pdf_content: PDF 파일 바이너리
            file_name: 파일명
            company_id: 회사 ID
            title: 채용 공고 제목

        Returns:
            Job: 생성된 Job 객체

        Raises:
            Exception: 처리 실패 시
        """
        try:
            print(f"\n{'='*60}")
            print(f"Starting JD PDF processing: {file_name}")
            print(f"{'='*60}")

            # 1. S3에 PDF 업로드
            print("\n[Step 1/5] Uploading PDF to S3...")
            s3_key = self.s3_service.upload_file(
                file_content=pdf_content,
                file_name=file_name,
                folder="jd_pdfs"
            )

            # 2. PDF 파싱 및 청크 분할
            print("\n[Step 2/5] Parsing PDF and creating chunks...")
            parsed_result = self.jd_parser.parse_and_chunk(
                pdf_content=pdf_content,
                metadata={
                    "company_id": company_id,
                    "s3_key": s3_key,
                    "file_name": file_name
                }
            )

            full_text = parsed_result["full_text"]
            chunks = parsed_result["chunks"]

            print(f"  - Total text length: {len(full_text)} characters")
            print(f"  - Number of chunks: {len(chunks)}")

            # 2-1. JD에서 회사 가중치 및 역량 추출
            print("\n[Step 2-1/6] Extracting competencies from JD...")
            weights_data = None

            try:
                weights_data = await self._extract_company_weights(full_text)
                print(f"  ✓ Competencies extracted: {len(weights_data.get('competencies', []))} competencies")
            except Exception as e:
                print(f"  ✗ Failed to extract company weights: {e}")
                print(f"  → Continuing without weight extraction...")
                weights_data = None

            if weights_data and "weights" in weights_data and Company:
                # Company 테이블 업데이트 (Company 모델이 있는 경우에만)
                try:
                    company = db.query(Company).filter(Company.id == company_id).first()
                    if company:
                        company.category_weights = weights_data["weights"]
                        # reasoning도 저장 (선택사항)
                        if not company.company_culture_desc and "reasoning" in weights_data:
                            company.company_culture_desc = str(weights_data.get("reasoning", {}))
                        db.flush()
                        print(f"  ✓ Company weights updated: {weights_data['weights']}")
                    else:
                        print(f"  ⚠ Company {company_id} not found, skipping weight update")
                except Exception as e:
                    print(f"  ⚠ Failed to update company weights: {e}")
            else:
                print("  ⚠ Skipping company weight update (no Company model or no weight data)")

            # 2-2. 🆕 동적 Persona 생성
            print("\n[Step 2-2/7] Generating interviewer persona from JD...")
            persona_data = None

            try:
                competencies = weights_data.get("competencies", []) if weights_data else []
                persona_data = await self._generate_persona_from_jd(full_text, competencies)
                if persona_data:
                    print(f"  ✓ Persona generated: {persona_data.company_name} - {persona_data.job_title}")
                else:
                    print(f"  ⚠ Persona generation returned None")
            except Exception as e:
                print(f"  ✗ Failed to generate persona: {e}")
                print(f"  → Continuing without persona data...")
                persona_data = None

            # 3. Job 생성
            print("\n[Step 3/7] Creating Job record...")
            job = Job(
                company_id=company_id,
                title=title,
                description=full_text,
                company_url=company_url  # 기업 URL 저장 (향후 파싱 예정)
            )

            # 역량 정보 저장 (추출 성공 시)
            if weights_data and "competencies" in weights_data:
                # Job 테이블에 역량 정보 저장
                job.dynamic_evaluation_criteria = weights_data.get("competencies", [])
                job.competency_weights = weights_data.get("weights", {})
                job.weights_reasoning = weights_data.get("reasoning", "")
                print(f"  ✓ Stored {len(weights_data['competencies'])} competencies in Job")

            # 🆕 Persona 정보 저장 (생성 성공 시)
            if persona_data:
                job.company_name = persona_data.company_name
                job.job_title = persona_data.job_title
                job.interviewer_identity = persona_data.interviewer_identity
                job.interviewer_name = persona_data.interviewer_name
                job.interviewer_tone = persona_data.interviewer_tone
                job.initial_questions = persona_data.initial_questions
                job.system_instruction = persona_data.system_instruction
                print(f"  ✓ Stored persona data in Job")

            db.add(job)
            db.flush()  # ID 생성을 위해 flush
            print(f"  - Job created with ID: {job.id}")

            # 4. 청크별 임베딩 생성
            print("\n[Step 4/5] Generating embeddings for chunks...")
            print(f"  - Total chunks to embed: {len(chunks)}")
            chunk_texts = [chunk["chunk_text"] for chunk in chunks]

            try:
                embeddings = self.embedding_service.generate_embeddings_batch(
                    texts=chunk_texts,
                    batch_size=5  # API 제한 고려
                )
            except Exception as e:
                print(f"  ✗ Embedding generation failed: {e}")
                raise Exception(f"Failed to generate embeddings: {str(e)}")

            # 5. JobChunk 저장
            print("\n[Step 5/5] Saving chunks to database...")
            created_chunks = []

            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                if embedding is None:
                    print(f"  ⚠ Skipping chunk {i} (embedding failed)")
                    continue

                job_chunk = JobChunk(
                    job_id=job.id,
                    chunk_text=chunk["chunk_text"],
                    embedding=embedding,
                    chunk_index=chunk["chunk_index"]
                )
                db.add(job_chunk)
                created_chunks.append(job_chunk)

            # 커밋
            db.commit()
            db.refresh(job)

            print(f"\n{'='*60}")
            print(f"✓ JD Processing completed successfully!")
            print(f"  - Job ID: {job.id}")
            print(f"  - Chunks saved: {len(created_chunks)}")
            print(f"  - S3 Key: {s3_key}")
            print(f"{'='*60}\n")

            return job

        except Exception as e:
            db.rollback()
            print(f"\n✗ JD Processing failed: {e}")
            raise Exception(f"Failed to process JD PDF: {str(e)}")

    def get_job_with_chunks(
        self,
        db: Session,
        job_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Job과 모든 청크 조회

        Args:
            db: 데이터베이스 세션
            job_id: Job ID

        Returns:
            Dict: Job 정보 + 청크 리스트
        """
        job = db.query(Job).filter(Job.id == job_id).first()

        if not job:
            return None

        chunks = db.query(JobChunk).filter(
            JobChunk.job_id == job_id
        ).order_by(JobChunk.chunk_index).all()

        return {
            "job_id": job.id,
            "company_id": job.company_id,
            "title": job.title,
            "description": job.description,
            "created_at": job.created_at,
            "chunks": [
                {
                    "chunk_id": chunk.id,
                    "chunk_text": chunk.chunk_text,
                    "chunk_index": chunk.chunk_index,
                    "has_embedding": chunk.embedding is not None
                }
                for chunk in chunks
            ],
            "total_chunks": len(chunks)
        }

    def search_similar_chunks(
        self,
        db: Session,
        query_text: str,
        top_k: int = 5,
        job_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        유사한 청크 검색 (벡터 유사도)

        Args:
            db: 데이터베이스 세션
            query_text: 검색 쿼리
            top_k: 반환할 상위 결과 개수
            job_id: 특정 Job으로 제한 (선택)

        Returns:
            List[Dict]: 유사한 청크 리스트
        """
        from pgvector.sqlalchemy import cosine_distance

        # 쿼리 임베딩 생성
        print(f"Generating embedding for query: {query_text[:100]}...")
        query_embedding = self.embedding_service.generate_embedding(query_text)

        # 벡터 검색
        query = db.query(
            JobChunk.id,
            JobChunk.job_id,
            JobChunk.chunk_text,
            JobChunk.chunk_index,
            cosine_distance(JobChunk.embedding, query_embedding).label("distance")
        )

        if job_id:
            query = query.filter(JobChunk.job_id == job_id)

        results = query.order_by("distance").limit(top_k).all()

        return [
            {
                "chunk_id": r.id,
                "job_id": r.job_id,
                "chunk_text": r.chunk_text,
                "chunk_index": r.chunk_index,
                "similarity": 1 - r.distance  # 거리를 유사도로 변환
            }
            for r in results
        ]

    def delete_job(
        self,
        db: Session,
        job_id: int
    ) -> bool:
        """
        Job 및 관련 청크 삭제

        Args:
            db: 데이터베이스 세션
            job_id: Job ID

        Returns:
            bool: 삭제 성공 여부
        """
        try:
            job = db.query(Job).filter(Job.id == job_id).first()

            if not job:
                return False

            # Job 삭제 (CASCADE로 청크도 자동 삭제)
            db.delete(job)
            db.commit()

            print(f"✓ Job {job_id} deleted successfully")
            return True

        except Exception as e:
            db.rollback()
            print(f"✗ Failed to delete job {job_id}: {e}")
            return False

    async def _extract_company_weights(self, jd_text: str) -> Optional[Dict[str, Any]]:
        """
        JD 텍스트에서 회사의 핵심 역량 가중치 추출

        멀티에이전트 평가 시스템과 연동을 위해 5개 고정 컨설팅 역량으로 매핑

        Args:
            jd_text: JD 전체 텍스트

        Returns:
            Dict: {
                "weights": {...},
                "reasoning": {...},
                "competencies": [5개 고정 역량]
            }
        """
        print(f"[_extract_company_weights] Starting competency extraction...")
        try:
            # 5개 고정 컨설팅 역량 (멀티에이전트 평가와 동일)
            FIXED_COMPETENCIES = [
                "Strategic Planning & Analysis (전략 기획 및 분석력)",
                "Stakeholder Management (이해관계자 관리)",
                "Project & Timeline Management (프로젝트 실행 관리)",
                "Business Insight & Market Research (시장 조사 및 인사이트)",
                "Data Management & Reporting (데이터 관리 및 보고)"
            ]

            prompt = f"""
당신은 채용 공고(JD)를 분석하여 회사가 요구하는 역량을 평가하는 전문가입니다.

다음 JD를 분석하여, 아래 **5개 컨설팅 직무 역량** 각각에 대해 이 JD가 얼마나 중요하게 요구하는지 점수(0-100)를 매기세요.

**분석할 5개 역량:**
1. Strategic Planning & Analysis (전략 기획 및 분석력)
   - 전략 수립, 문제 분석, 의사결정, 논리적 사고
2. Stakeholder Management (이해관계자 관리)
   - 커뮤니케이션, 협업, 설득력, 관계 구축
3. Project & Timeline Management (프로젝트 실행 관리)
   - 일정 관리, 업무 조율, 실행력, 리소스 관리
4. Business Insight & Market Research (시장 조사 및 인사이트)
   - 시장 분석, 트렌드 파악, 고객 이해, 인사이트 도출
5. Data Management & Reporting (데이터 관리 및 보고)
   - 데이터 분석, 보고서 작성, 지표 관리, 결과 정리

**JD:**
{jd_text[:3000]}

**응답 형식 (JSON):**
{{
  "competencies": [
    {{
      "name": "Strategic Planning & Analysis (전략 기획 및 분석력)",
      "category": "technical",
      "score": 85,
      "description": "이 JD에서 이 역량이 중요한 이유"
    }},
    {{
      "name": "Stakeholder Management (이해관계자 관리)",
      "category": "cultural",
      "score": 75,
      "description": "..."
    }},
    ... (5개 모두)
  ],
  "category_weights": {{
    "technical": 0.35,
    "cultural": 0.30,
    "experience": 0.20,
    "leadership": 0.15
  }},
  "reasoning": "JD 전체 분석 요약"
}}

**중요:**
- 반드시 위 5개 역량 모두에 대해 점수를 매기세요
- 유효한 JSON만 반환하세요
- category_weights의 합은 1.0이어야 합니다
"""

            print(f"[_extract_company_weights] Calling LLM with prompt length: {len(prompt)} chars")
            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.3
            )
            print(f"[_extract_company_weights] ✓ LLM response received, length: {len(str(response))} chars")

            # JSON 파싱
            import json
            import re

            print(f"[_extract_company_weights] Parsing LLM response...")

            # response가 dict인 경우 (generate 메서드가 dict를 반환할 때)
            if isinstance(response, dict):
                if 'text' in response:
                    response_text = response['text']
                else:
                    print(f"[_extract_company_weights] ⚠ Unexpected response format: {response}")
                    response_text = str(response)
            else:
                response_text = str(response)

            # JSON 추출 (```json ... ``` 형태인 경우 처리)
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # JSON 블록이 없으면 전체를 JSON으로 파싱 시도
                json_str = response_text

            try:
                result = json.loads(json_str)
                print(f"[_extract_company_weights] ✓ JSON parsed successfully")
            except json.JSONDecodeError as e:
                print(f"[_extract_company_weights] ✗ JSON parsing failed: {e}")
                print(f"  Response preview: {response_text[:500]}...")
                raise

            # 5개 역량이 모두 있는지 검증
            competencies = result.get("competencies", [])
            if len(competencies) != 5:
                print(f"[_extract_company_weights] ⚠ Warning: Expected 5 competencies, got {len(competencies)}")
                # 부족한 역량은 기본값으로 채우기
                existing_names = {c.get("name", "") for c in competencies}
                for fixed_comp in FIXED_COMPETENCIES:
                    if fixed_comp not in existing_names:
                        competencies.append({
                            "name": fixed_comp,
                            "category": "technical",
                            "score": 50,
                            "description": "분석되지 않음"
                        })

            final_result = {
                "weights": result.get("category_weights", {}),
                "competencies": competencies[:5],  # 최대 5개만
                "reasoning": result.get("reasoning", "")
            }
            print(f"[_extract_company_weights] ✓ Extraction completed: {len(final_result['competencies'])} competencies")
            return final_result

        except Exception as e:
            print(f"[_extract_company_weights] ✗ Failed to extract company weights: {e}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            return None

    async def _generate_persona_from_jd(
        self,
        jd_text: str,
        competencies: Optional[List[Dict]] = None
    ) -> Optional[InterviewerPersona]:
        """
        JD 텍스트에서 면접관 페르소나를 동적으로 생성

        Args:
            jd_text: JD 전체 텍스트
            competencies: 추출된 직무 역량 리스트 (선택)

        Returns:
            InterviewerPersona: 면접관 페르소나 정보 또는 None
        """
        print(f"[_generate_persona_from_jd] Starting persona generation...")

        try:
            # 역량 정보 포맷팅
            competencies_text = ""
            if competencies:
                competencies_text = "\n".join([
                    f"- {comp.get('name', 'Unknown')}: {comp.get('description', '')}"
                    for comp in competencies[:5]
                ])
            else:
                competencies_text = "(역량 정보 없음)"

            prompt = f"""
당신은 채용 공고(JD)를 분석하여 AI 면접관 페르소나를 생성하는 전문가입니다.

다음 JD를 분석하여 면접관 페르소나를 생성하세요.

**JD:**
{jd_text[:3000]}

**추출된 핵심 역량:**
{competencies_text}

**요구사항:**
1. **company_name**: JD에서 회사명을 추출하세요. 없으면 "미상" 또는 "기업"으로 표기
2. **job_title**: 구체적인 직무명 (예: "백엔드 엔지니어", "상품기획(MD)")
3. **interviewer_identity**: 면접관의 정체성 (예: "OO부문 15년 차 시니어 채용 담당자", "기술팀 리드")
4. **interviewer_name**: 가상의 면접관 이름 (예: "김OO 책임", "이OO 매니저")
5. **interviewer_tone**: 면접관의 말투와 스타일 (3-5개 항목)
   - 예: ["전문적이고 정중함", "논리적 근거를 중시함", "구체적인 예시를 재질문함"]
6. **initial_questions**: 면접 시작 질문 3-5개
   - 회사와 직무에 특화된 질문
   - 지원자의 경험과 동기를 파악하는 질문
   - 해당 직무의 핵심 역량을 평가하는 질문
7. **system_instruction**: 면접관 시스템 지시사항 (1-2문장)
   - 예: "당신은 면접관입니다. 지원자의 경험이 우리 회사의 JD와 얼마나 일치하는지 검증하세요."

**중요:**
- 회사와 직무에 맞는 구체적이고 현실적인 페르소나를 생성하세요
- 면접 질문은 일반적인 질문이 아닌, 해당 직무와 회사에 특화된 질문이어야 합니다
- 한국어로 작성하세요
"""

            print(f"[_generate_persona_from_jd] Calling OpenAI with structured output...")

            # OpenAI API 직접 호출 (Pydantic Structured Output)
            from openai import AsyncOpenAI
            from core.config import OPENAI_API_KEY

            client = AsyncOpenAI(api_key=OPENAI_API_KEY)

            response = await client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in creating interviewer personas from job descriptions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.4,
                max_tokens=2000,
                response_format=InterviewerPersona
            )

            # Pydantic 모델로 자동 파싱됨
            persona = response.choices[0].message.parsed

            print(f"[_generate_persona_from_jd] ✓ Persona generated successfully")
            print(f"  - Company: {persona.company_name}")
            print(f"  - Job Title: {persona.job_title}")
            print(f"  - Initial Questions: {len(persona.initial_questions)} questions")

            return persona

        except Exception as e:
            print(f"[_generate_persona_from_jd] ✗ Failed to generate persona: {e}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            return None
