# server/services/job_service.py
"""
Job 처리 서비스 - JD PDF 업로드 및 벡터화
"""
import logging
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
from ai.agents.rag_agent import RAGAgent

logger = logging.getLogger(__name__)


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
        self.llm_client = LLMClient()
        self.rag_agent = RAGAgent(llm_client=self.llm_client)
        logger.info("JobService initialized with RAG support")

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
        JD PDF 전체 처리 플로우 (RAG 기반)

        전체 플로우:
        1. S3 업로드
        2. PDF 파싱 및 청킹
        3. RAG 역량 추출 (required_skills, competency_weights, evaluation_criteria)
        4. 페르소나 생성 (RAG 결과 기반)
        5. 임베딩 생성
        6. DB 저장

        Args:
            db: 데이터베이스 세션
            pdf_content: PDF 파일 바이너리
            file_name: 파일명
            company_id: 회사 ID
            title: 채용 공고 제목
            company_url: 회사 URL (선택)

        Returns:
            Job: 생성된 Job 객체

        Raises:
            ValueError: 입력 검증 실패
            RuntimeError: 처리 중 오류 발생
        """
        # Input validation
        if not pdf_content or len(pdf_content) == 0:
            logger.error("Empty PDF content provided")
            raise ValueError("PDF content cannot be empty")

        if not file_name or not file_name.strip():
            logger.error("Invalid file name provided")
            raise ValueError("File name cannot be empty")

        logger.info(f"Starting JD PDF processing: file={file_name}, company_id={company_id}")

        try:
            # 1. S3에 PDF 업로드
            logger.info("[Step 1/6] Uploading PDF to S3...")
            try:
                s3_key = self.s3_service.upload_file(
                    file_content=pdf_content,
                    file_name=file_name,
                    folder="jd_pdfs"
                )
                logger.info(f"PDF uploaded to S3: {s3_key}")
            except Exception as e:
                logger.error(f"S3 upload failed: {str(e)}", exc_info=True)
                raise RuntimeError(f"Failed to upload PDF to S3: {str(e)}")

            # 2. PDF 파싱 및 청크 분할
            logger.info("[Step 2/6] Parsing PDF and creating chunks...")
            try:
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

                logger.info(f"PDF parsed successfully: {len(full_text)} chars, {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"PDF parsing failed: {str(e)}", exc_info=True)
                raise RuntimeError(f"Failed to parse PDF: {str(e)}")

            # 3. RAG 기반 역량 추출
            logger.info("[Step 3/6] Extracting competencies via RAG...")
            rag_data = None

            try:
                rag_data = await self.rag_agent.parse_jd(
                    job_description=full_text,
                    job_title=title
                )
                logger.info(
                    f"RAG extraction completed: "
                    f"{len(rag_data.get('required_skills', []))} required skills, "
                    f"{len(rag_data.get('dynamic_evaluation_criteria', []))} evaluation criteria"
                )
            except ValueError as e:
                logger.warning(f"RAG validation failed: {str(e)}")
                logger.warning("Continuing without RAG data...")
                rag_data = None
            except RuntimeError as e:
                logger.warning(f"RAG parsing failed: {str(e)}")
                logger.warning("Continuing without RAG data...")
                rag_data = None
            except Exception as e:
                logger.error(f"Unexpected RAG error: {str(e)}", exc_info=True)
                logger.warning("Continuing without RAG data...")
                rag_data = None

            # 4. 페르소나 생성 (RAG 결과 기반)
            logger.info("[Step 4/6] Generating interviewer persona from RAG results...")
            persona_data = None

            try:
                persona_data = await self._generate_persona_from_jd(
                    jd_text=full_text,
                    rag_results=rag_data
                )
                if persona_data:
                    logger.info(
                        f"Persona generated: {persona_data.company_name} - {persona_data.job_title}"
                    )
                else:
                    logger.warning("Persona generation returned None")
            except Exception as e:
                logger.error(f"Persona generation failed: {str(e)}", exc_info=True)
                logger.warning("Continuing without persona data...")
                persona_data = None

            # 5. Job 생성
            logger.info("[Step 5/6] Creating Job record with RAG and persona data...")
            job = Job(
                company_id=company_id,
                title=title,
                description=full_text,
                company_url=company_url
            )

            # RAG 결과 저장
            if rag_data:
                job.dynamic_evaluation_criteria = rag_data.get("dynamic_evaluation_criteria", [])
                job.competency_weights = rag_data.get("competency_weights", {})
                job.required_skills = rag_data.get("required_skills", [])
                job.preferred_skills = rag_data.get("preferred_skills", [])
                job.domain_requirements = rag_data.get("domain_requirements", [])
                job.position_type = rag_data.get("position_type", "unknown")
                job.seniority_level = rag_data.get("seniority_level", "mid")
                job.main_responsibilities = rag_data.get("main_responsibilities", [])

                logger.info(
                    f"Stored RAG data: {len(rag_data.get('dynamic_evaluation_criteria', []))} criteria, "
                    f"{len(rag_data.get('required_skills', []))} required skills"
                )

            # Persona 저장
            if persona_data:
                job.company_name = persona_data.company_name
                job.job_title = persona_data.job_title
                job.interviewer_identity = persona_data.interviewer_identity
                job.interviewer_name = persona_data.interviewer_name
                job.interviewer_tone = persona_data.interviewer_tone
                job.initial_questions = persona_data.initial_questions
                job.system_instruction = persona_data.system_instruction
                logger.info("Stored persona data in Job")

            db.add(job)
            db.flush()  # ID 생성
            logger.info(f"Job created with ID: {job.id}")

            # 6. 청크별 임베딩 생성 및 저장
            logger.info(f"[Step 6/6] Generating embeddings for {len(chunks)} chunks...")
            chunk_texts = [chunk["chunk_text"] for chunk in chunks]

            try:
                embeddings = self.embedding_service.generate_embeddings_batch(
                    texts=chunk_texts,
                    batch_size=5  # API 제한 고려
                )
                logger.info(f"Embeddings generated successfully for {len(embeddings)} chunks")
            except Exception as e:
                logger.error(f"Embedding generation failed: {str(e)}", exc_info=True)
                raise RuntimeError(f"Failed to generate embeddings: {str(e)}")

            # JobChunk 저장
            created_chunks = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                if embedding is None:
                    logger.warning(f"Skipping chunk {i} (embedding is None)")
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

            logger.info(
                f"JD processing completed successfully: "
                f"job_id={job.id}, chunks={len(created_chunks)}, s3_key={s3_key}"
            )

            return job

        except (ValueError, RuntimeError):
            db.rollback()
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"Unexpected error in JD processing: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to process JD PDF: {str(e)}")

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
        logger.debug(f"Generating embedding for query preview: {query_text[:100]}...")
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

            logger.info(f"Job {job_id} deleted successfully")
            return True

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete job {job_id}: {e}", exc_info=True)
            return False

    # ==================== DEPRECATED: _extract_company_weights ====================
    # This method has been replaced by RAGAgent.parse_jd()
    # RAG provides more comprehensive JD analysis with required_skills,
    # dynamic_evaluation_criteria, competency_weights, and more.
    # ==============================================================================

    async def _generate_persona_from_jd(
        self,
        jd_text: str,
        rag_results: Optional[Dict[str, Any]] = None
    ) -> Optional[InterviewerPersona]:
        """
        JD 텍스트와 RAG 결과에서 면접관 페르소나를 동적으로 생성

        Args:
            jd_text: JD 전체 텍스트
            rag_results: RAG 추출 결과 (required_skills, dynamic_evaluation_criteria, competency_weights 등)

        Returns:
            InterviewerPersona: 면접관 페르소나 정보 또는 None

        Raises:
            RuntimeError: 페르소나 생성 실패
        """
        logger.info("Starting persona generation with RAG results...")

        try:
            # RAG 결과 포맷팅
            rag_context = ""
            if rag_results:
                # 필수 스킬
                required_skills = rag_results.get("required_skills", [])
                if required_skills:
                    rag_context += f"\n**필수 기술 스킬:**\n{', '.join(required_skills[:10])}\n"

                # 우대 스킬
                preferred_skills = rag_results.get("preferred_skills", [])
                if preferred_skills:
                    rag_context += f"\n**우대 기술 스킬:**\n{', '.join(preferred_skills[:5])}\n"

                # 평가 기준
                criteria = rag_results.get("dynamic_evaluation_criteria", [])
                if criteria:
                    rag_context += f"\n**핵심 평가 기준:**\n"
                    for i, criterion in enumerate(criteria[:5], 1):
                        rag_context += f"{i}. {criterion}\n"

                # 직무 정보
                position_type = rag_results.get("position_type", "")
                seniority_level = rag_results.get("seniority_level", "")
                if position_type or seniority_level:
                    rag_context += f"\n**직무 정보:** {position_type} ({seniority_level} 레벨)\n"

                # 주요 업무
                responsibilities = rag_results.get("main_responsibilities", [])
                if responsibilities:
                    rag_context += f"\n**주요 업무:**\n"
                    for i, resp in enumerate(responsibilities[:5], 1):
                        rag_context += f"{i}. {resp}\n"

                logger.debug(f"RAG context formatted: {len(rag_context)} chars")
            else:
                rag_context = "(RAG 분석 결과 없음)"
                logger.warning("No RAG results provided for persona generation")

            prompt = f"""
당신은 채용 공고(JD)를 분석하여 AI 면접관 페르소나를 생성하는 전문가입니다.

다음 JD와 RAG 분석 결과를 바탕으로 면접관 페르소나를 생성하세요.

**JD 텍스트 (요약):**
{jd_text[:2000]}

**RAG 분석 결과:**
{rag_context}

**생성할 페르소나 정보:**
1. **company_name**: JD에서 회사명 추출. 없으면 "미상" 또는 "기업"
2. **job_title**: 구체적인 직무명 (예: "백엔드 엔지니어", "데이터 분석가")
3. **interviewer_identity**: 면접관 정체성 (예: "기술팀 15년 차 시니어 채용 담당자")
4. **interviewer_name**: 가상 면접관 이름 (예: "김OO 책임")
5. **interviewer_tone**: 면접 스타일 3-5개 (예: ["전문적이고 정중함", "논리적 근거 중시"])
6. **initial_questions**: RAG 분석 결과의 핵심 평가 기준과 필수 스킬을 반영한 면접 시작 질문 3-5개
   - 일반적 질문이 아닌, 직무와 기술 스택에 특화된 질문
   - 필수 기술과 평가 기준을 검증하는 질문
7. **system_instruction**: 면접관 시스템 지시사항 1-2문장

**중요:**
- RAG 분석 결과를 적극 반영하여 구체적이고 현실적인 페르소나 생성
- 면접 질문은 필수 기술과 평가 기준에 기반해야 함
- 한국어로 작성
"""

            logger.debug(f"Calling OpenAI with prompt length: {len(prompt)} chars")

            # OpenAI API 직접 호출 (Pydantic Structured Output)
            from openai import AsyncOpenAI
            from core.config import OPENAI_API_KEY

            client = AsyncOpenAI(api_key=OPENAI_API_KEY)

            response = await client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in creating interviewer personas from job descriptions and RAG analysis results."
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

            logger.info(
                f"Persona generated successfully: "
                f"company={persona.company_name}, "
                f"job={persona.job_title}, "
                f"questions={len(persona.initial_questions)}"
            )

            return persona

        except Exception as e:
            logger.error(f"Failed to generate persona: {str(e)}", exc_info=True)
            raise RuntimeError(f"Persona generation failed: {str(e)}")
