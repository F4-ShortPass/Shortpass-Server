# server/services/applicant_service.py
"""
Applicant service for handling applicant-related business logic
"""
import logging
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any
from datetime import datetime

from models.interview import Applicant, ResumeChunk
from services.s3_service import S3Service
from services.embedding_service import EmbeddingService
from ai.parsers.resume_parser import ResumeParser

logger = logging.getLogger(__name__)


class ApplicantService:
    """지원자 관련 비즈니스 로직 처리"""

    def __init__(self):
        self.s3_service = S3Service()
        self.embedding_service = EmbeddingService()
        self.resume_parser = ResumeParser(chunk_size=1000, chunk_overlap=200)

    def create_applicant(
        self,
        db: Session,
        name: str,
        email: str,
        gender: Optional[str] = None,
        education: Optional[str] = None,
        birthdate: Optional[str] = None,
    ) -> Applicant:
        """
        지원자 생성 또는 업데이트 (Upsert)

        이메일이 중복되면 기존 지원자 정보를 업데이트합니다.

        Args:
            db: Database session
            name: 이름
            email: 이메일
            gender: 성별
            education: 학력
            birthdate: 생년월일

        Returns:
            Applicant: 생성 또는 업데이트된 지원자
        """
        # 생년월일에서 나이 계산 (선택)
        age = None
        if birthdate:
            try:
                birth_year = int(birthdate.split('-')[0])
                current_year = datetime.now().year
                age = current_year - birth_year
            except (ValueError, IndexError, AttributeError) as e:
                logger.warning(f"Failed to parse birthdate '{birthdate}' for age calculation: {str(e)}")
                # Continue without age - not critical

        # 이메일로 기존 지원자 확인
        existing_applicant = db.query(Applicant).filter(Applicant.email == email).first()

        if existing_applicant:
            # 기존 지원자 정보 업데이트
            existing_applicant.name = name
            if gender:
                existing_applicant.gender = gender
            if education:
                existing_applicant.education = education
            if age is not None:
                existing_applicant.age = age
            existing_applicant.updated_at = datetime.utcnow()

            db.commit()
            db.refresh(existing_applicant)

            return existing_applicant
        else:
            # 새 지원자 생성
            applicant = Applicant(
                name=name,
                email=email,
                gender=gender,
                education=education,
                age=age,
            )

            db.add(applicant)
            db.commit()
            db.refresh(applicant)

            return applicant

    async def upload_portfolio(
        self,
        db: Session,
        applicant_id: int,
        file_content: bytes,
        file_name: str
    ) -> str:
        """
        포트폴리오 PDF를 S3에 업로드하고 RAG 파이프라인 처리

        전체 플로우:
        1. S3에 PDF 업로드
        2. PDF 파싱 및 텍스트 추출
        3. 텍스트 청킹
        4. 임베딩 생성 (Bedrock Titan)
        5. DB 저장 (ResumeChunk)

        Args:
            db: Database session
            applicant_id: 지원자 ID
            file_content: 파일 내용
            file_name: 파일명

        Returns:
            str: S3 파일 경로
        """
        try:
            print(f"\n{'='*60}")
            print(f"[ApplicantService] Starting portfolio processing: {file_name}")
            print(f"{'='*60}")

            applicant = db.query(Applicant).filter(Applicant.id == applicant_id).first()
            if not applicant:
                raise ValueError(f"Applicant {applicant_id} not found")

            # 1. S3에 업로드
            print("\n[Step 1/5] Uploading PDF to S3...")
            s3_key = self.s3_service.upload_file(
                file_content=file_content,
                file_name=f"applicant_{applicant_id}_{file_name}",
                folder="portfolios"
            )
            print(f"  ✓ S3 Key: {s3_key}")

            # 2. PDF 파싱 및 청킹
            print("\n[Step 2/5] Parsing PDF and creating chunks...")
            parsed_result = self.resume_parser.parse_and_chunk(
                pdf_content=file_content,
                metadata={
                    "applicant_id": applicant_id,
                    "s3_key": s3_key,
                    "file_name": file_name,
                    "source_type": "portfolio"
                }
            )

            full_text = parsed_result["full_text"]
            chunks = parsed_result["chunks"]

            print(f"  - Total text length: {len(full_text)} characters")
            print(f"  - Number of chunks: {len(chunks)}")

            # 3. Applicant 업데이트 (파일 경로 및 파싱 데이터 저장)
            print("\n[Step 3/5] Updating applicant record...")
            applicant.portfolio_file_path = s3_key
            applicant.portfolio_parsed_data = {
                "full_text_length": len(full_text),
                "total_chunks": len(chunks),
                "file_name": file_name
            }
            db.flush()

            # 4. 임베딩 생성
            print("\n[Step 4/5] Generating embeddings for chunks...")
            chunk_texts = [chunk["chunk_text"] for chunk in chunks]

            try:
                embeddings = self.embedding_service.generate_embeddings_batch(
                    texts=chunk_texts,
                    batch_size=5  # API 제한 고려
                )
            except Exception as e:
                print(f"  ✗ Embedding generation failed: {e}")
                raise Exception(f"Failed to generate embeddings: {str(e)}")

            # 5. ResumeChunk 저장
            print("\n[Step 5/5] Saving chunks to database...")
            created_chunks = []

            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                if embedding is None:
                    print(f"  ⚠ Skipping chunk {i} (embedding failed)")
                    continue

                resume_chunk = ResumeChunk(
                    applicant_id=applicant_id,
                    chunk_text=chunk["chunk_text"],
                    embedding=embedding,
                    chunk_index=chunk["chunk_index"],
                    source_type="portfolio"
                )
                db.add(resume_chunk)
                created_chunks.append(resume_chunk)

            # 커밋
            db.commit()
            db.refresh(applicant)

            print(f"\n{'='*60}")
            print(f"✓ Portfolio processing completed successfully!")
            print(f"  - Applicant ID: {applicant_id}")
            print(f"  - Chunks saved: {len(created_chunks)}")
            print(f"  - S3 Key: {s3_key}")
            print(f"{'='*60}\n")

            return s3_key

        except Exception as e:
            db.rollback()
            print(f"\n✗ Portfolio processing failed: {e}")
            raise Exception(f"Failed to process portfolio: {str(e)}")

    def get_applicant(self, db: Session, applicant_id: int) -> Optional[Applicant]:
        """
        지원자 조회

        Args:
            db: Database session
            applicant_id: 지원자 ID

        Returns:
            Optional[Applicant]: 지원자 정보
        """
        return db.query(Applicant).filter(Applicant.id == applicant_id).first()

    def get_applicants(
        self,
        db: Session,
        skip: int = 0,
        limit: int = 100
    ) -> List[Applicant]:
        """
        지원자 목록 조회

        Args:
            db: Database session
            skip: 건너뛸 개수
            limit: 조회할 개수

        Returns:
            List[Applicant]: 지원자 목록
        """
        return db.query(Applicant).offset(skip).limit(limit).all()

    def update_applicant(
        self,
        db: Session,
        applicant_id: int,
        **kwargs
    ) -> Optional[Applicant]:
        """
        지원자 정보 수정

        Args:
            db: Database session
            applicant_id: 지원자 ID
            **kwargs: 수정할 필드들

        Returns:
            Optional[Applicant]: 수정된 지원자 정보
        """
        applicant = db.query(Applicant).filter(Applicant.id == applicant_id).first()
        if not applicant:
            return None

        for key, value in kwargs.items():
            if value is not None and hasattr(applicant, key):
                setattr(applicant, key, value)

        db.commit()
        db.refresh(applicant)

        return applicant

    def delete_applicant(self, db: Session, applicant_id: int) -> bool:
        """
        지원자 삭제

        Args:
            db: Database session
            applicant_id: 지원자 ID

        Returns:
            bool: 삭제 성공 여부
        """
        applicant = db.query(Applicant).filter(Applicant.id == applicant_id).first()
        if not applicant:
            return False

        db.delete(applicant)
        db.commit()

        return True

    def search_similar_resume_chunks(
        self,
        db: Session,
        applicant_id: int,
        query_text: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        이력서/포트폴리오에서 유사한 청크 검색 (벡터 유사도)

        Args:
            db: 데이터베이스 세션
            applicant_id: 지원자 ID
            query_text: 검색 쿼리
            top_k: 반환할 상위 결과 개수

        Returns:
            List[Dict]: 유사한 청크 리스트
        """
        from pgvector.sqlalchemy import cosine_distance

        # 쿼리 임베딩 생성
        print(f"[ApplicantService] Generating embedding for query: {query_text[:100]}...")
        query_embedding = self.embedding_service.generate_embedding(query_text)

        # 벡터 검색
        query = db.query(
            ResumeChunk.id,
            ResumeChunk.chunk_text,
            ResumeChunk.chunk_index,
            ResumeChunk.source_type,
            cosine_distance(ResumeChunk.embedding, query_embedding).label("distance")
        ).filter(ResumeChunk.applicant_id == applicant_id)

        results = query.order_by("distance").limit(top_k).all()

        return [
            {
                "chunk_id": r.id,
                "chunk_text": r.chunk_text,
                "chunk_index": r.chunk_index,
                "source_type": r.source_type,
                "similarity": 1 - r.distance  # 거리를 유사도로 변환
            }
            for r in results
        ]

    def get_applicant_with_chunks(
        self,
        db: Session,
        applicant_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        지원자 정보와 모든 청크 조회

        Args:
            db: 데이터베이스 세션
            applicant_id: 지원자 ID

        Returns:
            Dict: 지원자 정보 + 청크 리스트
        """
        applicant = db.query(Applicant).filter(Applicant.id == applicant_id).first()

        if not applicant:
            return None

        chunks = db.query(ResumeChunk).filter(
            ResumeChunk.applicant_id == applicant_id
        ).order_by(ResumeChunk.source_type, ResumeChunk.chunk_index).all()

        return {
            "applicant_id": applicant.id,
            "name": applicant.name,
            "email": applicant.email,
            "resume_file_path": applicant.resume_file_path,
            "portfolio_file_path": applicant.portfolio_file_path,
            "created_at": applicant.created_at,
            "chunks": [
                {
                    "chunk_id": chunk.id,
                    "chunk_text": chunk.chunk_text,
                    "chunk_index": chunk.chunk_index,
                    "source_type": chunk.source_type,
                    "has_embedding": chunk.embedding is not None
                }
                for chunk in chunks
            ],
            "total_chunks": len(chunks)
        }
