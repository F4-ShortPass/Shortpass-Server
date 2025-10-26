# server/services/company_service.py
"""
Company service for handling company-related business logic
"""
from sqlalchemy.orm import Session
from typing import Optional, List, Dict
from models.interview import Company, Question
from models.job import Job


class CompanyService:
    """기업 관련 비즈니스 로직 처리"""

    def create_company(
        self,
        db: Session,
        name: str,
        company_values_text: Optional[str] = None,
        company_culture_desc: Optional[str] = None,
        blind_mode: bool = False,
    ) -> Company:
        """
        기업 생성

        Args:
            db: Database session
            name: 회사명
            company_values_text: 회사 가치관/인재상
            company_culture_desc: 조직 문화 설명
            blind_mode: 블라인드 채용 여부

        Returns:
            Company: 생성된 기업
        """
        company = Company(
            name=name,
            company_values_text=company_values_text,
            company_culture_desc=company_culture_desc,
            blind_mode=blind_mode,
        )

        db.add(company)
        db.commit()
        db.refresh(company)

        return company

    def get_company(self, db: Session, company_id: int) -> Optional[Company]:
        """
        기업 조회

        Args:
            db: Database session
            company_id: 기업 ID

        Returns:
            Optional[Company]: 기업 정보
        """
        return db.query(Company).filter(Company.id == company_id).first()

    def get_companies(
        self,
        db: Session,
        skip: int = 0,
        limit: int = 100
    ) -> List[Company]:
        """
        기업 목록 조회

        Args:
            db: Database session
            skip: 건너뛸 개수
            limit: 조회할 개수

        Returns:
            List[Company]: 기업 목록
        """
        return db.query(Company).offset(skip).limit(limit).all()

    def update_company(
        self,
        db: Session,
        company_id: int,
        **kwargs
    ) -> Optional[Company]:
        """
        기업 정보 수정

        Args:
            db: Database session
            company_id: 기업 ID
            **kwargs: 수정할 필드들

        Returns:
            Optional[Company]: 수정된 기업 정보
        """
        company = db.query(Company).filter(Company.id == company_id).first()
        if not company:
            return None

        for key, value in kwargs.items():
            if value is not None and hasattr(company, key):
                setattr(company, key, value)

        db.commit()
        db.refresh(company)

        return company

    def delete_company(self, db: Session, company_id: int) -> bool:
        """
        기업 삭제

        Args:
            db: Database session
            company_id: 기업 ID

        Returns:
            bool: 삭제 성공 여부
        """
        company = db.query(Company).filter(Company.id == company_id).first()
        if not company:
            return False

        db.delete(company)
        db.commit()

        return True

    def get_company_with_jobs(self, db: Session, company_id: int) -> Optional[Dict]:
        """
        기업 정보와 채용공고 목록 조회

        Args:
            db: Database session
            company_id: 기업 ID

        Returns:
            Optional[Dict]: 기업 정보 + Job 목록
        """
        company = db.query(Company).filter(Company.id == company_id).first()
        if not company:
            return None

        jobs = db.query(Job).filter(Job.company_id == company_id).all()

        return {
            "id": company.id,
            "name": company.name,
            "company_values_text": company.company_values_text,
            "company_culture_desc": company.company_culture_desc,
            "core_values": company.core_values,
            "category_weights": company.category_weights,
            "priority_weights": company.priority_weights,
            "blind_mode": company.blind_mode,
            "created_at": company.created_at,
            "updated_at": company.updated_at,
            "total_jobs": len(jobs),
            "jobs": [
                {
                    "id": job.id,
                    "title": job.title,
                    "created_at": job.created_at,
                }
                for job in jobs
            ]
        }

    def create_questions(
        self,
        db: Session,
        question_texts: List[str],
        job_id: Optional[int] = None,
        question_type: str = "custom"
    ) -> List[Question]:
        """
        추가 질문 세트 생성

        Args:
            db: Database session
            question_texts: 질문 텍스트 리스트
            job_id: 특정 Job에 연결 (선택)
            question_type: 질문 타입

        Returns:
            List[Question]: 생성된 질문 리스트
        """
        questions = []
        for text in question_texts:
            question = Question(
                question_text=text,
                question_type=question_type,
                job_id=job_id,
            )
            db.add(question)
            questions.append(question)

        db.commit()
        for q in questions:
            db.refresh(q)

        return questions

    def get_questions(
        self,
        db: Session,
        job_id: Optional[int] = None
    ) -> List[Question]:
        """
        질문 목록 조회

        Args:
            db: Database session
            job_id: 특정 Job으로 필터링 (선택)

        Returns:
            List[Question]: 질문 목록
        """
        query = db.query(Question)
        if job_id is not None:
            query = query.filter(Question.job_id == job_id)

        return query.all()