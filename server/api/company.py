# server/api/company.py
"""
Company 관련 API 엔드포인트
"""
from fastapi import APIRouter, Depends, HTTPException, Form
from sqlalchemy.orm import Session
from typing import Optional, List
import logging

from db.database import get_db
from services.company_service import CompanyService
from schemas.company import (
    CompanyCreate,
    CompanyResponse,
    CompanyDetailResponse,
    CompanyUpdate,
    QuestionCreate,
    QuestionResponse
)


router = APIRouter(prefix="/companies", tags=["companies"])
logger = logging.getLogger("uvicorn")


@router.post("/", response_model=CompanyResponse)
async def create_company(
    name: str = Form(..., description="회사명"),
    company_values_text: Optional[str] = Form(None, description="회사 가치관/인재상"),
    company_culture_desc: Optional[str] = Form(None, description="조직 문화 설명"),
    blind_mode: bool = Form(False, description="블라인드 채용 여부"),
    db: Session = Depends(get_db)
):
    """
    기업 생성

    Args:
        name: 회사명
        company_values_text: 회사 가치관/인재상
        company_culture_desc: 조직 문화 설명
        blind_mode: 블라인드 채용 여부
        db: Database session

    Returns:
        CompanyResponse: 생성된 기업 정보
    """
    logger.info(f"Creating company with name: {name}")
    try:
        company_service = CompanyService()

        company = company_service.create_company(
            db=db,
            name=name,
            company_values_text=company_values_text,
            company_culture_desc=company_culture_desc,
            blind_mode=blind_mode,
        )

        return CompanyResponse.model_validate(company)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create company: {str(e)}"
        )


@router.get("/{company_id}", response_model=CompanyDetailResponse)
async def get_company(
    company_id: int,
    db: Session = Depends(get_db)
):
    """
    기업 상세 정보 조회 (채용공고 포함)

    Args:
        company_id: 기업 ID
        db: Database session

    Returns:
        CompanyDetailResponse: 기업 상세 정보
    """
    logger.info(f"Getting company with ID: {company_id}")
    company_service = CompanyService()
    company_data = company_service.get_company_with_jobs(db, company_id)

    if not company_data:
        raise HTTPException(status_code=404, detail="Company not found")

    return CompanyDetailResponse(**company_data)


@router.get("/", response_model=List[CompanyResponse])
async def get_companies(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    기업 목록 조회

    Args:
        skip: 건너뛸 개수
        limit: 조회할 개수
        db: Database session

    Returns:
        List[CompanyResponse]: 기업 목록
    """
    logger.info(f"Getting companies with skip: {skip}, limit: {limit}")
    company_service = CompanyService()
    companies = company_service.get_companies(db, skip=skip, limit=limit)

    return [CompanyResponse.model_validate(c) for c in companies]


@router.patch("/{company_id}", response_model=CompanyResponse)
async def update_company(
    company_id: int,
    company_update: CompanyUpdate,
    db: Session = Depends(get_db)
):
    """
    기업 정보 수정

    Args:
        company_id: 기업 ID
        company_update: 수정할 정보
        db: Database session

    Returns:
        CompanyResponse: 수정된 기업 정보
    """
    logger.info(f"Updating company with ID: {company_id}")
    company_service = CompanyService()

    update_data = company_update.model_dump(exclude_unset=True)
    company = company_service.update_company(
        db=db,
        company_id=company_id,
        **update_data
    )

    if not company:
        raise HTTPException(status_code=404, detail="Company not found")

    return CompanyResponse.model_validate(company)


@router.delete("/{company_id}")
async def delete_company(
    company_id: int,
    db: Session = Depends(get_db)
):
    """
    기업 삭제

    Args:
        company_id: 기업 ID
        db: Database session

    Returns:
        dict: 삭제 결과
    """
    logger.info(f"Deleting company with ID: {company_id}")
    company_service = CompanyService()
    success = company_service.delete_company(db, company_id)

    if not success:
        raise HTTPException(status_code=404, detail="Company not found")

    return {"message": f"Company {company_id} deleted successfully"}


@router.post("/{company_id}/questions", response_model=List[QuestionResponse])
async def create_company_questions(
    company_id: int,
    question_texts: List[str],
    job_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """
    기업 추가 질문 세트 생성

    Args:
        company_id: 기업 ID
        question_texts: 질문 텍스트 리스트
        job_id: 특정 Job에 연결 (선택)
        db: Database session

    Returns:
        List[QuestionResponse]: 생성된 질문 리스트
    """
    logger.info(f"Creating questions for company ID: {company_id}")
    company_service = CompanyService()

    # 기업 존재 여부 확인
    company = company_service.get_company(db, company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")

    try:
        questions = company_service.create_questions(
            db=db,
            question_texts=question_texts,
            job_id=job_id,
            question_type="custom"
        )

        return [QuestionResponse.model_validate(q) for q in questions]

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create questions: {str(e)}"
        )


@router.get("/{company_id}/questions", response_model=List[QuestionResponse])
async def get_company_questions(
    company_id: int,
    job_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """
    기업 질문 목록 조회

    Args:
        company_id: 기업 ID
        job_id: 특정 Job으로 필터링 (선택)
        db: Database session

    Returns:
        List[QuestionResponse]: 질문 목록
    """
    logger.info(f"Getting questions for company ID: {company_id}")
    company_service = CompanyService()

    # 기업 존재 여부 확인
    company = company_service.get_company(db, company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")

    questions = company_service.get_questions(db, job_id=job_id)

    return [QuestionResponse.model_validate(q) for q in questions]