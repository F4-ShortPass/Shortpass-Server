import logging
from typing import Dict, Any

from fastapi import APIRouter, UploadFile, File, HTTPException, Form

from ai.parsers.jd_parser import JDParser
from ai.agents.rag_agent import RAGAgent
from ai.utils.llm_client import LLMClient
from core.config import settings


router = APIRouter()
logger = logging.getLogger(__name__)

# PDF 파일 크기 제한 (10MB)
MAX_PDF_SIZE = 10 * 1024 * 1024


@router.post("/jd-parser/parse-competencies", response_model=Dict[str, Any])
async def parse_jd_competencies_endpoint(
    file: UploadFile = File(..., description="JD PDF 파일"),
    company_name: str = Form(..., min_length=1, max_length=200, description="회사 이름"),
    job_title: str = Form(..., min_length=1, max_length=200, description="직무명")
):
    """
    JD PDF 파서 (RAG 기반)

    **흐름:**
    1. PDF 파일 검증 (확장자, 크기)
    2. JDParser로 텍스트 추출
    3. RAGAgent (LLM)로 역량/가중치/필수기술 파싱
    4. 결과 + raw 텍스트 반환

    **에러 처리:**
    - 400: 잘못된 파일 형식, 크기 초과, 빈 파일
    - 500: PDF 파싱 실패, LLM 파싱 실패
    """
    logger.info(f"JD parse request: company='{company_name}', job='{job_title}', file='{file.filename}'")

    # 1. 파일 확장자 검증
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        logger.warning(f"Invalid file type: {file.filename}")
        raise HTTPException(
            status_code=400,
            detail="PDF 파일만 업로드할 수 있습니다. (.pdf 확장자 필수)"
        )

    # 2. 파일 읽기 및 크기 검증
    try:
        pdf_content = await file.read()
        file_size = len(pdf_content)

        if file_size == 0:
            logger.warning(f"Empty file uploaded: {file.filename}")
            raise HTTPException(status_code=400, detail="빈 파일입니다.")

        if file_size > MAX_PDF_SIZE:
            logger.warning(f"File too large: {file_size} bytes (max: {MAX_PDF_SIZE})")
            raise HTTPException(
                status_code=400,
                detail=f"파일 크기가 너무 큽니다. (최대: {MAX_PDF_SIZE // (1024*1024)}MB)"
            )

        logger.debug(f"File read successfully: {file_size} bytes")

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to read file '{file.filename}': {str(exc)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"파일을 읽을 수 없습니다: {str(exc)}")

    # 3. PDF → 텍스트 추출
    try:
        jd_parser = JDParser()
        raw_text = jd_parser.parse_pdf(pdf_content)

        if not raw_text or len(raw_text.strip()) < 50:
            logger.warning(f"Extracted text too short: {len(raw_text)} chars")
            raise HTTPException(
                status_code=400,
                detail="PDF에서 유효한 텍스트를 추출할 수 없습니다. 이미지 기반 PDF이거나 손상된 파일일 수 있습니다."
            )

        logger.info(f"PDF parsed successfully: {len(raw_text)} characters extracted")

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"JD PDF parsing failed for '{file.filename}': {str(exc)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF 파싱 실패: {str(exc)}")

    # 4. RAG 기반 LLM 파싱 (역량/가중치/기술 추출)
    try:
        llm_client = LLMClient()
        rag_agent = RAGAgent(llm_client=llm_client)

        logger.info(f"Starting RAG parsing for job='{job_title}'")
        parsed = await rag_agent.parse_jd(job_description=raw_text, job_title=job_title)

        logger.info(f"RAG parsing completed: {len(parsed.get('required_skills', []))} required skills, "
                   f"{len(parsed.get('dynamic_evaluation_criteria', []))} criteria")

    except ValueError as exc:
        # RAGAgent에서 발생한 검증 에러
        logger.error(f"RAG validation failed: {str(exc)}")
        raise HTTPException(status_code=400, detail=f"JD 검증 실패: {str(exc)}")
    except RuntimeError as exc:
        # RAGAgent에서 발생한 파싱 에러
        logger.error(f"RAG parsing failed: {str(exc)}")
        raise HTTPException(status_code=500, detail=f"JD RAG 파싱 실패: {str(exc)}")
    except Exception as exc:
        logger.error(f"Unexpected RAG error: {str(exc)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"예상치 못한 파싱 오류: {str(exc)}")

    # 5. 결과 반환
    result = {
        "company_name": company_name,
        "job_title": job_title,
        "raw_text": raw_text,
        "text_length": len(raw_text),
        **parsed,
    }

    logger.info(f"JD parsing completed successfully for '{company_name}' - '{job_title}'")
    return result
