from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import List, Dict, Any
import os
import uuid
import logging



router = APIRouter()
logger = logging.getLogger("uvicorn")

@router.post("/jd-parser/parse-competencies/mock", response_model=List[Dict[str, Any]])
async def parse_jd_competencies_endpoint_mock(
    file: UploadFile = File(..., description="JD PDF 파일"),
    company_name: str = Form("Samsung C&T", description="회사 이름") # 가상의 회사 이름, 현재는 고정
):
    """
    데모 시나리오를 위한 JD PDF에서 5개 고정된 직무 역량을 추출하는 Mock API 엔드포인트.
    실제 파싱 로직 대신 하드코딩된 5개 역량을 반환합니다.
    """
    logger.info(f"Parsing JD competencies for company: {company_name}, file: {file.filename}")
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드할 수 있습니다.")

    # 파일 저장 (실제 서비스에서는 S3 등에 저장)
    # 현재는 mock 이므로 파일을 읽지 않고 바로 mock 데이터를 반환합니다.
    # 하지만 실제 파일 업로드 테스트를 위해 임시로 저장하는 로직을 포함합니다.
    try:
        # For actual file storage, use a proper service like S3
        temp_file_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        # os.remove(temp_file_path) # 실제 사용 시에는 파일 처리 후 삭제
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 저장 중 오류 발생: {e}")

    # TODO: Call the actual service function here
    # return parse_jd_competencies_mock(file.filename, company_name)
    
    # Mock 데이터 반환 (hardcoded 5 competencies)
    mock_competencies = [
        {"id": "global_market_analysis", "name": "글로벌 시장 분석 및 전략 수립", "weight": 0.2, "description": "복잡한 글로벌 시장 동향을 분석하고 효과적인 사업 전략을 수립하는 능력"},
        {"id": "project_management", "name": "프로젝트 관리", "weight": 0.2, "description": "다양한 이해관계자와 협력하여 프로젝트를 성공적으로 기획, 실행, 완료하는 능력"},
        {"id": "negotiation_communication", "name": "협상 및 커뮤니케이션", "weight": 0.2, "description": "내외부 이해관계자들과 효과적으로 소통하고, 상호 이익이 되는 협상 결과를 도출하는 능력"},
        {"id": "scm_understanding", "name": "공급망 관리 이해도", "weight": 0.2, "description": "글로벌 공급망의 특성을 이해하고 효율적인 관리 방안을 모색하는 능력"},
        {"id": "financial_acumen_risk_management", "name": "재무 감각 및 리스크 관리", "weight": 0.2, "description": "사업의 재무적 측면을 이해하고, 잠재적 리스크를 식별 및 관리하는 능력"},
    ]
    return mock_competencies
