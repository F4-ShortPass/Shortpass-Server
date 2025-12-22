# /api/interview.py

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, Path, Request 
from starlette.websockets import WebSocketState
from sqlalchemy.orm import Session
from db.database import get_db
from models.interview import InterviewSession, InterviewStatus
from schemas.interview import PrepareInterviewRequest, PrepareInterviewResponse 
from datetime import datetime
import logging
from services.s3_service_factory import get_s3_service
from services.evaluation_pipeline_service import EvaluationPipelineService
from pydantic import BaseModel
from services.interview_service_v3 import interview_service_v3
from services.interview_service_v4 import interview_service_v4

router = APIRouter()
logger = logging.getLogger("uvicorn")


class EvaluationResponse(BaseModel):
    message: str
    log_path: str

# 면접 평가 api
@router.post("/interviews/{interview_id}/run-evaluation", response_model=EvaluationResponse)
async def run_evaluation_pipeline(
    interview_id: int = Path(..., gt=0, description="면접 세션 ID"),
    db: Session = Depends(get_db)
):
    """
    지정된 면접에 대한 전체 평가 파이프라인을 실행합니다.

    - InterviewSession 조회
    - MAS 평가 파이프라인 실행
    - S3에 결과 저장
    """
    logger.info(f"Triggering evaluation pipeline for interview ID: {interview_id}")

    # 1. InterviewSession 존재 여부 확인
    session = db.query(InterviewSession).filter(InterviewSession.id == interview_id).first()
    if not session:
        logger.warning(f"InterviewSession {interview_id} not found")
        raise HTTPException(status_code=404, detail=f"InterviewSession {interview_id} not found")

    # 2. 면접 완료 여부 확인
    if session.status != InterviewStatus.COMPLETED:
        logger.warning(f"Interview {interview_id} is not completed (status: {session.status})")
        raise HTTPException(
            status_code=400,
            detail=f"면접이 완료되지 않았습니다. 현재 상태: {session.status}"
        )

    # 3. 필수 데이터 검증
    if not session.applicant_id:
        logger.error(f"Interview {interview_id} has no applicant_id")
        raise HTTPException(status_code=400, detail="지원자 정보가 없습니다.")

    if not session.job_id:
        logger.error(f"Interview {interview_id} has no job_id")
        raise HTTPException(status_code=400, detail="직무 정보가 없습니다.")

    try:
        s3_service = get_s3_service()
        pipeline_service = EvaluationPipelineService(s3_service=s3_service)

        logger.info(f"Starting evaluation pipeline for interview {interview_id} (applicant: {session.applicant_id}, job: {session.job_id})")

        result = await pipeline_service.run_pipeline(
            company_id=session.company_id,
            job_id=session.job_id,
            applicant_id=session.applicant_id,
            interview_id=interview_id,
            db=db,
        )

        logger.info(f"Evaluation pipeline completed successfully for interview {interview_id}")

        return EvaluationResponse(
            message=result["message"],
            log_path=result["log_path"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running evaluation pipeline for interview {interview_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Evaluation pipeline failed: {str(e)}")


# 면접 세션 준비 API
@router.post("/interviews/prepare", response_model=PrepareInterviewResponse)
async def prepare_interview(
    request_data: PrepareInterviewRequest,
    fastapi_request: Request,
    db: Session = Depends(get_db)
):
    """
    면접 세션을 준비하고,
    클라이언트가 접속할 WebSocket URL을 반환합니다.

    1:1 구조: 1개 회사 + 여러 페르소나 순차 패널
    """
    logger.info(f"Preparing interview for applicant: {request_data.candidateId}, company: {request_data.companyId}")

    try:
        # 지원자 ID와 회사 ID를 정수로 변환
        applicant_id = int(request_data.candidateId)
        company_id = int(request_data.companyId)
        persona_instance_ids = [int(pid) for pid in request_data.personaInstanceIds]

        if not persona_instance_ids:
            logger.warning(f"No persona instances provided for interview preparation")
            raise HTTPException(status_code=400, detail="최소 1개 이상의 페르소나가 필요합니다.")

        # 트랜잭션 시작 - InterviewSession과 SessionPersona를 하나의 트랜잭션으로 처리
        try:
            # InterviewSession 생성 (1:1 구조)
            new_session = InterviewSession(
                applicant_id=applicant_id,
                company_id=company_id,
                status=InterviewStatus.PENDING,
                current_question_index=0,
                current_persona_index=0,
                created_at=datetime.utcnow()
            )
            db.add(new_session)
            db.flush()  # ID 생성을 위해 flush (commit 전)

            # SessionPersona 생성 (순차 패널)
            from models.interview import SessionPersona
            for idx, persona_instance_id in enumerate(persona_instance_ids):
                session_persona = SessionPersona(
                    session_id=new_session.id,
                    persona_instance_id=persona_instance_id,
                    order=idx,
                    role="primary"
                )
                db.add(session_persona)

            # 하나의 트랜잭션으로 모두 커밋
            db.commit()
            db.refresh(new_session)

            logger.info(f"Successfully created interview session {new_session.id} with {len(persona_instance_ids)} personas")

        except Exception as db_error:
            db.rollback()
            logger.error(f"Database transaction failed during interview preparation: {str(db_error)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"면접 세션 생성 실패: {str(db_error)}")

        # WebSocket URL 생성
        ws_protocol = "wss" if fastapi_request.url.scheme == "https" else "ws"
        host_and_port = fastapi_request.url.netloc
        ws_url = f"{ws_protocol}://{host_and_port}/api/v1/ws/interview/{new_session.id}?applicant_id={new_session.applicant_id}"

        logger.info(f"Generated WebSocket URL for interview {new_session.id}: {ws_url}")

        # API 응답 반환
        return PrepareInterviewResponse(
            interviewId=new_session.id,
            applicantId=new_session.applicant_id,
            companyId=new_session.company_id,
            personaInstanceIds=persona_instance_ids,
            status="pending",
            message="면접이 준비되었습니다.",
            websocketUrl=ws_url
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Invalid ID format during interview preparation: {str(e)}")
        raise HTTPException(status_code=400, detail=f"잘못된 ID 형식입니다: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during interview preparation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"면접 준비 실패: {str(e)}")


@router.websocket("/ws/interview/{interview_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    interview_id: int = Path(...),
    applicant_id: int = None  # 쿼리 파라미터: ?applicant_id=101
):
    # interview_id: 준비된 면접 세션 ID
    # applicant_id: 이력서 기반 맞춤 질문 로드용 (optional)

    await websocket.accept()

    try: # v3 : 테스트용 / v4 : 실제 로직
        # await interview_service_v3.handle_interview_session(websocket, interview_id)
        await interview_service_v4.handle_interview_session(websocket, interview_id, applicant_id) 

    except WebSocketDisconnect:
        print(f"API: 클라이언트 연결 종료됨. (ID: {interview_id})")
    except Exception as e:
        print(f"API: 예외 발생 (ID: {interview_id}) - {e}")
        if websocket.state == WebSocketState.CONNECTED:
             await websocket.close(code=1011) # 서버 에러
