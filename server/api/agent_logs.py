"""
Agent Logs API - MAS 평가 과정 상세 조회
S3에 저장된 에이전트별 실행 로그를 조회합니다.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from db.database import SessionLocal
from models.evaluation import Evaluation
from models.interview import Applicant
from services.storage.s3_service import S3Service
from api.deps import get_settings

router = APIRouter(prefix="/agent-logs", tags=["Agent Logs"])

def parse_s3_key(s3_url: str) -> str:
    """S3 URL에서 key 추출 (s3://bucket/key -> key)"""
    if s3_url.startswith("s3://"):
        parts = s3_url.replace("s3://", "").split("/", 1)
        return parts[1] if len(parts) > 1 else ""
    return s3_url


class AgentLogsResponse(BaseModel):
    """에이전트 로그 응답"""
    evaluation_id: int
    interview_id: int
    applicant_id: int
    evaluation_run_ts: str

    # 실행 요약
    execution_summary: Dict[str, Any]

    # Stage별 상세 로그
    stage1_evidence: Optional[Dict[str, Any]] = None
    stage2_aggregator: Optional[Dict[str, Any]] = None
    stage3_final: Optional[Dict[str, Any]] = None
    stage4_presentation: Optional[Dict[str, Any]] = None

    # 실행 로그 (타임라인)
    execution_logs: List[Dict[str, Any]] = []


@router.get("/{evaluation_id}", summary="전체 에이전트 로그 조회")
async def get_agent_logs(evaluation_id: int, settings=Depends(get_settings)):
    """
    평가 ID로 전체 에이전트 실행 로그를 조회합니다.

    - Stage 1: 10개 역량 병렬 평가 결과
    - Stage 2: 통합 및 검증 결과
    - Stage 3: 최종 결과
    - Stage 4: 프레젠테이션 포맷
    """
    db = SessionLocal()
    try:
        # 1. DB에서 evaluation 조회
        evaluation = db.query(Evaluation).filter(Evaluation.id == evaluation_id).first()
        if not evaluation:
            raise HTTPException(status_code=404, detail=f"Evaluation {evaluation_id} not found")

        # 2. S3 경로 추출
        metadata = evaluation.evaluation_metadata or {}
        s3_paths = metadata.get("s3_paths", {})
        evaluation_run_ts = metadata.get("evaluation_run_ts", "unknown")

        # 3. S3에서 각 Stage 로그 다운로드
        stage1_data = None
        stage2_data = None
        stage3_data = None
        stage4_data = None
        execution_logs_data = []

        # Stage 1: Evidence
        s3_service = S3Service(bucket_name=settings.S3_BUCKET_NAME, region_name=settings.AWS_REGION)

        if s3_paths.get("stage1_evidence"):
            key = parse_s3_key(s3_paths["stage1_evidence"])
            stage1_data = s3_service.download_json(key)

        # Stage 2: Aggregator
        if s3_paths.get("stage2_aggregator"):
            key = parse_s3_key(s3_paths["stage2_aggregator"])
            stage2_data = s3_service.download_json(key)

        # Stage 3: Final Integration
        if s3_paths.get("stage3_final_integration"):
            key = parse_s3_key(s3_paths["stage3_final_integration"])
            stage3_data = s3_service.download_json(key)

        # Stage 4: Presentation
        if s3_paths.get("stage4_presentation_frontend"):
            key = parse_s3_key(s3_paths["stage4_presentation_frontend"])
            stage4_data = s3_service.download_json(key)

        # Execution Logs
        if s3_paths.get("execution_logs"):
            key = parse_s3_key(s3_paths["execution_logs"])
            execution_logs_data = s3_service.download_json(key) or []

        # 4. 실행 요약 생성
        execution_summary = {
            "total_stages": 4,
            "competencies_evaluated": 10,
            "final_score": evaluation.match_score,
            "confidence_score": evaluation.confidence_score,
            "status": evaluation.evaluation_status,
            "s3_paths": s3_paths
        }

        # Stage별 소요 시간 계산
        stage_durations = {}
        for log in execution_logs_data:
            stage = log.get("stage", log.get("node", "unknown"))
            duration = log.get("duration_seconds", 0)
            if stage not in stage_durations:
                stage_durations[stage] = duration
        execution_summary["stage_durations"] = stage_durations

        return JSONResponse(content={
            "evaluation_id": evaluation_id,
            "interview_id": evaluation.interview_id,
            "applicant_id": evaluation.applicant_id,
            "evaluation_run_ts": evaluation_run_ts,
            "execution_summary": execution_summary,
            "stage1_evidence": stage1_data,
            "stage2_aggregator": stage2_data,
            "stage3_final": stage3_data,
            "stage4_presentation": stage4_data,
            "execution_logs": execution_logs_data
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch agent logs: {str(e)}")
    finally:
        db.close()


@router.get("/{evaluation_id}/stage/{stage_number}", summary="특정 Stage 로그 조회")
async def get_stage_logs(evaluation_id: int, stage_number: int):
    """
    특정 Stage의 상세 로그만 조회합니다.

    - stage_number: 1, 2, 3, 4
    """
    if stage_number not in [1, 2, 3, 4]:
        raise HTTPException(status_code=400, detail="stage_number must be 1, 2, 3, or 4")

    db = SessionLocal()
    try:
        evaluation = db.query(Evaluation).filter(Evaluation.id == evaluation_id).first()
        if not evaluation:
            raise HTTPException(status_code=404, detail=f"Evaluation {evaluation_id} not found")

        metadata = evaluation.evaluation_metadata or {}
        s3_paths = metadata.get("s3_paths", {})

        stage_key_map = {
            1: "stage1_evidence",
            2: "stage2_aggregator",
            3: "stage3_final_integration",
            4: "stage4_presentation_frontend"
        }

        s3_key_name = stage_key_map[stage_number]
        s3_url = s3_paths.get(s3_key_name)

        if not s3_url:
            raise HTTPException(status_code=404, detail=f"Stage {stage_number} logs not found")

        key = parse_s3_key(s3_url)
        data = s3_service.download_json(key)

        if not data:
            raise HTTPException(status_code=404, detail=f"Failed to download Stage {stage_number} logs from S3")

        return JSONResponse(content={
            "evaluation_id": evaluation_id,
            "stage": stage_number,
            "stage_name": s3_key_name,
            "data": data
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stage logs: {str(e)}")
    finally:
        db.close()


@router.get("/{evaluation_id}/competency/{competency_name}", summary="특정 역량 평가 상세 조회")
async def get_competency_detail(evaluation_id: int, competency_name: str):
    """
    특정 역량의 평가 상세 정보를 조회합니다.

    - competency_name: problem_solving, organizational_fit, growth_potential 등
    """
    db = SessionLocal()
    try:
        evaluation = db.query(Evaluation).filter(Evaluation.id == evaluation_id).first()
        if not evaluation:
            raise HTTPException(status_code=404, detail=f"Evaluation {evaluation_id} not found")

        metadata = evaluation.evaluation_metadata or {}
        s3_paths = metadata.get("s3_paths", {})

        # Stage 1에서 역량별 상세 데이터 가져오기
        stage1_url = s3_paths.get("stage1_evidence")
        if not stage1_url:
            raise HTTPException(status_code=404, detail="Stage 1 evidence not found")

        key = parse_s3_key(stage1_url)
        stage1_data = s3_service.download_json(key)

        if not stage1_data:
            raise HTTPException(status_code=404, detail="Failed to download Stage 1 data")

        # 역량 데이터 추출
        competency_data = stage1_data.get(competency_name)
        if not competency_data:
            available = list(stage1_data.keys())
            raise HTTPException(
                status_code=404,
                detail=f"Competency '{competency_name}' not found. Available: {available}"
            )

        # Stage 2에서 segment evaluations 가져오기
        stage2_url = s3_paths.get("stage2_aggregator")
        segment_evaluations = []
        if stage2_url:
            key = parse_s3_key(stage2_url)
            stage2_data = s3_service.download_json(key)
            if stage2_data:
                all_segments = stage2_data.get("segment_evaluations_with_resume", [])
                segment_evaluations = [
                    seg for seg in all_segments
                    if seg.get("competency") == competency_name
                ]

        return JSONResponse(content={
            "evaluation_id": evaluation_id,
            "competency_name": competency_name,
            "evaluation_detail": competency_data,
            "segment_evaluations": segment_evaluations,
            "segment_count": len(segment_evaluations)
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch competency detail: {str(e)}")
    finally:
        db.close()


@router.get("/list/recent", summary="최근 평가 목록 조회")
async def get_recent_evaluations(limit: int = 10):
    """
    최근 평가 목록을 조회합니다. (에이전트 로그 페이지 진입용)
    """
    db = SessionLocal()
    try:
        # Evaluation과 Applicant 조인
        evaluations = db.query(Evaluation, Applicant)\
            .outerjoin(Applicant, Evaluation.applicant_id == Applicant.id)\
            .order_by(Evaluation.created_at.desc())\
            .limit(limit)\
            .all()

        result = []
        for ev, applicant in evaluations:
            metadata = ev.evaluation_metadata or {}
            result.append({
                "evaluation_id": ev.id,
                "interview_id": ev.interview_id,
                "applicant_id": ev.applicant_id,
                "applicant_name": applicant.name if applicant else f"지원자 {ev.applicant_id}",
                "job_id": ev.job_id,
                "match_score": ev.match_score,
                "confidence_score": ev.confidence_score,
                "status": ev.evaluation_status,
                "evaluation_run_ts": metadata.get("evaluation_run_ts"),
                "created_at": ev.created_at.isoformat() if ev.created_at else None
            })

        return JSONResponse(content={"evaluations": result, "count": len(result)})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch evaluations: {str(e)}")
    finally:
        db.close()
