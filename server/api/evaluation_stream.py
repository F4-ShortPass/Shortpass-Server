"""
실시간 평가 스트리밍 API (SSE)
- 프론트엔드에서 평가 진행 상황을 실시간으로 표시
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Optional, AsyncGenerator
import asyncio
import json
from datetime import datetime

from services.evaluation.evaluation_service import EvaluationService
from services.storage.s3_service import S3Service
from api.deps import get_settings

router = APIRouter(prefix="/evaluations/stream")

# 진행 상황 저장소
progress_store: Dict[int, dict] = {}


class StreamEvaluationRequest(BaseModel):
    """스트리밍 평가 요청"""
    interview_id: int
    applicant_id: int
    job_id: int
    competency_weights: Optional[Dict[str, float]] = None


def create_sse_message(event: str, data: dict) -> str:
    """SSE 메시지 포맷"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_evaluation_progress(
    interview_id: int,
    applicant_id: int,
    job_id: int,
    competency_weights: Dict[str, float],
    settings
) -> AsyncGenerator[str, None]:
    """평가 진행 상황을 SSE로 스트리밍"""

    evaluation_id = interview_id  # 간단히 interview_id를 사용

    # 초기화
    progress_store[evaluation_id] = {
        "status": "initializing",
        "current_stage": 0,
        "stages": []
    }

    try:
        # === Stage 0: 초기화 ===
        yield create_sse_message("progress", {
            "stage": 0,
            "stage_name": "초기화",
            "status": "in_progress",
            "message": "평가 시스템 초기화 중...",
            "progress": 0
        })

        # S3에서 데이터 로드
        s3_service = S3Service(bucket_name=settings.S3_BUCKET_NAME, region_name=settings.AWS_REGION)

        # transcript 로드
        transcript_key = f"interviews/{interview_id}/transcript.json"
        transcript = s3_service.download_json(transcript_key)

        if not transcript:
            yield create_sse_message("error", {
                "message": f"Transcript를 찾을 수 없습니다: {transcript_key}"
            })
            return

        # resume 로드
        resume_key = f"applicants/{applicant_id}/resume.json"
        resume_data = s3_service.download_json(resume_key)

        yield create_sse_message("progress", {
            "stage": 0,
            "stage_name": "초기화",
            "status": "completed",
            "message": f"데이터 로드 완료 (Segments: {len(transcript.get('segments', []))}개)",
            "progress": 5
        })

        await asyncio.sleep(0.3)

        # === Stage 1: 10개 역량 평가 시작 ===
        yield create_sse_message("progress", {
            "stage": 1,
            "stage_name": "Stage 1: 역량별 평가",
            "status": "in_progress",
            "message": "10개 역량 병렬 평가 시작...",
            "progress": 10
        })

        competencies = [
            "achievement_motivation", "growth_potential", "interpersonal_skill",
            "organizational_fit", "problem_solving",
            "customer_journey_marketing", "md_data_analysis", "seasonal_strategy_kpi",
            "stakeholder_collaboration", "value_chain_optimization"
        ]

        # 역량별 평가 시뮬레이션 (실제로는 병렬 실행)
        for i, comp in enumerate(competencies):
            yield create_sse_message("competency_start", {
                "stage": 1,
                "competency": comp,
                "index": i + 1,
                "total": 10,
                "message": f"[평가 시작] {comp}"
            })

            await asyncio.sleep(0.1)  # 실제 평가 시간 대신 짧은 딜레이

        yield create_sse_message("progress", {
            "stage": 1,
            "stage_name": "Stage 1: 역량별 평가",
            "status": "in_progress",
            "message": "AI 평가 진행 중... (약 30초 소요)",
            "progress": 20
        })

        # 실제 평가 실행
        service = EvaluationService()

        # 평가 시작 (내부에서 Stage 1-3 실행)
        result = await service.evaluate_interview(
            interview_id=interview_id,
            applicant_id=applicant_id,
            job_id=job_id,
            transcript=transcript,
            competency_weights=competency_weights,
            resume_data=resume_data
        )

        # 역량별 결과 전송
        aggregated = result.get("aggregated_competencies", {})
        for i, comp in enumerate(competencies):
            comp_data = aggregated.get(comp, {})
            score = comp_data.get("overall_score", 0)

            yield create_sse_message("competency_complete", {
                "stage": 1,
                "competency": comp,
                "score": score,
                "index": i + 1,
                "total": 10,
                "message": f"[평가 완료] {comp}: {score}점"
            })

        yield create_sse_message("progress", {
            "stage": 1,
            "stage_name": "Stage 1: 역량별 평가",
            "status": "completed",
            "message": "10개 역량 평가 완료",
            "progress": 40
        })

        # === Stage 2: Aggregator ===
        yield create_sse_message("progress", {
            "stage": 2,
            "stage_name": "Stage 2: 통합 분석",
            "status": "in_progress",
            "message": "Resume 검증 및 Confidence 계산 중...",
            "progress": 50
        })

        await asyncio.sleep(0.3)

        # Sub-steps
        segment_evals = result.get("segment_evaluations_with_resume", [])
        verified_count = sum(1 for s in segment_evals if s.get("resume_verified"))

        yield create_sse_message("substep", {
            "stage": 2,
            "substep": "resume_verification",
            "message": f"Resume 검증 완료: {verified_count}개 검증됨"
        })

        yield create_sse_message("substep", {
            "stage": 2,
            "substep": "confidence_v2",
            "message": f"Confidence V2 계산 완료: 평균 {result.get('avg_confidence', 0):.2f}"
        })

        yield create_sse_message("progress", {
            "stage": 2,
            "stage_name": "Stage 2: 통합 분석",
            "status": "completed",
            "message": "통합 분석 완료",
            "progress": 70
        })

        # === Stage 3: Final Integration ===
        yield create_sse_message("progress", {
            "stage": 3,
            "stage_name": "Stage 3: 최종 통합",
            "status": "in_progress",
            "message": "최종 점수 계산 및 심사평 생성 중...",
            "progress": 80
        })

        await asyncio.sleep(0.3)

        yield create_sse_message("progress", {
            "stage": 3,
            "stage_name": "Stage 3: 최종 통합",
            "status": "completed",
            "message": "최종 통합 완료",
            "progress": 90
        })

        # === Stage 4: Presentation ===
        yield create_sse_message("progress", {
            "stage": 4,
            "stage_name": "Stage 4: 결과 포맷팅",
            "status": "in_progress",
            "message": "프론트엔드용 데이터 변환 중...",
            "progress": 95
        })

        await asyncio.sleep(0.2)

        yield create_sse_message("progress", {
            "stage": 4,
            "stage_name": "Stage 4: 결과 포맷팅",
            "status": "completed",
            "message": "포맷팅 완료",
            "progress": 100
        })

        # === 최종 결과 ===
        final_score = result.get("final_score", 0)
        competency_details = result.get("competency_details", {})

        # 역량별 점수 추출
        competency_scores = {}
        for comp in competencies:
            data = competency_details.get(comp, aggregated.get(comp, {}))
            competency_scores[comp] = {
                "score": data.get("overall_score", 0),
                "confidence": data.get("confidence_v2", data.get("interview_confidence", 0))
            }

        yield create_sse_message("complete", {
            "status": "completed",
            "final_score": final_score,
            "avg_confidence": result.get("avg_confidence", 0),
            "reliability_level": result.get("final_reliability", ""),
            "competency_scores": competency_scores,
            "s3_urls": {
                "transcript": result.get("transcript_s3_url"),
                "stage1": result.get("stage1_evidence_s3_url"),
                "stage2": result.get("stage2_aggregator_s3_url"),
                "stage3": result.get("stage3_final_integration_s3_url"),
                "presentation": result.get("stage4_presentation_s3_url")
            },
            "message": f"평가 완료! 최종 점수: {final_score:.1f}점"
        })

    except Exception as e:
        yield create_sse_message("error", {
            "status": "failed",
            "message": f"평가 중 오류 발생: {str(e)}"
        })

    finally:
        # 정리
        if evaluation_id in progress_store:
            del progress_store[evaluation_id]


@router.post("/start")
async def start_streaming_evaluation(
    request: StreamEvaluationRequest,
    settings=Depends(get_settings)
):
    """
    실시간 평가 스트리밍 시작

    Returns:
        SSE 스트림으로 진행 상황 전송
    """

    # 기본 가중치 (데이터 분석 강조 - 김지원 유리)
    weights = request.competency_weights or {
        "achievement_motivation": 0.08,
        "growth_potential": 0.08,
        "interpersonal_skill": 0.05,   # 낮춤 (0.10 -> 0.05)
        "organizational_fit": 0.08,
        "problem_solving": 0.11,
        "customer_journey_marketing": 0.10,
        "md_data_analysis": 0.25,      # 높임 (0.15 -> 0.25)
        "seasonal_strategy_kpi": 0.10,
        "stakeholder_collaboration": 0.05,
        "value_chain_optimization": 0.10,  # 높임 (0.05 -> 0.10)
    }

    return StreamingResponse(
        stream_evaluation_progress(
            interview_id=request.interview_id,
            applicant_id=request.applicant_id,
            job_id=request.job_id,
            competency_weights=weights,
            settings=settings
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
