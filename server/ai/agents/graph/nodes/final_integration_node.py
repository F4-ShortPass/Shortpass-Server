"""
Stage 3: 최종 점수 계산 및 신뢰도 평가
"""

from typing import Dict
from datetime import datetime

from ..state import EvaluationState
from ...aggregators.final_integration_service import FinalIntegrator
from services.evaluation.post_processing_service import PostProcessingService


async def final_integration_node(state: EvaluationState) -> Dict:
    """
    Stage 3: Final Integration Node

    Aggregator 결과와 Collaboration 결과를 최종 리포트로 통합합니다.
    """
    start_time = datetime.now()

    print("\n" + "=" * 60)
    print("[Stage 3] Final Integration 시작")
    print("=" * 60)

    aggregated_competencies = state.get("aggregated_competencies", {})
    competency_weights = state.get("competency_weights", {})
    collaboration_results = state.get("collaboration_results", [])
    low_confidence_list = state.get("low_confidence_list", [])
    openai_client = state.get("openai_client")
    post_processing_service = PostProcessingService()

    final_result = await FinalIntegrator.integrate(
        openai_client=openai_client,
        aggregated_competencies=aggregated_competencies,
        competency_weights=competency_weights,
        collaboration_results=collaboration_results,
        low_confidence_list=low_confidence_list
    )

    # 규칙 기반 후처리 요약
    analysis_summary = post_processing_service.build_analysis_summary(
        aggregated_competencies=aggregated_competencies,
        final_result=final_result
    )

    # 요약 로그
    final_score = final_result.get("final_score")
    avg_confidence = final_result.get("avg_confidence")
    reliability = final_result.get("reliability", {}).get("level_display") or final_result.get("reliability", {}).get("level")
    print("\n[Stage 3 요약]")
    print(f"  - final_score: {final_score}")
    print(f"  - avg_confidence: {avg_confidence}")
    print(f"  - reliability: {reliability}")

    duration = (datetime.now() - start_time).total_seconds()

    execution_log = {
        "node": "final_integration",
        "duration_seconds": round(duration, 2),
        "timestamp": datetime.now().isoformat(),
        "status": "success"
    }

    return {
        "avg_confidence": final_result.get("avg_confidence"),
        "final_result": final_result,
        "analysis_summary": analysis_summary,
        "post_processing": {
            "version": "postproc_v1",
            "source": "rules_over_llm_fallback",
            "llm_used": False
        },
        "execution_logs": state.get("execution_logs", []) + [execution_log]
    }