"""
LangGraph Nodes
Stage 1: 10개 Agent 병렬 평가
"""

from typing import Dict
from datetime import datetime
from .state import EvaluationState
from ..competency_agent import CompetencyAgent, evaluate_all_competencies


async def batch_evaluation_node(state: EvaluationState) -> Dict:
    """
    Stage 1: 10개 역량 배치 평가 Node
    
    처리 내용:
        - 10개 Agent를 asyncio.gather로 병렬 실행
        - 각 Agent는 독립적으로 평가 수행
        - Interview Confidence 계산 (Resume 검증 전)
    
    Returns:
        업데이트할 State 필드:
            - achievement_motivation_result
            - growth_potential_result
            - interpersonal_skill_result
            - organizational_fit_result
            - problem_solving_result
            - customer_journey_marketing_result
            - md_data_analysis_result
            - seasonal_strategy_kpi_result
            - stakeholder_collaboration_result
            - value_chain_optimization_result
            - execution_logs
    """
    
    start_time = datetime.now()

    print("\n" + "="*60)
    print("[Stage 1] 에이전트들이 역량 평가를 시작합니다 (10개 병렬)")
    print("="*60)


    # Agent 생성
    agent = CompetencyAgent(
        state["openai_client"],
        max_concurrent=4  # 동시 실행 최대 4개 (TPM 초과 방지)
    )


    # 10개 역량 배치 평가 (동적 역량 전달)
    job_competencies = state.get("job_competencies", None)
    all_results = await evaluate_all_competencies(
        agent,
        state["transcript_content"],
        state["prompts"],
        state.get("session_id"),
        job_competencies
    )


    # 결과 검증
    success_count = sum(
        1
        for r in all_results.values()
        if isinstance(r, dict) and r.get("overall_score", 0) > 0
    )
    error_count = sum(
        1
        for r in all_results.values()
        if isinstance(r, dict) and "error" in r
    )
    scores = [
        r.get("overall_score", 0)
        for r in all_results.values()
        if isinstance(r, dict)
    ]
    avg_score = round(sum(scores) / len(scores), 1) if scores else 0.0
    max_score = max(scores) if scores else 0.0
    min_score = min(scores) if scores else 0.0
    
    print(f"\n[Stage 1] 평가 결과:")
    print(f"  - 성공: {success_count}개")
    print(f"  - 실패: {error_count}개")
    print(f"  - 점수 요약: avg={avg_score:.1f}, max={max_score:.1f}, min={min_score:.1f}")
    
    if error_count > 0:
        print(f"\n    실패한 역량:")
        for name, result in all_results.items():
            if isinstance(result, dict) and "error" in result:
                print(f"    - {name}: {result['error']}")


    # 성능 로깅
    execution_time = (datetime.now() - start_time).total_seconds()
    print(f"\n[Stage 1] 배치 평가 완료 ({execution_time:.2f}초)")
    print("="*60)

    execution_log = {
        "stage": "stage_1",
        "node": "batch_evaluation",
        "duration_seconds": round(execution_time, 2),
        "competencies_evaluated": len(all_results),
        "success_count": success_count,
        "error_count": error_count,
        "timestamp": datetime.now().isoformat(),
        "status": "success" if error_count == 0 else "partial_success"
    }


    # State 업데이트
    return {
        # Common Competencies (5개)
        "achievement_motivation_result": all_results.get("achievement_motivation"),
        "growth_potential_result": all_results.get("growth_potential"),
        "interpersonal_skill_result": all_results.get("interpersonal_skill"),  # ⚠️ "skill" (단수)
        "organizational_fit_result": all_results.get("organizational_fit"),
        "problem_solving_result": all_results.get("problem_solving"),
        
        # Job Competencies (5개)
        "customer_journey_marketing_result": all_results.get("customer_journey_marketing"),
        "md_data_analysis_result": all_results.get("md_data_analysis"),
        "seasonal_strategy_kpi_result": all_results.get("seasonal_strategy_kpi"),
        "stakeholder_collaboration_result": all_results.get("stakeholder_collaboration"),
        "value_chain_optimization_result": all_results.get("value_chain_optimization"),
        
        # Execution Logs
        "execution_logs": [execution_log]  # 첫 번째 로그
    }
