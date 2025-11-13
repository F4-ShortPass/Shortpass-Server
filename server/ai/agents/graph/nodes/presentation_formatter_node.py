"""
Presentation Formatter Node
프론트엔드용 데이터 재구성 + LLM 배치 근거 재생성
"""

from typing import Dict
from datetime import datetime

from ..state import EvaluationState
from ...formatter.presentation_formatter_service import PresentationFormatter


async def presentation_formatter_node(state: EvaluationState) -> Dict:
    """
    Presentation Formatter Node
    
    프론트엔드용 데이터 재구성 + 배치 LLM 근거 재생성
    """
    
    start_time = datetime.now()
    
    print("\n" + "="*60)
    print("[Presentation Formatter] 프론트용 데이터 변환 시작")
    print("="*60)
    
    openai_client = state.get("openai_client")
    final_result = state.get("final_result", {})
    aggregated_competencies = state.get("aggregated_competencies", {})
    competency_weights = state.get("competency_weights", {})
    transcript = state.get("transcript")
    formatter = PresentationFormatter(openai_client)
    
    presentation_result = await formatter.format(
        final_result,
        aggregated_competencies,
        competency_weights,
        transcript
    )
    
    duration = (datetime.now() - start_time).total_seconds()
    
    total_evidences = sum(
        len(cd.get('evidences', [])) 
        for cd in presentation_result['competency_details'].values()
    )
    
    print(f"\n  총 근거 생성: {total_evidences}개")
    print("  강점/약점/관찰 평서형 변환: 10개 역량")
    print("  Resume 검증 결과 포함: 완료")
    print(f"  처리 시간: {duration:.2f}초")
    print("  배치 효율: 10개 역량 → 1회 LLM 호출")
    
    print("\n" + "="*60)
    print("[Presentation Formatter] 완료")
    print("="*60)
    
    return {
        "presentation_result": presentation_result,
        "execution_logs": state.get("execution_logs", []) + [{
            "node": "presentation_formatter",
            "duration_seconds": round(duration, 2),
            "total_evidences_generated": total_evidences,
            "batch_llm_calls": 1,
            "components_regenerated": ["evidences", "strengths", "weaknesses", "key_observations"],
            "timestamp": datetime.now().isoformat()
        }]
    }
