"""
Collaboration Node
Low Confidence 역량에 대한 증거 교차검증(Cross-Validation)을 통해 신뢰도를 재평가합니다.
"""

import asyncio
import json
from typing import Dict, List, Coroutine, Any
from datetime import datetime
from openai import AsyncOpenAI
from ..state import EvaluationState

def _find_neighbor_perspectives(
    target_segment_id: int,
    target_competency: str,
    all_segment_evaluations: List[Dict]
) -> List[Dict]:
    """특정 증거(세그먼트)에 대한 다른 에이전트의 평가(관점)를 찾습니다."""
    
    neighbor_perspectives = []
    for seg_eval in all_segment_evaluations:
        # 다른 에이전트가 동일한 세그먼트를 사용한 경우
        if seg_eval.get("segment_id") == target_segment_id and seg_eval.get("competency") != target_competency:
            neighbor_perspectives.append({
                "competency": seg_eval.get("competency"),
                "score": seg_eval.get("score"),
                "reasoning": seg_eval.get("reasoning"),
                "resume_verification": seg_eval.get("resume_verification", {})
            })
    return neighbor_perspectives

def _build_cross_validation_prompt(
    low_confidence_item: Dict,
    aggregated_competencies: Dict,
    all_segment_evaluations: List[Dict]
) -> str:
    """증거 교차검증을 위한 LLM 프롬프트를 생성합니다."""
    
    target_competency = low_confidence_item["competency"]
    original_evaluation = aggregated_competencies.get(target_competency, {})
    
    # 해당 역량 평가에 사용된 모든 증거(세그먼트)를 수집
    evidence_details = original_evaluation.get("perspectives", {}).get("evidence_details", [])
    
    cross_validation_data = []
    for evidence in evidence_details:
        segment_id = evidence.get("segment_id")
        if not segment_id:
            continue
            
        neighbor_perspectives = _find_neighbor_perspectives(
            segment_id,
            target_competency,
            all_segment_evaluations
        )
        
        cross_validation_data.append({
            "quote": evidence.get("text"),
            "segment_id": segment_id,
            "original_evaluation": {
                "score": evidence.get("score"),
                "reasoning": evidence.get("reasoning")
            },
            "neighbor_perspectives": neighbor_perspectives
        })

    prompt = f"""당신은 여러 AI 에이전트의 평가 결과를 검토하는 수석 평가자입니다.

'주니어 평가자'가 '{target_competency}' 역량에 대해 내린 평가의 신뢰도가 낮습니다.
당신의 임무는 주니어 평가자가 사용한 핵심 증거들을 다른 '이웃 평가자'들의 관점과 비교하여, 원래 평가의 신뢰도를 재조정하는 것입니다.

## 낮은 신뢰도의 평가: '{target_competency}'

- **초기 점수**: {original_evaluation.get('overall_score')}점
- **초기 신뢰도**: {original_evaluation.get('confidence_v2'):.2f}
- **초기 강점 분석**: {original_evaluation.get('strengths')}
- **초기 약점 분석**: {original_evaluation.get('weaknesses')}

---

## 증거 교차검증 (Evidence Cross-Validation)

아래는 '{target_competency}' 평가에 사용된 증거와, 동일한 증거에 대한 이웃 평가자들의 상반된 관점입니다.

```json
{json.dumps(cross_validation_data, indent=2, ensure_ascii=False)}
```

---

## 당신의 임무

1.  **신뢰도 재평가**: 위 교차검증 데이터를 바탕으로 '{target_competency}' 역량 평가의 신뢰도를 0.0에서 1.0 사이로 다시 평가하세요.
    - **신뢰도 상승 요인**: 이웃 평가자들의 평가가 원래 평가를 지지하거나, Resume 검증이 'high' 강도로 성공한 경우.
    - **신뢰도 하락 요인**: 이웃 평가자들이 상반된 평가(예: 긍정 vs 부정)를 내리거나, 증거가 여러 역량에 걸쳐 애매하게 사용된 경우.
    - **점수는 바꾸지 말고 신뢰도만 재평가**하세요.

2.  **판단 근거 작성**: 왜 그렇게 신뢰도를 조정했는지 2-3문장으로 명확히 설명하세요. "어떤 증거"에서 "어떤 불일치/일치"를 발견했는지 구체적으로 언급해야 합니다.

## 출력 형식 (JSON)

반드시 아래 JSON 형식만 출력하세요. 다른 텍스트는 절대 포함하지 마세요.

{{
  "competency": "{target_competency}",
  "adjusted_confidence": 0.75,
  "reasoning": "증거(segment 15)에 대해 이웃 에이전트(interpersonal_skill)도 긍정적으로 평가했고, Resume 경력과도 일치하여 신뢰도를 상향 조정함. 다만, 다른 증거(segment 22)는 여러 역량에 걸쳐 해석되어 신뢰도 상승폭을 제한함."
}}
"""
    return prompt

async def _run_collaboration_for_item(
    item: Dict,
    state: EvaluationState
) -> Dict:
    """개별 Low-Confidence 항목에 대한 협업(LLM 호출)을 실행합니다."""
    
    openai_client = state.get("openai_client")
    if not openai_client:
        return {
            "competency": item["competency"],
            "original_score": item["score"],
            "adjusted_score": item["score"],
            "confidence_adjusted": item["confidence_v2"],
            "reason": "Collaboration skipped: OpenAI client not found."
        }

    prompt = _build_cross_validation_prompt(
        item,
        state.get("aggregated_competencies", {}),
        state.get("segment_evaluations_with_resume", [])
    )
    
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a senior AI evaluator performing cross-validation on junior agents' results."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content.strip()
        result_json = json.loads(result_text)
        
        return {
            "competency": item["competency"],
            "original_score": item["score"],
            "adjusted_score": item["score"],  # 점수는 변경하지 않음
            "confidence_adjusted": result_json.get("adjusted_confidence", item["confidence_v2"]),
            "reason": result_json.get("reasoning", "No reasoning provided by LLM.")
        }
    except Exception as e:
        print(f"  [오류] Collaboration for {item['competency']} failed: {e}")
        return {
            "competency": item["competency"],
            "original_score": item["score"],
            "adjusted_score": item["score"],
            "confidence_adjusted": item["confidence_v2"],
            "reason": f"Collaboration failed due to an error: {e}"
        }

async def collaboration_node(state: EvaluationState) -> Dict:
    """
    Collaboration Node
    Low Confidence 역량에 대해 증거 교차검증을 실행하여 신뢰도를 재평가합니다.
    """
    start_time = datetime.now()
    
    print("\n" + "="*60)
    print("[Stage 3] Collaboration 시작")
    print("="*60)
    
    low_confidence_list = state.get("low_confidence_list", [])
    
    if not low_confidence_list:
        print("  Low Confidence 역량 없음 - 스킵")
        return {"collaboration_results": [], "collaboration_count": 0}

    print(f"  교차검증 대상: {len(low_confidence_list)}개 역량")
    for item in low_confidence_list:
        print(f"    - {item['competency']} (current confidence: {item['confidence_v2']:.2f})")

    # 각 Low-Confidence 항목에 대해 병렬로 협업 실행
    collaboration_tasks: List[Coroutine[Any, Any, Dict]] = [
        _run_collaboration_for_item(item, state) for item in low_confidence_list
    ]
    collaboration_results = await asyncio.gather(*collaboration_tasks)
    
    duration = (datetime.now() - start_time).total_seconds()
    
    print(f"\n  Collaboration 완료: {duration:.2f}초")
    print("  조정된 신뢰도:")
    for result in collaboration_results:
        print(f"    - {result['competency']}: {result['confidence_adjusted']:.2f} ({result['reason']})")
    print("="*60)
    
    execution_log = {
        "node": "collaboration",
        "duration_seconds": round(duration, 2),
        "collaborations_done": len(collaboration_results),
        "timestamp": datetime.now().isoformat(),
        "status": "success"
    }
    
    return {
        "collaboration_results": collaboration_results,
        "collaboration_count": len(collaboration_results),
        "execution_logs": state.get("execution_logs", []) + [execution_log]
    }