import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from server.ai.agents.graph.nodes.collaboration_node import collaboration_node

# 테스트용 Mock 데이터 생성
def create_mock_state(low_confidence_list, aggregated_competencies, segment_evaluations):
    """테스트에 필요한 Mock EvaluationState를 생성합니다."""
    mock_openai_client = MagicMock()
    
    # AsyncMock을 사용하여 chat.completions.create 메서드를 모킹합니다.
    # await 키워드가 이 AsyncMock의 return_value를 반환하게 됩니다.
    mock_openai_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[MagicMock(
                message=MagicMock(
                    content='{"competency": "problem_solving", "adjusted_confidence": 0.85, "reasoning": "Test reason: Confidence increased due to supporting evidence."}'
                )
            )]
        )
    )

    return {
        "low_confidence_list": low_confidence_list,
        "aggregated_competencies": aggregated_competencies,
        "segment_evaluations_with_resume": segment_evaluations,
        "openai_client": mock_openai_client,
        "execution_logs": []
    }

@pytest.mark.asyncio
async def test_collaboration_node_revises_confidence():
    """
    collaboration_node가 low-confidence 역량의 신뢰도를 성공적으로 재평가하는지 테스트합니다.
    """
    # 1. Given: 테스트 데이터 설정
    low_confidence_list = [
        {"competency": "problem_solving", "score": 70, "confidence_v2": 0.55}
    ]
    
    aggregated_competencies = {
        "problem_solving": {
            "overall_score": 70,
            "confidence_v2": 0.55,
            "strengths": ["Good analysis"],
            "weaknesses": ["Lacks structure"],
            "perspectives": {
                "evidence_details": [
                    {"segment_id": 10, "text": "This is evidence 1.", "score": 75, "reasoning": "Shows good analytical skills."}
                ]
            }
        },
        "interpersonal_skill": {
             "overall_score": 80,
             "confidence_v2": 0.9,
             "perspectives": {}
        }
    }
    
    segment_evaluations = [
        # problem_solving 에이전트의 평가
        {"segment_id": 10, "competency": "problem_solving", "score": 75, "reasoning": "Shows good analytical skills."},
        # 이웃 에이전트(interpersonal_skill)의 동일 증거에 대한 평가
        {
            "segment_id": 10, 
            "competency": "interpersonal_skill", 
            "score": 80, 
            "reasoning": "This evidence also shows good communication.",
            "resume_verification": {"verified": True, "strength": "high"}
        }
    ]

    mock_state = create_mock_state(low_confidence_list, aggregated_competencies, segment_evaluations)

    # 2. When: collaboration_node 실행
    result = await collaboration_node(mock_state)

    # 3. Then: 결과 검증
    assert result is not None
    assert result["collaboration_count"] == 1
    
    collab_result = result["collaboration_results"][0]
    assert collab_result["competency"] == "problem_solving"
    assert collab_result["original_score"] == 70
    assert collab_result["adjusted_score"] == 70  # 점수는 변경되지 않음
    assert collab_result["confidence_adjusted"] == 0.85  # Mock LLM이 반환한 값으로 변경됨
    assert "Test reason" in collab_result["reason"]
    
    # OpenAI client가 정확한 프롬프트로 호출되었는지 확인
    mock_state["openai_client"].chat.completions.create.assert_called_once()
    call_args = mock_state["openai_client"].chat.completions.create.call_args
    prompt_content = call_args[1]['messages'][1]['content']
    
    assert "수석 평가자" in prompt_content
    assert "'problem_solving' 역량에 대해 내린 평가의 신뢰도가 낮습니다" in prompt_content
    assert "neighbor_perspectives" in prompt_content
    assert "interpersonal_skill" in prompt_content # 이웃 에이전트의 정보가 포함되었는지 확인

@pytest.mark.asyncio
async def test_collaboration_node_skips_if_no_low_confidence():
    """
    low_confidence_list가 비어있을 때 collaboration_node가 실행을 건너뛰는지 테스트합니다.
    """
    # 1. Given: low_confidence_list가 비어있는 상태
    mock_state = create_mock_state([], {}, [])

    # 2. When: collaboration_node 실행
    result = await collaboration_node(mock_state)

    # 3. Then: 결과 검증
    assert result["collaboration_count"] == 0
    assert result["collaboration_results"] == []
    mock_state["openai_client"].chat.completions.create.assert_not_called() # LLM이 호출되지 않아야 함
