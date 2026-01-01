"""
LangGraph Evaluation Graph
Stage 1 → Stage 2 → Stage 3 (조건부 Collaboration)

플로우:
START → Stage 1 (10개 Agent 병렬) 
      → Stage 2 (Aggregator: Resume + Confidence V2 + Segment Overlap + Cross-Competency)
      → [조건부 분기]
          ├─ Low Confidence 있음 → Collaboration (선택적) → Final Integration
          └─ 문제 없음 → Final Integration
      → END
"""

from langgraph.graph import StateGraph, END
from .state import EvaluationState
from .nodes.nodes import batch_evaluation_node
from .nodes.aggregator_node import aggregator_node
from .nodes.collaboration_node import collaboration_node
from .nodes.final_integration_node import final_integration_node
from .nodes.presentation_formatter_node import presentation_formatter_node

def should_collaborate(state: EvaluationState) -> str:
    """
    Stage 2 후 협업 필요 여부 판단
    
    Returns:
        "collaboration": Collaboration Node로 진행 (선택적)
        "final_integration": Final Integration으로 바로 진행
    """
    requires_collaboration = state.get("requires_collaboration", False)
    low_confidence_list = state.get("low_confidence_list", [])
    
    print("\n[조건부 분기] Collaboration 필요 여부 판단...")
    print(f"  - requires_collaboration: {requires_collaboration}")
    print(f"  - Low Confidence 역량 수: {len(low_confidence_list)}")
    
    if requires_collaboration and low_confidence_list:
        print("  → 'collaboration' 노드로 진행")
        return "collaboration"
    else:
        print("  → 'final_integration' 노드로 바로 진행")
        return "final_integration"


def create_evaluation_graph():
    """
    평가 그래프 생성
    
    플로우:
    Stage 1: 10개 Agent 병렬 평가
    Stage 2: Aggregator (Resume + Confidence V2 + Segment Overlap + Cross-Competency)
    Stage 3: Final Integration (+ 선택적 Collaboration)
    """
    
    graph = StateGraph(EvaluationState)
    
    

    # 1. Node 등록   
    # Stage 1: 10개 Agent 병렬 평가
    graph.add_node("batch_evaluation", batch_evaluation_node)
    
    # Stage 2: Aggregator
    graph.add_node("aggregator", aggregator_node)
    
    # Stage 3: Collaboration (선택적)
    graph.add_node("collaboration", collaboration_node)
    
    # Stage 3: Final Integration
    graph.add_node("final_integration", final_integration_node)
    
    graph.add_node("presentation_formatter", presentation_formatter_node)

    # 2. Edge 정의   
    # Entry Point
    graph.set_entry_point("batch_evaluation")
    
    # Stage 1 → Stage 2
    graph.add_edge("batch_evaluation", "aggregator")
    
    # Stage 2 → Stage 3 (조건부 분기)
    graph.add_conditional_edges(
        "aggregator",
        should_collaborate,
        {
            "collaboration": "collaboration",
            "final_integration": "final_integration"
        }
    )
    
    # Collaboration → Final Integration
    graph.add_edge("collaboration", "final_integration")

    # Final Integration → Presentation Formatter → END
    graph.add_edge("final_integration", "presentation_formatter")
    graph.add_edge("presentation_formatter", END)
    
    return graph.compile()
