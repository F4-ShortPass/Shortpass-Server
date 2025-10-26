"""
Phase 2 Node: 문제 탐지 (detect_issues_node)
LLM 호출 없이 규칙 기반으로 빠르게 문제 탐지

처리 내용:
1. Evidence 충돌 탐지 (같은 segment, 점수 차이 40점 이상)
2. Low Confidence 탐지 (Confidence < 0.6)
3. 협업 필요 여부 판단

예상 시간: ~1초
예상 비용: $0.00 (LLM 호출 없음)
"""

from typing import Dict
from datetime import datetime

from ...detectors.conflict_detector import ConflictDetector
from ...detectors.confidence_detector import ConfidenceDetector


async def detect_issues_node(state: Dict) -> Dict:
    """
    Phase 2: 문제 탐지 Node
    
    Args:
        state: EvaluationState
    
    Returns:
        업데이트할 State 필드
            - evidence_conflicts
            - low_confidence_list
            - requires_collaboration
    """
    
    start_time = datetime.now()
    
    print("\n" + "="*60)
    print("[Phase 2] 문제 탐지 시작 (규칙 기반, LLM 호출 없음)")
    print("="*60)
    
    
    # 1. 10개 역량 결과 수집
    all_results = {
        "problem_solving": state["problem_solving_result"],
        "organizational_fit": state["organizational_fit_result"],
        "growth_potential": state["growth_potential_result"],
        "interpersonal_skill": state["interpersonal_skill_result"],
        "achievement_motivation": state["achievement_motivation_result"],
        "structured_thinking": state["structured_thinking_result"],
        "business_documentation": state["business_documentation_result"],
        "financial_literacy": state["financial_literacy_result"],
        "industry_learning": state["industry_learning_result"],
        "stakeholder_management": state["stakeholder_management_result"],
    }
    
    # None 값 제거 (평가 실패한 역량)
    all_results = {k: v for k, v in all_results.items() if v is not None}
    
    print(f"✓ 평가 완료된 역량: {len(all_results)}개")
    
    
    # 2. Evidence 충돌 탐지
    print("\n[2-1] Evidence 충돌 탐지...")
    conflicts = ConflictDetector.detect_conflicts(
        all_results,
        threshold=0.4,      # 40점 이상 차이
        max_conflicts=5     # 최대 5개 (Phase 3 비용 제한)
    )
    
    print(ConflictDetector.format_conflicts_for_log(conflicts))
    
    if conflicts:
        summary = ConflictDetector.get_conflict_summary(conflicts)
        print(f"\n   충돌 요약:")
        print(f"     - 최대 gap: {summary['max_gap']}")
        print(f"     - 영향받은 segment: {len(summary['affected_segments'])}개")
        print(f"     - 영향받은 역량: {len(summary['affected_competencies'])}개")
    
    
    # 3. Low Confidence 탐지
    print("\n[2-2] Low Confidence 탐지...")
    low_confidence_issues = ConfidenceDetector.detect_low_confidence(
        all_results,
        threshold=0.6,      # Confidence < 0.6
        max_issues=5        # 최대 5개 (Phase 3 비용 제한)
    )
    
    print(ConfidenceDetector.format_issues_for_log(low_confidence_issues))
    
    if low_confidence_issues:
        summary = ConfidenceDetector.get_issue_summary(low_confidence_issues)
        print(f"\n   Low Confidence 요약:")
        print(f"     - 최소 Confidence: {summary['min_confidence']}")
        print(f"     - 원인 분석:")
        print(f"       * 증거 부족 (지원자): {summary['reason_breakdown']['evidence_weak']}개")
        print(f"       * 일관성 부족 (평가): {summary['reason_breakdown']['consistency_low']}개")
        print(f"       * 둘 다: {summary['reason_breakdown']['both']}개")
    
    
    # 4. 협업 필요 여부 판단
    
    
    requires_collaboration = len(conflicts) > 0 or len(low_confidence_issues) > 0
    
    print("\n" + "-"*60)
    if requires_collaboration:
        print(f"⚠️  협업 필요: Evidence 충돌 {len(conflicts)}건 + Low Confidence {len(low_confidence_issues)}개")
        print("   → Phase 3 (Targeted Collaboration)로 진행")
    else:
        print("✅ 문제 없음 - Phase 4 (최종 통합)로 바로 진행")
    print("-"*60)
    
    
    # 5. 성능 로깅
    duration = (datetime.now() - start_time).total_seconds()
    
    execution_log = {
        "phase": "phase_2",
        "node": "detect_issues",
        "duration_seconds": round(duration, 2),
        "conflicts_found": len(conflicts),
        "low_confidence_found": len(low_confidence_issues),
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"\n  Phase 2 완료: {duration:.2f}초")
    print("="*60)
    
    
    # 6. State 업데이트
    return {
        "evidence_conflicts": [c.to_dict() for c in conflicts],
        "low_confidence_list": [i.to_dict() for i in low_confidence_issues],
        "requires_collaboration": requires_collaboration,
        "execution_logs": state["execution_logs"] + [execution_log]
    }
