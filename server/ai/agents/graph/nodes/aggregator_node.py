"""
Aggregator Node
Stage 2: 10개 역량 집계 및 검증

Sub-steps:
    2.1. Resume Verification (AI 1회, Batch)
    2.2. Confidence V2 계산 (Rule-based)
    2.3. Segment Overlap Check (조건부, Rule + AI)
    2.4. Cross-Competency Validation (Rule-based)
"""

from typing import Dict, List
from datetime import datetime
from ..state import EvaluationState
from ...aggregators.resume_verifier import ResumeVerifier
from ...aggregators.confidence_calculator import ConfidenceCalculator
from ...aggregators.segment_overlap_checker import SegmentOverlapChecker


def _extract_resume_verification_summary(comp_segments: List[Dict]) -> Dict:
    """
    역량별 Resume 검증 근거 추출 (상위 3개만)
    
    Args:
        comp_segments: 특정 역량의 Segment 평가 목록
    
    Returns:
        {
            "verified_count": 3,
            "high_strength_count": 2,
            "key_evidence": [
                {
                    "segment_id": 3,
                    "quote": "학부생 때 창업 프로젝트를 이끌었습니다",
                    "resume_section": "projects",
                    "matched_content": "학부생 창업 프로젝트 (2020.03-2021.02) | 팀장",
                    "verification_strength": "high",
                    "reasoning": "면접 답변과 Resume 경력이 정확히 일치. 역할(팀장)도 확인됨."
                },
                ...
            ]
        }
    """
    verified_segments = [
        s for s in comp_segments 
        if s.get("resume_verification", {}).get("verified", False)
    ]
    
    high_strength_segments = [
        s for s in verified_segments
        if s.get("resume_verification", {}).get("strength") == "high"
    ]
    
    # 검증 강도 순으로 정렬 (high > medium > low)
    strength_order = {"high": 3, "medium": 2, "low": 1, "none": 0}
    verified_segments_sorted = sorted(
        verified_segments,
        key=lambda s: strength_order.get(s.get("resume_verification", {}).get("strength", "none"), 0),
        reverse=True
    )
    
    # 상위 3개만 추출
    key_evidence = []
    for seg in verified_segments_sorted[:3]:
        resume_verification = seg.get("resume_verification", {})
        resume_matches = resume_verification.get("resume_matches", [])
        
        # 첫 번째 매칭 정보 사용
        first_match = resume_matches[0] if resume_matches else {}
        
        key_evidence.append({
            "segment_id": seg.get("segment_id"),
            "quote": seg.get("quote_text", ""),
            "resume_section": first_match.get("resume_section", ""),
            "matched_content": first_match.get("matched_content", ""),
            "resume_verification": seg.get("resume_verification", {}),
            "verification_strength": resume_verification.get("strength", "none"),
            "reasoning": resume_verification.get("reasoning", ""),
            "char_index": seg.get("char_index"), 
            "char_length": seg.get("char_length") or len(seg.get("quote_text", "")) 
        })
    
    return {
        "verified_count": len(verified_segments),
        "high_strength_count": len(high_strength_segments),
        "key_evidence": key_evidence
    }


async def aggregator_node(state: EvaluationState) -> Dict:
    """
    Stage 2: Aggregator Node
    
    처리 내용:
        1. Resume Verification (AI 1회, Batch)
        2. Confidence V2 계산 (Rule-based)
        3. Segment Overlap Check (조건부, Rule + AI)
        4. Cross-Competency Validation (Rule-based)
    
    Returns:
        업데이트할 State 필드:
            - segment_evaluations_with_resume
            - confidence_v2_calculated
            - segment_overlap_adjustments (내부 로직용)
            - cross_competency_flags (내부 로직용)
            - aggregated_competencies (Resume 검증 근거 포함)
            - low_confidence_list
            - requires_collaboration
    """
    
    start_time = datetime.now()
    
    print("\n" + "="*60)
    print("[Stage 2] Aggregator 시작")
    print("="*60)
    

    # 입력 데이터 준비

    
    # 10개 역량 결과 수집
    all_competency_results = {
        # Common Competencies (5개)
        "achievement_motivation": state.get("achievement_motivation_result"),
        "growth_potential": state.get("growth_potential_result"),
        "interpersonal_skill": state.get("interpersonal_skill_result"),
        "organizational_fit": state.get("organizational_fit_result"),
        "problem_solving": state.get("problem_solving_result"),
        
        # Job Competencies (5개)
        "customer_journey_marketing": state.get("customer_journey_marketing_result"),
        "md_data_analysis": state.get("md_data_analysis_result"),
        "seasonal_strategy_kpi": state.get("seasonal_strategy_kpi_result"),
        "stakeholder_collaboration": state.get("stakeholder_collaboration_result"),
        "value_chain_optimization": state.get("value_chain_optimization_result"),
    }
    
    # None 제거
    all_competency_results = {k: v for k, v in all_competency_results.items() if v is not None}
    
    print(f"\n✓ 평가 완료된 역량: {len(all_competency_results)}개")
    
    resume_data = state.get("resume_data")
    openai_client = state.get("openai_client")
    

    # Sub-step 2.1: Resume Verification

    
    print("\n[Sub-step 2.1] Resume Verification (AI 1회, Batch)")
    print("-" * 60)
    
    verifier = ResumeVerifier(openai_client)
    
    segment_evaluations_with_resume = await verifier.verify_batch(
        all_competency_results,
        resume_data
    )
    
    # 통계 출력
    verified_count = sum(
        1 for s in segment_evaluations_with_resume 
        if s.get("resume_verification", {}).get("verified", False)
    )
    high_strength_count = sum(
        1 for s in segment_evaluations_with_resume 
        if s.get("resume_verification", {}).get("strength") == "high"
    )
    
    print(f"\n  Resume 검증 완료:")
    print(f"    - 총 Segment 평가: {len(segment_evaluations_with_resume)}개")
    print(f"    - 검증됨: {verified_count}개")
    print(f"    - 강한 검증 (high): {high_strength_count}개")
    if segment_evaluations_with_resume:
        print("    - 샘플(상위 3개):")
        for seg in segment_evaluations_with_resume[:3]:
            rv = seg.get("resume_verification", {})
            print(
                f"      • seg#{seg.get('segment_id')} [{seg.get('competency')}] "
                f"verified={rv.get('verified')} strength={rv.get('strength', '-')}"
            )
    

    # Sub-step 2.2: Confidence V2 재계산

    
    print("\n[Sub-step 2.2] Confidence V2 재계산 (Rule-based)")
    print("-" * 60)
    
    segment_evaluations_with_conf_v2 = ConfidenceCalculator.calculate_for_segments(
        segment_evaluations_with_resume
    )
    
    # 통계 출력 (안전 처리)
    if segment_evaluations_with_conf_v2:
        avg_conf_v2 = sum(s["confidence_v2"] for s in segment_evaluations_with_conf_v2) / len(segment_evaluations_with_conf_v2)
        improved_count = sum(
            1 for s in segment_evaluations_with_conf_v2
            if s["confidence_v2"] > s["interview_confidence"]
        )

        print(f"\n  Confidence V2 계산 완료:")
        print(f"    - 평균 Confidence V2: {avg_conf_v2:.2f}")
        print(f"    - 개선된 평가: {improved_count}개")

        sorted_by_conf = sorted(segment_evaluations_with_conf_v2, key=lambda s: s["confidence_v2"], reverse=True)
        best = sorted_by_conf[0]
        worst = sorted_by_conf[-1]
        print(f"    - 최고/최저: seg#{best.get('segment_id')}={best['confidence_v2']:.2f} / seg#{worst.get('segment_id')}={worst['confidence_v2']:.2f}")
    else:
        print(f"\n  ⚠️  Segment 평가가 비어있음 - Confidence V2 계산 스킵")
        avg_conf_v2 = 0.5  # 기본값
        improved_count = 0
    

    # Sub-step 2.3: Segment Overlap Check (내부 로직용)

    
    print("\n[Sub-step 2.3] Segment Overlap Check (조건부)")
    print("-" * 60)
    
    overlap_checker = SegmentOverlapChecker(openai_client)
    
    adjusted_segments, segment_overlap_adjustments = await overlap_checker.check_and_adjust(
        segment_evaluations_with_conf_v2
    )
    
    print(f"\n  Segment Overlap 체크 완료:")
    print(f"    - 조정된 Segment: {len(segment_overlap_adjustments)}개")
    if segment_overlap_adjustments:
        for adj in segment_overlap_adjustments:
            print(f"      * Segment {adj['segment_id']}: {len(adj['adjustments'])}개 역량 조정 ({adj['adjustment_type']})")
            for a in adj.get("adjustments", [])[:2]:
                print(
                    f"         · {a.get('competency')}: {a.get('original_score')}→{a.get('adjusted_score')} "
                    f"(conf {a.get('original_confidence')}→{a.get('adjusted_confidence')})"
                )
            if len(adj.get("adjustments", [])) > 2:
                print("         · ...")
    print("      이 정보는 내부 로직용이며 프론트엔드에 노출하지 않습니다.")
    

    # Sub-step 2.4: Cross-Competency Validation (내부 로직용)

    
    print("\n[Sub-step 2.4] Cross-Competency Validation (Rule-based)")
    print("-" * 60)
    
    # 역량별 평균 Confidence V2 계산
    competency_confidences = {}
    
    for comp_name in all_competency_results.keys():
        # 해당 역량의 Segment 평가들만 필터링
        comp_segments = [
            s for s in adjusted_segments 
            if s["competency"] == comp_name
        ]
        
        if comp_segments:
            # 평균 Confidence V2
            avg_conf = sum(s["confidence_v2"] for s in comp_segments) / len(comp_segments)
            competency_confidences[comp_name] = round(avg_conf, 2)
        else:
            # Segment가 없으면 원본 Confidence 사용
            original_conf = all_competency_results[comp_name].get("confidence", {}).get("overall_confidence", 0.5)
            competency_confidences[comp_name] = original_conf
    
    # Low Confidence 탐지 (Threshold: 0.6)
    LOW_CONFIDENCE_THRESHOLD = 0.6
    
    low_confidence_list = []
    cross_competency_flags = []
    
    for comp_name, avg_conf in competency_confidences.items():
        if avg_conf < LOW_CONFIDENCE_THRESHOLD:
            comp_result = all_competency_results[comp_name]
            overall_score = comp_result.get("overall_score", 0)
            
            low_confidence_list.append({
                "competency": comp_name,
                "score": overall_score,
                "confidence_v2": avg_conf,
                "reason": "low_confidence_after_aggregation"
            })
            
            cross_competency_flags.append({
                "competency": comp_name,
                "flag_type": "low_confidence",
                "confidence_v2": avg_conf,
                "note": f"Confidence V2 ({avg_conf:.2f}) < Threshold ({LOW_CONFIDENCE_THRESHOLD})"
            })
    
    print(f"\n  Cross-Competency 검증 완료:")
    print(f"    - Low Confidence 역량: {len(low_confidence_list)}개")
    if low_confidence_list:
        for item in low_confidence_list:
            print(f"      * {item['competency']}: Conf={item['confidence_v2']:.2f}")
    print("      이 정보는 내부 로직용이며 프론트엔드에 노출하지 않습니다.")
    

    # 역량별 최종 결과 집계 (Resume 검증 근거 포함)
    print("\n[역량별 최종 결과 집계]")
    print("-" * 60)
    
    aggregated_competencies = {}
    
    for comp_name, comp_result in all_competency_results.items():
        # 해당 역량의 Segment 평가들
        comp_segments = [s for s in adjusted_segments if s["competency"] == comp_name]
        
        # 평균 Confidence V2
        avg_conf_v2 = competency_confidences.get(comp_name, 0.5)
        
        # 원본 점수 (Agent가 계산한 overall_score)
        original_score = comp_result.get("overall_score", 0)
        
        #  Resume 검증 근거 추출 (상위 3개만)
        resume_verification_summary = _extract_resume_verification_summary(comp_segments)
        
        perspectives = comp_result.get("perspectives", {})
        evidence_details = perspectives.get("evidence_details", [])
        
        # char_length 추가 (없으면 계산)
        for ev in evidence_details:
            if "char_length" not in ev or ev["char_length"] is None:
                ev["char_length"] = len(ev.get("text", ""))
        aggregated_competencies[comp_name] = {
            "competency_name": comp_name,
            "overall_score": original_score,  # Agent 점수 유지
            "interview_confidence": comp_result.get("confidence", {}).get("overall_confidence", 0.5),
            "confidence_v2": avg_conf_v2,  # Resume 검증 반영
            "segment_count": len(comp_segments),
            "resume_verified_count": sum(
                1 for s in comp_segments 
                if s.get("resume_verification", {}).get("verified", False)
            ),
            "adjusted_by_overlap": any(s.get("adjusted") for s in comp_segments),
            "perspectives": {
                **perspectives,
                "evidence_details": evidence_details  
            },
            "strengths": comp_result.get("strengths"),
            "weaknesses": comp_result.get("weaknesses"),
            "key_observations": comp_result.get("key_observations", []),
            "resume_verification_summary": resume_verification_summary  
        }
    
    print(f"\n  역량별 집계 완료: {len(aggregated_competencies)}개")
    for comp_name, agg in aggregated_competencies.items():
        verified = agg['resume_verification_summary']['verified_count']
        high_strength = agg['resume_verification_summary']['high_strength_count']
        print(f"    - {comp_name}: {agg['overall_score']}점 (Conf V2: {agg['confidence_v2']:.2f}, Resume: {verified}개 검증, {high_strength}개 강함)")
        print(
            f"      · segments={agg.get('segment_count', 0)} "
            f"resume_verified={agg.get('resume_verified_count', 0)} overlap_adjusted={agg.get('adjusted_by_overlap', False)}"
        )
    

    # 협업 필요 여부 판단   
    requires_collaboration = len(low_confidence_list) > 0
    
    print("\n" + "-"*60)
    if requires_collaboration:
        print(f"    협업 필요: Low Confidence 역량 {len(low_confidence_list)}개")
        print("   → Collaboration Node로 진행 (선택적)")
    else:
        print("   모든 역량 신뢰도 양호 - Final Integration으로 바로 진행")
    print("-"*60)
    

    # 성능 로깅

    
    duration = (datetime.now() - start_time).total_seconds()
    
    execution_log = {
        "stage": "stage_2",
        "node": "aggregator",
        "duration_seconds": round(duration, 2),
        "segments_processed": len(adjusted_segments),
        "resume_verified_count": verified_count,
        "overlap_adjustments": len(segment_overlap_adjustments),
        "low_confidence_count": len(low_confidence_list),
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"\n  Stage 2 완료: {duration:.2f}초")
    print("="*60)
    

    # State 업데이트   
    return {
        # Sub-step 결과
        "segment_evaluations_with_resume": segment_evaluations_with_resume,
        "confidence_v2_calculated": True,
        "segment_overlap_adjustments": segment_overlap_adjustments,  # 내부 로직용
        "cross_competency_flags": cross_competency_flags,  # 내부 로직용
        
        # 최종 집계 (Resume 검증 근거 포함)
        "aggregated_competencies": aggregated_competencies,
        
        # 협업 필요 여부
        "low_confidence_list": low_confidence_list,
        "requires_collaboration": requires_collaboration,
        
        # 로그
        "execution_logs": state.get("execution_logs", []) + [execution_log]
    }
