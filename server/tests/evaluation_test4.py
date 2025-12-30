"""
외부 파일에서 transcript와 가중치를 불러와 전체 4단계 결과를 매우 상세히 출력하는 테스트.
evaluation_test3.py의 출력 포맷을 유지하되 입력만 파일로 교체한 버전.
"""
import sys
import os
import json
import asyncio
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.evaluation.evaluation_service import EvaluationService


async def test_full_evaluation():
    """전체 평가 시스템 테스트 (파일 기반 + 상세 출력)"""

    data_dir = Path(__file__).resolve().parent.parent / "test_data"
    transcript_path = data_dir / "transcript_under10.json"
    weights_path = data_dir / "weights_sample.json"

    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)
    with open(weights_path, "r", encoding="utf-8") as f:
        weights = json.load(f)

    def _to_int(value, default):
        try:
            if isinstance(value, str) and value.isdigit():
                return int(value)
            if isinstance(value, (int, float)):
                return int(value)
        except (ValueError, TypeError, AttributeError):
            pass
        return default

    job_weights = weights["job_weights"]
    common_weights = weights["common_weights"]

    # transcript 파일 스키마가 metadata 없이 flat일 수 있으므로 안전하게 추출
    meta = transcript.get("metadata", {})
    interview_raw = meta.get("interview_id") or transcript.get("interview_id") or 1
    applicant_raw = meta.get("applicant_id") or transcript.get("applicant_id") or transcript.get("candidate_id") or 100
    job_raw = meta.get("job_id") or transcript.get("job_id") or 200
    interview_id = _to_int(interview_raw, 1)
    applicant_id = _to_int(applicant_raw, 100)
    job_id = _to_int(job_raw, 200)

    print("\n" + "=" * 80)
    print(" AI 면접 평가 시스템 - 파일 기반 테스트 (Phase 1-4)")
    print("=" * 80 + "\n")

    service = EvaluationService()

    print(" 평가 시작...\n")
    print(f" 인터뷰 ID: {interview_id}, 지원자 ID: {applicant_id}, 잡 ID: {job_id}")

    try:
        result = await service.evaluate_interview(
            interview_id=interview_id,
            applicant_id=applicant_id,
            job_id=job_id,
            transcript=transcript,
            job_weights=job_weights,
            common_weights=common_weights,
        )

        print("\n 평가 완료!\n")

        # ========================
        # Phase 1 결과 출력
        # ========================
        print("=" * 80)
        print(" Phase 1: 개별 역량 평가 결과")
        print("=" * 80)

        job_agg = result["job_aggregation"]
        common_agg = result["common_aggregation"]

        print("\n" + "-" * 80)
        print(" Job 역량 (5개)")
        print("-" * 80)
        print(f"\n 종합 점수: {job_agg['overall_job_score']:.1f}점\n")
        for comp_name in [
            "structured_thinking",
            "business_documentation",
            "financial_literacy",
            "industry_learning",
            "stakeholder_management",
        ]:
            comp_data = job_agg.get(comp_name)
            if comp_data:
                display_name = comp_data.get("competency_display_name", comp_name)
                score = comp_data.get("overall_score", 0)
                confidence = comp_data.get("confidence", {})
                conf_overall = confidence.get("overall_confidence", 0)
                print(f"  ├─ [{display_name}]: {score}점 (신뢰도: {conf_overall:.2f})")

        print("\n" + "-" * 80)
        print(" Common 역량 (5개)")
        print("-" * 80)
        print(f"\n 종합 점수: {common_agg['overall_common_score']:.1f}점\n")
        for comp_name in [
            "problem_solving",
            "organizational_fit",
            "growth_potential",
            "interpersonal_skills",
            "achievement_motivation",
        ]:
            comp_data = common_agg.get(comp_name)
            if comp_data:
                display_name = comp_data.get("competency_display_name", comp_name)
                score = comp_data.get("overall_score", 0)
                confidence = comp_data.get("confidence", {})
                conf_overall = confidence.get("overall_confidence", 0)
                print(f"  ├─ [{display_name}]: {score}점 (신뢰도: {conf_overall:.2f})")

        # ========================
        # Phase 2 결과 출력
        # ========================
        print("\n" + "=" * 80)
        print(" Phase 2: 문제 탐지 결과")
        print("=" * 80)
        issues = result.get("issues_detected", {})
        conflicts = issues.get("evidence_conflicts", [])
        low_conf_list = issues.get("low_confidence_list", [])
        requires_collab = issues.get("requires_collaboration", False)

        print(f"\n  Evidence 충돌: {len(conflicts)}건")
        if conflicts:
            for conf in conflicts:
                seg_id = conf.get("segment_id")
                comps = conf.get("competencies", [])
                gap = conf.get("gap", 0)
                print(f"  ├─ Segment {seg_id}: {', '.join(comps)} (gap: {gap:.2f})")

        print(f"\n  Low Confidence: {len(low_conf_list)}개")
        if low_conf_list:
            for issue in low_conf_list:
                comp = issue.get("competency")
                conf = issue.get("overall_confidence", 0)
                reason = issue.get("reason", "")
                print(f"  ├─ {comp}: {conf:.2f} (원인: {reason})")
        print(f"\n 협업 필요 여부: {'YES (Phase 3 실행)' if requires_collab else 'NO (Phase 4로 바로 진행)'}")

        # ========================
        # Phase 3 결과 출력
        # ========================
        collaboration = result.get("collaboration", {})
        mediation_results = collaboration.get("mediation_results", [])
        adversarial_results = collaboration.get("adversarial_results", [])
        collab_count = collaboration.get("collaboration_count", 0)

        if requires_collab and collab_count > 0:
            print("\n" + "=" * 80)
            print(" Phase 3: 협업 처리 결과")
            print("=" * 80)
            print(f"\n Evidence 중재: {len(mediation_results)}건")
            if mediation_results:
                for med in mediation_results:
                    seg_id = med.get("segment_id")
                    primary = med.get("primary_competency")
                    print(f"  ├─ Segment {seg_id}: Primary={primary}")
            print(f"\n Adversarial 재평가: {len(adversarial_results)}개")
            if adversarial_results:
                for adv in adversarial_results:
                    comp = adv.get("competency")
                    orig = adv.get("original_score", 0)
                    adj = adv.get("adjusted_score", 0)
                    print(f"  ├─ {comp}: {orig}점 → {adj}점")

        # ========================
        # Phase 4 결과 출력
        # ========================
        print("\n" + "=" * 80)
        print(" Phase 4: 최종 통합 결과")
        print("=" * 80)

        final_score = result.get("final_score", 0)
        avg_confidence = result.get("avg_confidence", 0)
        final_reliability = result.get("final_reliability", "")
        reliability_note = result.get("reliability_note", "")

        print(f"\n 최종 점수: {final_score:.1f}점")
        print(f" 평균 Confidence: {avg_confidence:.2f}")
        print(f" 신뢰도 레벨: {final_reliability}")
        print(f" 신뢰도 설명: {reliability_note}")

        final_result = result.get("final_result", {})
        if final_result:
            job_score = final_result.get("job_score", 0)
            common_score = final_result.get("common_score", 0)
            ratio = final_result.get("job_common_ratio", {"job": 0.6, "common": 0.4})
            print(f"\n 세부 점수:")
            print(f"  ├─ Job 점수: {job_score:.1f}점 (가중치 {ratio['job']*100:.0f}%)")
            print(f"  └─ Common 점수: {common_score:.1f}점 (가중치 {ratio['common']*100:.0f}%)")

        # ========================
        # 상세 역량별 출력 (샘플: Common/problem_solving)
        # ========================
        print("\n" + "=" * 80)
        print(" 상세 역량별 평가 (샘플: problem_solving - Common)")
        print("=" * 80)

        st_data = common_agg.get("problem_solving")
        if st_data:
            print(f"\n역량명: {st_data.get('competency_display_name', 'N/A')}")
            print(f"카테고리: {st_data.get('competency_category', 'N/A')}")
            print(f"평가 시각: {st_data.get('evaluated_at', 'N/A')}")
            print(f"최종 점수: {st_data.get('overall_score', 0)}점")

            perspectives = st_data.get("perspectives", {})
            if perspectives:
                print(f"\n" + "-" * 60)
                print(" 3-Perspective 평가 상세")
                print("-" * 60)

                print(f"\n[1] Evidence Perspective:")
                print(f"  ├─ Evidence Score: {perspectives.get('evidence_score', 0)}점")
                print(f"  ├─ Evidence Weight: {perspectives.get('evidence_weight', 0)}")
                print(f"  └─ Reasoning: {perspectives.get('evidence_reasoning', 'N/A')[:200]}...")

                evidence_details = perspectives.get("evidence_details", [])
                if evidence_details:
                    print(f"\n  증거 Quote ({len(evidence_details)}개):")
                    for i, ev in enumerate(evidence_details[:3], 1):
                        seg_id = ev.get("segment_id", "N/A")
                        char_idx = ev.get("char_index", "N/A")
                        text = ev.get("text", "")[:60]
                        relevance = ev.get("relevance_note", "N/A")
                        quality = ev.get("quality_score", 0)
                        print(f"    {i}. [Seg {seg_id}, Idx {char_idx}] Quality: {quality:.2f}")
                        print(f"       \"{text}...\"")
                        print(f"       관련성: {relevance}")

                print(f"\n[2] Behavioral Perspective:")
                print(f"  ├─ Behavioral Score: {perspectives.get('behavioral_score', 0)}점")
                print(f"  └─ Reasoning: {perspectives.get('behavioral_reasoning', 'N/A')[:200]}...")

                behavioral_pattern = perspectives.get("behavioral_pattern", {})
                if behavioral_pattern:
                    print(f"\n  패턴 분석:")
                    print(f"    ├─ 설명: {behavioral_pattern.get('pattern_description', 'N/A')}")
                    examples = behavioral_pattern.get("specific_examples", [])
                    if examples:
                        print(f"    ├─ 구체적 예시 ({len(examples)}개):")
                        for i, ex in enumerate(examples[:2], 1):
                            print(f"    │   {i}. {ex}")
                    consistency = behavioral_pattern.get("consistency_note", "")
                    if consistency:
                        print(f"    └─ 일관성: {consistency}")

                print(f"\n[3] Critical Perspective:")
                print(f"  ├─ Critical Penalties: {perspectives.get('critical_penalties', 0)}점")
                print(f"  └─ Reasoning: {perspectives.get('critical_reasoning', 'N/A')[:200]}...")

                red_flags = perspectives.get("red_flags", [])
                if red_flags:
                    print(f"\n  Red Flags ({len(red_flags)}개):")
                    for i, flag in enumerate(red_flags, 1):
                        flag_type = flag.get("flag_type", "N/A")
                        description = flag.get("description", "N/A")
                        severity = flag.get("severity", "N/A")
                        penalty = flag.get("penalty", 0)
                        evidence_ref = flag.get("evidence_reference", "N/A")
                        print(f"    {i}. Type: {flag_type} | Severity: {severity} | Penalty: {penalty}점")
                        print(f"       설명: {description}")
                        print(f"       증거: {evidence_ref}")

            calculation = st_data.get("calculation", {})
            if calculation:
                print(f"\n" + "-" * 60)
                print(" 점수 계산 상세")
                print("-" * 60)
                print(f"  Base Score: {calculation.get('base_score', 0)}점")
                print(f"  Evidence Weight: {calculation.get('evidence_weight', 0)}")
                print(f"  Behavioral Adjustment: {calculation.get('behavioral_adjustment', 0)}")
                print(f"  Adjusted Base: {calculation.get('adjusted_base', 0)}")
                print(f"  Critical Penalties: {calculation.get('critical_penalties', 0)}")
                print(f"  Final Score: {calculation.get('final_score', 0)}점")
                print(f"  Formula: {calculation.get('formula', 'N/A')}")

            confidence = st_data.get("confidence", {})
            if confidence:
                print(f"\n" + "-" * 60)
                print(" 신뢰도 분석")
                print("-" * 60)
                overall_conf = confidence.get("overall_confidence", 0)
                evidence_str = confidence.get("evidence_strength", 0)
                internal_cons = confidence.get("internal_consistency", 0)
                confidence_note = confidence.get("confidence_note", "N/A")
                print(f"  ├─ Overall Confidence: {overall_conf:.3f} {'  (낮음)' if overall_conf < 0.7 else '✅ (높음)' if overall_conf >= 0.8 else ''}")
                print(f"  ├─ Evidence Strength: {evidence_str:.3f} {'  (부족)' if evidence_str < 0.6 else '✅' if evidence_str >= 0.8 else ''}")
                print(f"  ├─ Internal Consistency: {internal_cons:.3f}")
                print(f"  └─ Note: {confidence_note}")
                if overall_conf >= 0.8 and evidence_str >= 0.8:
                    print("\n   해석: 충분한 증거와 일관된 평가 (신뢰도 높음)")
                elif overall_conf >= 0.7 and evidence_str < 0.6:
                    print("\n   해석: 증거는 적지만 평가는 일관적 (추가 질문 권장)")
                elif overall_conf < 0.7:
                    print("\n   해석: 평가 신뢰도 낮음 (재평가 또는 협업 필요)")

            print(f"\n" + "-" * 60)
            print(" 평가 요약")
            print("-" * 60)
            strengths = st_data.get("strengths", [])
            if strengths:
                print(f"\n강점 ({len(strengths)}개):")
                for i, s in enumerate(strengths, 1):
                    print(f"  {i}. {s}")
            weaknesses = st_data.get("weaknesses", [])
            if weaknesses:
                print(f"\n약점 ({len(weaknesses)}개):")
                for i, w in enumerate(weaknesses, 1):
                    print(f"  {i}. {w}")
            key_observations = st_data.get("key_observations", [])
            if key_observations:
                print(f"\n핵심 관찰 ({len(key_observations)}개):")
                for i, obs in enumerate(key_observations, 1):
                    print(f"  {i}. {obs}")
            followup_questions = st_data.get("suggested_followup_questions", [])
            if followup_questions:
                print(f"\n권장 후속 질문 ({len(followup_questions)}개):")
                for i, q in enumerate(followup_questions, 1):
                    print(f"  {i}. {q}")

        # ========================
        # 실행 로그 출력
        # ========================
        execution_logs = result.get("execution_logs", [])
        if execution_logs:
            print("\n" + "=" * 80)
            print("  실행 로그 (Performance)")
            print("=" * 80 + "\n")
            total_duration = 0
            total_cost = 0
            for log in execution_logs:
                phase = log.get("phase", "")
                node = log.get("node", "")
                duration = log.get("duration_seconds", 0)
                cost = log.get("cost_usd", 0)
                total_duration += duration
                total_cost += cost
                print(f"[{phase}] {node}: {duration:.2f}초")
            print(f"\n총 소요 시간: {total_duration:.2f}초")
            print(f"총 비용 추정: ${total_cost:.4f}")

        # ========================
        # 최종 요약
        # ========================
        print("\n" + "=" * 80)
        print(" 테스트 완료")
        print("=" * 80)
        print(f"\n 요약:")
        print(f"  ├─ 최종 점수: {final_score:.1f}점")
        print(f"  ├─ 신뢰도: {final_reliability}")
        print(f"  ├─ 협업 처리: {collab_count}회")
        if execution_logs:
            print(f"  ├─ 소요 시간: {total_duration:.2f}초")
            print(f"  └─ 추정 비용: ${total_cost:.4f}")
        else:
            print("  ├─ 소요 시간: N/A")
            print("  └─ 추정 비용: N/A")
        print("\n" + "=" * 80 + "\n")

    except Exception as e:
        print(f"\n 오류 발생: {e}\n")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_full_evaluation())
