"""
Agent가 사용한 Segment 분석 스크립트

목적:
- OpenAI API 할당량 문제로 임베딩 대신 기존 Agent 로그 분석
- 역량별로 어떤 segment를 선택했는지 패턴 파악
- RAG 도입 시 예상 효과 시뮬레이션

데이터:
- stage1_evidence.json (Agent 평가 결과)
- transcript_jiwon_101.json (면접 대화)
"""
import os
import sys
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


COMPETENCY_DISPLAY_NAMES = {
    "achievement_motivation": "성취/동기 역량",
    "growth_potential": "성장 잠재력",
    "interpersonal_skill": "대인관계 역량",
    "organizational_fit": "조직 적합성",
    "problem_solving": "문제 해결",
    "customer_journey_marketing": "고객 여정 마케팅",
    "md_data_analysis": "MD 데이터 분석",
    "seasonal_strategy_kpi": "시즌 전략 KPI",
    "stakeholder_collaboration": "이해관계자 협업",
    "value_chain_optimization": "가치사슬 최적화",
}


def analyze_agent_segment_usage(stage1_evidence: Dict, transcript: Dict) -> Dict:
    """
    Agent가 사용한 segment 분석

    Returns:
        {
            "competency_name": {
                "segments_used": [1, 3, 5, ...],
                "segment_count": int,
                "evidence_count": int,
                "score": float,
                "segment_details": [...]
            }
        }
    """

    analysis = {}

    # Transcript segment 매핑
    segments_by_id = {}
    for seg in transcript.get("segments", []):
        seg_id = seg["segment_id"]
        segments_by_id[seg_id] = {
            "segment_id": seg_id,
            "segment_order": seg.get("segment_order"),
            "question": seg.get("question_text", ""),
            "answer": seg.get("answer_text", ""),
            "turn_type": seg.get("turn_type"),
            "interviewer_name": seg.get("interviewer_name"),
        }

    # 역량별 분석
    for comp_name, comp_data in stage1_evidence.items():
        if not isinstance(comp_data, dict):
            continue

        if comp_name not in COMPETENCY_DISPLAY_NAMES:
            continue

        # perspectives에서 evidence_details 추출
        perspectives = comp_data.get("perspectives", {})
        evidence_details = perspectives.get("evidence_details", [])

        # Segment IDs 추출
        segments_used = set()
        segment_details = []

        for ev in evidence_details:
            seg_id = ev.get("segment_id")
            if seg_id and seg_id in segments_by_id:
                segments_used.add(seg_id)

                seg_info = segments_by_id[seg_id]
                segment_details.append({
                    "segment_id": seg_id,
                    "segment_order": seg_info["segment_order"],
                    "question": seg_info["question"][:80],
                    "answer": seg_info["answer"][:120],
                    "relevance_note": ev.get("relevance_note", ""),
                    "impact": ev.get("impact", ""),
                })

        analysis[comp_name] = {
            "segments_used": sorted(list(segments_used)),
            "segment_count": len(segments_used),
            "evidence_count": len(evidence_details),
            "score": comp_data.get("overall_score", 0),
            "confidence": comp_data.get("confidence", {}),
            "segment_details": segment_details
        }

    return analysis


def calculate_segment_overlap(analysis: Dict) -> Dict:
    """역량 간 segment 중복 분석"""

    # Segment별로 사용한 역량 매핑
    segment_to_competencies = defaultdict(set)

    for comp_name, comp_analysis in analysis.items():
        for seg_id in comp_analysis["segments_used"]:
            segment_to_competencies[seg_id].add(comp_name)

    # 중복도 계산
    overlap_stats = {
        "total_unique_segments": len(segment_to_competencies),
        "segments_by_usage_count": defaultdict(list),
        "highly_shared_segments": []  # 3개 이상 역량에서 사용
    }

    for seg_id, competencies in segment_to_competencies.items():
        usage_count = len(competencies)
        overlap_stats["segments_by_usage_count"][usage_count].append({
            "segment_id": seg_id,
            "competencies": sorted(list(competencies)),
            "usage_count": usage_count
        })

        if usage_count >= 3:
            overlap_stats["highly_shared_segments"].append({
                "segment_id": seg_id,
                "competencies": sorted(list(competencies)),
                "usage_count": usage_count
            })

    return overlap_stats


def simulate_rag_effect(analysis: Dict, transcript: Dict) -> Dict:
    """
    RAG 도입 시 예상 효과 시뮬레이션

    가정:
    - RAG는 역량별로 관련성 높은 Top 5-8 segments만 선택
    - Agent는 전체 transcript를 받아서 평가
    """

    total_segments = len(transcript.get("segments", []))

    simulation = {
        "current_approach": {
            "method": "전체 Transcript를 10개 Agent에게 전송",
            "tokens_per_agent": total_segments * 300,  # segment당 평균 300 tokens
            "total_tokens": total_segments * 300 * 10,
            "cost_estimate_usd": (total_segments * 300 * 10) / 1000 * 0.005  # GPT-4o input
        },
        "rag_approach": {
            "method": "역량별 Top 5-8 segments만 선택",
            "avg_segments_per_agent": 0,
            "tokens_per_agent": 0,
            "total_tokens": 0,
            "cost_estimate_usd": 0,
            "savings_rate": 0
        }
    }

    # 역량별 평균 segment 사용량
    total_segments_used = sum(comp["segment_count"] for comp in analysis.values())
    avg_segments = total_segments_used / len(analysis)

    simulation["rag_approach"]["avg_segments_per_agent"] = avg_segments
    simulation["rag_approach"]["tokens_per_agent"] = int(avg_segments * 300)
    simulation["rag_approach"]["total_tokens"] = int(avg_segments * 300 * 10)
    simulation["rag_approach"]["cost_estimate_usd"] = (avg_segments * 300 * 10) / 1000 * 0.005

    # 절감률
    current_cost = simulation["current_approach"]["cost_estimate_usd"]
    rag_cost = simulation["rag_approach"]["cost_estimate_usd"]
    savings_rate = (current_cost - rag_cost) / current_cost if current_cost > 0 else 0

    simulation["rag_approach"]["savings_rate"] = savings_rate

    return simulation


def main():
    """메인 실행"""

    print("\n" + "="*80)
    print("  Agent Segment 사용 패턴 분석")
    print("="*80)

    # 데이터 로드
    transcript_path = Path(__file__).parent.parent / "test_data" / "transcript_jiwon_101.json"
    evaluation_result_path = Path(__file__).parent.parent / "test_data" / "evaluation_result_jiwon_test.json"

    print(f"\n  데이터 로드:")
    print(f"  - Transcript: {transcript_path}")
    print(f"  - Evaluation Result: {evaluation_result_path}")

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    with open(evaluation_result_path, "r", encoding="utf-8") as f:
        evaluation_result = json.load(f)

    stage1_evidence = evaluation_result.get("competency_results", {})

    print(f"  - Transcript: {len(transcript['segments'])}개 segment")
    print(f"  - 평가 역량: {len([k for k in stage1_evidence.keys() if k in COMPETENCY_DISPLAY_NAMES])}개")


    # Step 1: Agent segment 사용 분석
    print("\n" + "="*80)
    print("[Step 1] 역량별 Segment 사용 패턴")
    print("="*80)

    analysis = analyze_agent_segment_usage(stage1_evidence, transcript)

    for comp_name, comp_analysis in analysis.items():
        comp_display = COMPETENCY_DISPLAY_NAMES.get(comp_name, comp_name)

        print(f"\n[{comp_display}]")
        print(f"  사용 segments: {comp_analysis['segments_used']} ({comp_analysis['segment_count']}개)")
        print(f"  평가 점수: {comp_analysis['score']}점")

        # 상위 3개 segment 상세
        if comp_analysis['segment_details']:
            print(f"  주요 근거:")
            for detail in comp_analysis['segment_details'][:3]:
                print(f"    • Segment {detail['segment_id']}: {detail['question'][:50]}...")
                print(f"      → {detail['answer'][:60]}...")


    # Step 2: Segment 중복도 분석
    print("\n" + "="*80)
    print("[Step 2] Segment 중복 사용 분석")
    print("="*80)

    overlap_stats = calculate_segment_overlap(analysis)

    print(f"\n  총 사용된 고유 Segment: {overlap_stats['total_unique_segments']}개")
    print(f"  (전체 {len(transcript['segments'])}개 중 {overlap_stats['total_unique_segments']}개 사용)")

    print(f"\n  Segment 사용 빈도:")
    for usage_count in sorted(overlap_stats['segments_by_usage_count'].keys(), reverse=True):
        segments = overlap_stats['segments_by_usage_count'][usage_count]
        print(f"    {usage_count}개 역량에서 사용: {len(segments)}개 segment")

    if overlap_stats['highly_shared_segments']:
        print(f"\n  ⚠️  3개 이상 역량에서 공유되는 Segment ({len(overlap_stats['highly_shared_segments'])}개):")
        for seg_info in overlap_stats['highly_shared_segments'][:5]:
            comp_names = [COMPETENCY_DISPLAY_NAMES.get(c, c) for c in seg_info['competencies']]
            print(f"    • Segment {seg_info['segment_id']}: {seg_info['usage_count']}개 역량")
            print(f"      → {', '.join(comp_names)}")


    # Step 3: RAG 효과 시뮬레이션
    print("\n" + "="*80)
    print("[Step 3] RAG 도입 시 예상 효과")
    print("="*80)

    simulation = simulate_rag_effect(analysis, transcript)

    print(f"\n  [현재 방식]")
    print(f"  - 방법: {simulation['current_approach']['method']}")
    print(f"  - Agent당 tokens: {simulation['current_approach']['tokens_per_agent']:,}")
    print(f"  - 총 tokens: {simulation['current_approach']['total_tokens']:,}")
    print(f"  - 예상 비용: ${simulation['current_approach']['cost_estimate_usd']:.4f}")

    print(f"\n  [RAG 방식]")
    print(f"  - 방법: {simulation['rag_approach']['method']}")
    print(f"  - Agent당 평균 segments: {simulation['rag_approach']['avg_segments_per_agent']:.1f}개")
    print(f"  - Agent당 tokens: {simulation['rag_approach']['tokens_per_agent']:,}")
    print(f"  - 총 tokens: {simulation['rag_approach']['total_tokens']:,}")
    print(f"  - 예상 비용: ${simulation['rag_approach']['cost_estimate_usd']:.4f}")

    savings_rate = simulation['rag_approach']['savings_rate']
    print(f"\n  ✅ 예상 절감:")
    print(f"     Token: {savings_rate*100:.1f}% 절감")
    print(f"     비용: ${simulation['current_approach']['cost_estimate_usd'] - simulation['rag_approach']['cost_estimate_usd']:.4f} 절감")


    # Step 4: RAG 도입 제안
    print("\n" + "="*80)
    print("[Step 4] RAG 도입 분석 결과")
    print("="*80)

    print(f"\n  📊 현황 분석:")
    print(f"     - 전체 Segment: {len(transcript['segments'])}개")
    print(f"     - Agent가 실제 사용한 평균 Segment: {simulation['rag_approach']['avg_segments_per_agent']:.1f}개")
    print(f"     - 사용률: {simulation['rag_approach']['avg_segments_per_agent'] / len(transcript['segments']) * 100:.1f}%")

    print(f"\n  💡 인사이트:")
    print(f"     1. Agent는 전체의 {simulation['rag_approach']['avg_segments_per_agent'] / len(transcript['segments']) * 100:.1f}%만 실제로 사용")
    print(f"     2. 나머지 {100 - simulation['rag_approach']['avg_segments_per_agent'] / len(transcript['segments']) * 100:.1f}%는 노이즈")
    print(f"     3. RAG로 관련 segment만 선별하면 {savings_rate*100:.1f}% 절감 가능")

    print(f"\n  ✅ 결론:")
    if savings_rate >= 0.5:
        print(f"     RAG 도입 **강력 추천** (비용 {savings_rate*100:.1f}% 절감 + 정확도 향상 기대)")
    elif savings_rate >= 0.3:
        print(f"     RAG 도입 추천 (비용 {savings_rate*100:.1f}% 절감)")
    else:
        print(f"     RAG 효과 제한적 (비용 {savings_rate*100:.1f}% 절감)")


    # 결과 저장
    output_path = Path(__file__).parent.parent / "test_data" / "agent_segment_analysis.json"

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "transcript_file": "transcript_jiwon_101.json",
        "total_segments": len(transcript["segments"]),
        "competency_analysis": analysis,
        "overlap_stats": {
            "total_unique_segments": overlap_stats["total_unique_segments"],
            "usage_distribution": {
                str(k): len(v) for k, v in overlap_stats["segments_by_usage_count"].items()
            },
            "highly_shared_count": len(overlap_stats["highly_shared_segments"])
        },
        "simulation": simulation
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n  결과 저장: {output_path}")

    print("\n" + "="*80)
    print("  분석 완료!")
    print("="*80)


if __name__ == "__main__":
    main()
