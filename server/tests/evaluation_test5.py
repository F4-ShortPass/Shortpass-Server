"""
새로운 평가 시스템 테스트 (Stage 1-3)
- Stage 1: 10개 Agent 병렬 평가
- Stage 2: Aggregator (Resume + Confidence V2 + Overlap + Cross-Comp)
- Stage 3: Final Integration (+ 선택적 Collaboration)
"""
import sys
import os
import json
import asyncio
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.evaluation.evaluation_service import EvaluationService


async def test_full_evaluation():
    """전체 평가 시스템 테스트 (새 아키텍처)"""

    # ========================================
    # 1. 테스트 데이터 로드
    # ========================================
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
        weights_data = json.load(f)

    # ========================================
    # 2. 가중치 통합 (Job + Common → 10개 역량)
    # ========================================
    
    # 기존 weights_sample.json 구조:
    # {
    #   "job_weights": {"structured_thinking": 0.25, ...},
    #   "common_weights": {"problem_solving": 0.25, ...}
    # }
    
    # 새 구조로 변환 (10개 역량 가중치)
    competency_weights = {
        # Common Competencies (5개)
        "achievement_motivation": 0.12,
        "growth_potential": 0.10,
        "interpersonal_skill": 0.08,
        "organizational_fit": 0.10,
        "problem_solving": 0.10,
        
        # Job Competencies (5개)
        "customer_journey_marketing": 0.12,
        "md_data_analysis": 0.12,
        "seasonal_strategy_kpi": 0.10,
        "stakeholder_collaboration": 0.08,
        "value_chain_optimization": 0.08,
    }
    
    # 가중치 합계 검증
    total_weight = sum(competency_weights.values())
    if abs(total_weight - 1.0) > 0.01:
        print(f"⚠️ 가중치 합계: {total_weight:.2f} (1.0이어야 함)")

    # ========================================
    # 3. Resume Mock 데이터 생성
    # ========================================
    
    resume_data = {
        "education": [
            {
                "institution": "서울대학교",
                "degree": "학사",
                "major": "경영학",
                "period": "2018.03 - 2022.02",
                "gpa": "3.8/4.5"
            }
        ],
        "experience": [
            {
                "company": "패션 리테일 회사",
                "position": "MD 인턴",
                "period": "2021.07 - 2021.08",
                "description": "시즌 상품 기획 보조, 매출 데이터 분석, 원가/마진 시뮬레이션, VMD 제안서 작성"
            },
            {
                "company": "패션 큐레이션 스타트업",
                "position": "마케팅/데이터 인턴",
                "period": "2020.12 - 2021.02",
                "description": "SNS 콘텐츠 기획, 인스타그램 운영, 고객 반응 데이터 수집 및 대시보드 정리"
            }
        ],
        "projects": [
            {
                "name": "학부생 창업 프로젝트 (패션 큐레이션 플랫폼)",
                "period": "2020.03 - 2021.02",
                "role": "팀장",
                "description": "5인 팀 리드, 스타일 추천 콘텐츠 기획, 팀원 일정 관리 및 갈등 조율 경험"
            },
            {
                "name": "리테일 데이터 분석 공모전",
                "period": "2021.09 - 2021.11",
                "role": "데이터 분석 담당",
                "description": "매출/트렌드 데이터로 시즌 기획안 제시, KPI(전환율, 재구매율) 기반 개선안 도출"
            }
        ],
        "skills": [
            "Excel (상) - 피벗/그래프, 비용-마진 시뮬레이션",
            "PowerPoint (상) - 제안서/리포트 작성",
            "Python (중) - pandas로 매출 데이터 분석",
            "SQL (중) - 기본 질의로 고객 행동 분석"
        ],
        "awards": [
            {
                "name": "리테일 데이터 분석 공모전 우수상",
                "date": "2021.11",
                "organization": "한국데이터산업진흥원"
            },
            {
                "name": "대학생 창업 공모전 대상",
                "date": "2021.11",
                "organization": "중소벤처기업부"
            }
        ],
        "certifications": [
            {
                "name": "유통관리사 2급",
                "date": "2021.06",
                "organization": "대한상공회의소"
            }
        ]
    }

    # ========================================
    # 4. 메타 정보 추출
    # ========================================
    
    def _to_int(value, default):
        try:
            if isinstance(value, str) and value.isdigit():
                return int(value)
            if isinstance(value, (int, float)):
                return int(value)
        except (ValueError, TypeError, AttributeError):
            pass
        return default

    meta = transcript.get("metadata", {})
    interview_raw = meta.get("interview_id") or transcript.get("interview_id") or 1
    applicant_raw = meta.get("applicant_id") or transcript.get("applicant_id") or transcript.get("candidate_id") or 100
    job_raw = meta.get("job_id") or transcript.get("job_id") or 200
    interview_id = _to_int(interview_raw, 1)
    applicant_id = _to_int(applicant_raw, 100)
    job_id = _to_int(job_raw, 200)

    # ========================================
    # 5. 평가 실행
    # ========================================
    
    print("\n" + "=" * 80)
    print(" AI 면접 평가 시스템 - 새 아키텍처 테스트 (Stage 1-3)")
    print("=" * 80 + "\n")

    service = EvaluationService()

    print(" 평가 시작...\n")
    print(f" Interview ID: {interview_id}")
    print(f" Applicant ID: {applicant_id}")
    print(f" Job ID: {job_id}")
    print(f" Resume 제공: {'Yes' if resume_data else 'No'}")
    print(f" 10개 역량 가중치 합계: {total_weight:.2f}\n")

    try:
        result = await service.evaluate_interview(
            interview_id=interview_id,
            applicant_id=applicant_id,
            job_id=job_id,
            transcript=transcript,
            competency_weights=competency_weights,
            resume_data=resume_data  # 🆕 Resume 추가
        )

        print("\n✅ 평가 완료!\n")

        # ========================================
        # Stage 1 결과 출력
        # ========================================
        print("=" * 80)
        print(" Stage 1: 10개 역량 개별 평가 결과")
        print("=" * 80)

        competency_details = result.get("competency_details", {})
        
        if competency_details:
            print("\n" + "-" * 80)
            print(" Common Competencies (5개)")
            print("-" * 80)
            
            common_comps = [
                "achievement_motivation",
                "growth_potential",
                "interpersonal_skill",
                "organizational_fit",
                "problem_solving"
            ]
            
            for comp_name in common_comps:
                comp_data = competency_details.get(comp_name)
                if comp_data:
                    score = comp_data.get("overall_score", 0)
                    conf_v2 = comp_data.get("confidence_v2", 0)
                    weight = comp_data.get("weight", 0)
                    contribution = comp_data.get("weighted_contribution", 0)
                    resume_verified = comp_data.get("resume_verified_count", 0)
                    
                    print(f"  ├─ [{comp_name}]")
                    print(f"  │   점수: {score:.1f}점 | Conf V2: {conf_v2:.2f} | 가중치: {weight:.2f} | 기여: {contribution:.2f}")
                    print(f"  │   Resume 검증: {resume_verified}개")

            print("\n" + "-" * 80)
            print(" Job Competencies (5개)")
            print("-" * 80)
            
            job_comps = [
                "customer_journey_marketing",
                "md_data_analysis",
                "seasonal_strategy_kpi",
                "stakeholder_collaboration",
                "value_chain_optimization"
            ]
            
            for comp_name in job_comps:
                comp_data = competency_details.get(comp_name)
                if comp_data:
                    score = comp_data.get("overall_score", 0)
                    conf_v2 = comp_data.get("confidence_v2", 0)
                    weight = comp_data.get("weight", 0)
                    contribution = comp_data.get("weighted_contribution", 0)
                    resume_verified = comp_data.get("resume_verified_count", 0)
                    
                    print(f"  ├─ [{comp_name}]")
                    print(f"  │   점수: {score:.1f}점 | Conf V2: {conf_v2:.2f} | 가중치: {weight:.2f} | 기여: {contribution:.2f}")
                    print(f"  │   Resume 검증: {resume_verified}개")

        # ========================================
        # Stage 2 결과 출력
        # ========================================
        print("\n" + "=" * 80)
        print(" Stage 2: Aggregator 결과")
        print("=" * 80)

        low_conf_summary = result.get("low_confidence_summary", {})
        collab_summary = result.get("collaboration_summary", {})
        overlap_logs = result.get("segment_overlap_adjustments", [])
        cross_flags = result.get("cross_competency_flags", [])
        segment_evals = result.get("segment_evaluations_with_resume", [])
        exec_logs = result.get("execution_logs", [])
        agg_comps = result.get("aggregated_competencies", {})
        
        low_conf_count = low_conf_summary.get("total_low_confidence", 0)
        low_conf_comps = low_conf_summary.get("competencies", [])
        
        print(f"\n  Low Confidence 역량: {low_conf_count}개")
        if low_conf_comps:
            for comp in low_conf_comps:
                print(f"    ├─ {comp}")
        else:
            print("    ✅ 모든 역량 신뢰도 양호")
        
        # 세부: Resume 검증 + Confidence V2 재계산 결과 (상위 5개만 샘플)
        if segment_evals:
            print("\n  Segment 검증/Confidence 샘플 (상위 5개):")
            for seg in segment_evals[:5]:
                print(
                    f"    - [{seg.get('competency')}] seg#{seg.get('segment_id')} | "
                    f"score={seg.get('score', 0)} | confV2={seg.get('confidence_v2', seg.get('interview_confidence', 0)):.2f} | "
                    f"resume_verified={seg.get('resume_verified')}({seg.get('verification_strength')})"
                )
        else:
            print("\n  Segment 검증/Confidence 데이터 없음")

        # 세부: Segment Overlap 조정 로그
        print("\n  Segment Overlap 조정 로그:")
        if overlap_logs:
            for adj in overlap_logs:
                print(f"    - Segment {adj.get('segment_id')} | type={adj.get('adjustment_type')} | gap={adj.get('score_gap')}")
                for a in adj.get("adjustments", [])[:5]:
                    print(
                        f"       • {a.get('competency')}: {a.get('original_score')} → {a.get('adjusted_score')} "
                        f"(conf {a.get('original_confidence')}→{a.get('adjusted_confidence')})"
                    )
                if len(adj.get("adjustments", [])) > 5:
                    print("       • ...")
        else:
            print("    (조정 없음)")

        # 세부: Cross-Competency 플래그
        print("\n  Cross-Competency Flags:")
        if cross_flags:
            for flag in cross_flags:
                print(
                    f"    - {flag.get('competency')} | {flag.get('flag_type')} | confV2={flag.get('confidence_v2')}"
                )
        else:
            print("    (플래그 없음)")

        # ========================================
        # Stage 3 결과 출력 (Collaboration - 선택적)
        # ========================================
        collab_count = collab_summary.get("total_collaborations", 0)
        
        if collab_count > 0:
            print("\n" + "=" * 80)
            print(" Stage 3: Collaboration 결과 (선택적)")
            print("=" * 80)
            
            collab_comps = collab_summary.get("competencies_adjusted", [])
            adjustments = collab_summary.get("adjustments", [])
            
            print(f"\n  협업 처리: {collab_count}건")
            print(f"  조정된 역량: {', '.join(collab_comps)}")
            
            if adjustments:
                for adj in adjustments:
                    comp = adj.get("competency")
                    orig = adj.get("original_score", 0)
                    adjusted = adj.get("adjusted_score", 0)
                    reason = adj.get("reason", "")
                    print(f"    ├─ {comp}: {orig}점 → {adjusted}점 ({reason})")

        # ========================================
        # Final 결과 출력
        # ========================================
        print("\n" + "=" * 80)
        print(" Final: 최종 통합 결과")
        print("=" * 80)

        final_score = result.get("final_score", 0)
        avg_confidence = result.get("avg_confidence", 0)
        final_reliability = result.get("final_reliability", "")
        reliability_note = result.get("reliability_note", "")

        print(f"\n  🎯 최종 점수: {final_score:.1f}점")
        print(f"  📊 평균 Confidence V2: {avg_confidence:.2f}")
        print(f"  🔒 신뢰도 레벨: {final_reliability}")
        print(f"  📝 신뢰도 설명:")
        print(f"     {reliability_note}")

        # ========================================
        # 상세 역량별 출력 (샘플: problem_solving)
        # ========================================
        print("\n" + "=" * 80)
        print(" 상세 역량별 평가 (샘플: problem_solving)")
        print("=" * 80)

        ps_data = competency_details.get("problem_solving")
        if ps_data:
            print(f"\n  역량명: problem_solving")
            print(f"  최종 점수: {ps_data.get('overall_score', 0):.1f}점")
            print(f"  Interview Confidence: {ps_data.get('interview_confidence', 0):.2f}")
            print(f"  Confidence V2: {ps_data.get('confidence_v2', 0):.2f}")
            print(f"  가중치: {ps_data.get('weight', 0):.2f}")
            print(f"  기여도: {ps_data.get('weighted_contribution', 0):.2f}")
            print(f"  Segment 수: {ps_data.get('segment_count', 0)}개")
            print(f"  Resume 검증: {ps_data.get('resume_verified_count', 0)}개")
            
            strengths = ps_data.get("strengths", [])
            if strengths:
                print(f"\n  💪 강점 ({len(strengths)}개):")
                for i, s in enumerate(strengths[:3], 1):
                    print(f"    {i}. {s}")
            
            weaknesses = ps_data.get("weaknesses", [])
            if weaknesses:
                print(f"\n  ⚠️  약점 ({len(weaknesses)}개):")
                for i, w in enumerate(weaknesses[:3], 1):
                    print(f"    {i}. {w}")

        # ========================================
        # 실행 로그 출력
        # ========================================
        print("\n" + "=" * 80)
        print(" 실행 로그 (Performance)")
        print("=" * 80 + "\n")
        
        if exec_logs:
            print(f"  총 로그 이벤트: {len(exec_logs)}개 (상위 10개만 표시)")
            for log in exec_logs[:10]:
                node = log.get("node", "unknown")
                step = log.get("step", "")
                note = log.get("note", log.get("message", ""))
                print(f"    - [{node}] {step} {note}")
            if len(exec_logs) > 10:
                print("    ... (생략)")
        else:
            print("  (로그 없음 또는 S3 전용)")

        # ========================================
        # 최종 요약
        # ========================================
        print("\n" + "=" * 80)
        print(" 테스트 완료 ✅")
        print("=" * 80)
        print(f"\n  요약:")
        print(f"    ├─ 최종 점수: {final_score:.1f}점")
        print(f"    ├─ 평균 Confidence V2: {avg_confidence:.2f}")
        print(f"    ├─ 신뢰도: {final_reliability}")
        print(f"    ├─ Low Confidence: {low_conf_count}개")
        print(f"    └─ Collaboration: {collab_count}건")
        print("\n" + "=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_full_evaluation())
