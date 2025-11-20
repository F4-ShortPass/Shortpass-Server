"""
박서진 면접 평가 (transcript_박서진_102.json + resume_seojin.json)

Usage:
    python server/tests/evaluation_test_seojin.py
"""
import sys
import os
import json
import asyncio
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.evaluation.evaluation_service import EvaluationService


async def test_seojin_evaluation():
    """박서진 면접 평가"""

    # ========================================
    # 1. 테스트 데이터 로드
    # ========================================
    data_dir = Path(__file__).resolve().parent.parent / "test_data"

    transcript_path = data_dir / "transcript_박서진_102.json"
    resume_path = data_dir / "resume_seojin.json"

    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume file not found: {resume_path}")

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)
    with open(resume_path, "r", encoding="utf-8") as f:
        resume_data = json.load(f)

    # ========================================
    # 2. 가중치 설정 (삼성물산 패션부문 기준)
    # ========================================

    competency_weights = {
        # Common Competencies (5개)
        "achievement_motivation": 0.10,
        "growth_potential": 0.10,
        "interpersonal_skill": 0.10,
        "organizational_fit": 0.10,
        "problem_solving": 0.10,

        # Job Competencies (5개)
        "customer_journey_marketing": 0.15,
        "md_data_analysis": 0.15,
        "seasonal_strategy_kpi": 0.10,
        "stakeholder_collaboration": 0.05,
        "value_chain_optimization": 0.05,
    }

    total_weight = sum(competency_weights.values())
    if abs(total_weight - 1.0) > 0.01:
        print(f"Warning: 가중치 합계: {total_weight:.2f} (1.0이어야 함)")

    # ========================================
    # 3. 메타 정보 추출
    # ========================================

    interview_id = transcript.get("interview_id", 102)
    applicant_id = transcript.get("applicant_id", 102)
    job_id = transcript.get("job_id", 1)

    # ========================================
    # 4. 평가 실행
    # ========================================

    print("\n" + "=" * 80)
    print(" AI 면접 평가 시스템 - 박서진 평가")
    print("=" * 80 + "\n")

    print(f"  Applicant: {resume_data.get('applicant_name', '박서진')}")
    print(f"  Position: {resume_data.get('experience', [{}])[0].get('position', 'MD')}")
    print(f"  Company: {resume_data.get('experience', [{}])[0].get('company', '한섬')}")
    print(f"  Interview ID: {interview_id}")
    print(f"  Applicant ID: {applicant_id}")
    print(f"  Job ID: {job_id}")
    print(f"  Segments: {len(transcript.get('segments', []))}개")
    print(f"  가중치 합계: {total_weight:.2f}\n")

    service = EvaluationService()

    print(" 평가 시작...\n")

    try:
        result = await service.evaluate_interview(
            interview_id=interview_id,
            applicant_id=applicant_id,
            job_id=job_id,
            transcript=transcript,
            competency_weights=competency_weights,
            resume_data=resume_data
        )

        print("\n" + "=" * 80)
        print(" 평가 완료!")
        print("=" * 80)

        # ========================================
        # 결과 출력
        # ========================================

        final_score = result.get("final_score", 0)
        avg_confidence = result.get("avg_confidence", 0)
        final_reliability = result.get("final_reliability", "")
        reliability_note = result.get("reliability_note", "")

        print(f"\n  최종 점수: {final_score:.1f}점")
        print(f"  평균 Confidence: {avg_confidence:.2f}")
        print(f"  신뢰도 레벨: {final_reliability}")
        print(f"  신뢰도 설명: {reliability_note}")

        # 역량별 점수
        print("\n" + "-" * 80)
        print(" 역량별 점수")
        print("-" * 80)

        competency_details = result.get("competency_details", {})
        aggregated = result.get("aggregated_competencies", {})

        if competency_details:
            print("\n  [Common Competencies]")
            common_comps = ["achievement_motivation", "growth_potential", "interpersonal_skill",
                          "organizational_fit", "problem_solving"]
            for comp in common_comps:
                data = competency_details.get(comp, aggregated.get(comp, {}))
                if data:
                    score = data.get("overall_score", 0)
                    conf = data.get("confidence_v2", data.get("interview_confidence", 0))
                    print(f"    {comp}: {score:.1f}점 (conf: {conf:.2f})")

            print("\n  [Job Competencies]")
            job_comps = ["customer_journey_marketing", "md_data_analysis", "seasonal_strategy_kpi",
                        "stakeholder_collaboration", "value_chain_optimization"]
            for comp in job_comps:
                data = competency_details.get(comp, aggregated.get(comp, {}))
                if data:
                    score = data.get("overall_score", 0)
                    conf = data.get("confidence_v2", data.get("interview_confidence", 0))
                    print(f"    {comp}: {score:.1f}점 (conf: {conf:.2f})")

        # S3 URLs
        print("\n" + "-" * 80)
        print(" S3 저장 경로")
        print("-" * 80)
        print(f"\n  Transcript: {result.get('transcript_s3_url', 'N/A')}")
        print(f"  Stage1 Evidence: {result.get('stage1_evidence_s3_url', 'N/A')}")
        print(f"  Stage2 Aggregator: {result.get('stage2_aggregator_s3_url', 'N/A')}")
        print(f"  Stage3 Final: {result.get('stage3_final_integration_s3_url', 'N/A')}")
        print(f"  Stage4 Presentation: {result.get('stage4_presentation_s3_url', 'N/A')}")
        print(f"  Agent Logs: {result.get('agent_logs_s3_url', 'N/A')}")

        # Presentation 결과 (프론트엔드용)
        presentation = result.get("presentation_result", {})
        if presentation:
            print("\n" + "-" * 80)
            print(" Presentation 결과 (프론트엔드용)")
            print("-" * 80)

            if "overall_evaluation_summary" in presentation:
                print(f"\n  종합 심사평:")
                print(f"    {presentation['overall_evaluation_summary'][:200]}...")

            if "key_strengths" in presentation:
                print(f"\n  핵심 강점:")
                for s in presentation["key_strengths"][:3]:
                    print(f"    - {s}")

            if "key_weaknesses" in presentation:
                print(f"\n  핵심 약점:")
                for w in presentation["key_weaknesses"][:3]:
                    print(f"    - {w}")

        # 결과 JSON 저장
        output_path = data_dir / f"evaluation_result_{interview_id}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            # datetime 객체 제거
            clean_result = {k: v for k, v in result.items()
                          if not isinstance(v, (type(None.__class__),))}
            json.dump(clean_result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n  결과 저장: {output_path}")

        print("\n" + "=" * 80)
        print(" 테스트 완료")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n 오류 발생: {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_seojin_evaluation())
