"""
일관성 테스트 실행
같은 입력으로 3회 평가하여 결과 일관성 확인
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# 상위 디렉토리를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_mas_validation import MASValidator
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv()


async def main():
    """
    일관성 테스트 실행
    """
    print("\n" + "="*80)
    print("MAS 일관성 테스트 (3회 반복 평가)")
    print("="*80)

    # OpenAI 클라이언트 생성
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 테스트 데이터 로드
    test_data_dir = Path(__file__).parent.parent / "test_data"

    with open(test_data_dir / "transcript_jiwon_101.json", "r", encoding="utf-8") as f:
        transcript = json.load(f)

    with open(test_data_dir / "resume_jiwon.json", "r", encoding="utf-8") as f:
        resume = json.load(f)

    # weights는 transcript에서 추출
    weights = transcript.get("weights", {}).get("competency", {})

    # Validator 생성
    validator = MASValidator(openai_client=openai_client)

    # 일관성 테스트 실행 (3회 반복)
    consistency_result = await validator.run_consistency_test(
        transcript=transcript,
        resume=resume,
        weights=weights,
        iterations=3
    )

    # 결과 저장 (JSON serializable하게 변환)
    def make_serializable(obj):
        """객체를 JSON serializable하게 변환"""
        from datetime import datetime
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items() if k != "openai_client"}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        else:
            return obj

    serializable_result = make_serializable(consistency_result)

    output_path = test_data_dir / f"mas_consistency_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_result, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n결과 파일: {output_path}")

    # 결과 요약 출력
    print("\n" + "="*80)
    print("일관성 테스트 결과 요약")
    print("="*80)

    analysis = consistency_result["analysis"]

    print(f"\n[최종 점수 일관성]")
    print(f"  평균: {analysis['final_score_mean']:.2f}점")
    print(f"  표준편차: {analysis['final_score_std']:.2f}점")
    print(f"  범위: {analysis['final_score_min']:.2f} ~ {analysis['final_score_max']:.2f}점")
    print(f"  편차: ±{analysis['final_score_std']:.2f}점")

    # 허용 범위 체크
    passed = analysis['final_score_std'] <= 5.0
    print(f"\n  검증 결과: {'✅ PASS' if passed else '❌ FAIL'} (허용 범위: ±5.0점)")

    print(f"\n[역량별 점수 일관성]")
    print(f"{'역량':<40} {'평균':>8} {'표준편차':>10} {'상태':>8}")
    print("-" * 80)

    for comp, stats in analysis['competency_scores'].items():
        comp_passed = stats['std'] <= 10.0
        status = "✅" if comp_passed else "❌"
        print(f"{comp:<40} {stats['mean']:>8.2f} ±{stats['std']:>9.2f} {status:>8}")

    print("\n" + "="*80)

    if consistency_result["passed"]:
        print("✅ 일관성 테스트 통과!")
        print("   → MAS 아키텍처가 일관된 평가 결과를 생성합니다.")
    else:
        print("❌ 일관성 테스트 실패")
        print("   → 평가 결과에 큰 편차가 있습니다. 원인 분석이 필요합니다.")

    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
