"""
일관성 테스트 결과 요약 출력
"""
import json
from pathlib import Path

test_data_dir = Path(__file__).parent.parent / "test_data"

# 가장 최근 결과 파일 찾기
result_files = list(test_data_dir.glob("mas_consistency_test_*.json"))
if not result_files:
    print("❌ 결과 파일을 찾을 수 없습니다.")
    print("   먼저 run_consistency_test.py를 실행하세요.")
    exit(1)

latest_file = max(result_files, key=lambda p: p.stat().st_mtime)

print(f"\n최신 결과 파일: {latest_file.name}")
print("="*80)

with open(latest_file, "r", encoding="utf-8") as f:
    result = json.load(f)

analysis = result["analysis"]

print(f"\n✅ 일관성 테스트 결과 ({'PASS' if result['passed'] else 'FAIL'})")
print("="*80)

print(f"\n[최종 점수 일관성]")
print(f"  반복 횟수: {result['iterations']}회")
print(f"  평균: {analysis['final_score_mean']:.2f}점")
print(f"  표준편차: {analysis['final_score_std']:.2f}점")
print(f"  범위: {analysis['final_score_min']:.2f} ~ {analysis['final_score_max']:.2f}점")
print(f"  편차: ±{analysis['final_score_std']:.2f}점")

print(f"\n[역량별 점수 일관성]")
print(f"{'역량':<40} {'평균':>8} {'표준편차':>10} {'상태':>8}")
print("-" * 80)

for comp, stats in analysis['competency_scores'].items():
    comp_passed = stats['std'] <= 10.0
    status = "✅" if comp_passed else "❌"
    print(f"{comp:<40} {stats['mean']:>8.2f} ±{stats['std']:>9.2f} {status:>8}")

print("\n" + "="*80)
print(f"\n{'✅ 검증 통과' if result['passed'] else '❌ 검증 실패'}")
print(f"  허용 기준: 최종 점수 표준편차 ≤ 5.0점")
print(f"  실제 결과: {analysis['final_score_std']:.2f}점")
print("\n" + "="*80)
