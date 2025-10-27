"""
간단한 MAS 검증 테스트 (Stage 검증만)
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
    간단한 검증 테스트 실행
    """
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

    print("\n" + "="*80)
    print("MAS 아키텍처 간단 검증 테스트")
    print("="*80)

    # Stage 검증만 실행
    stage_validation_result = await validator.run_stage_validation(
        transcript=transcript,
        resume=resume,
        weights=weights
    )

    # 결과 저장
    output_path = test_data_dir / f"mas_stage_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stage_validation_result, f, ensure_ascii=False, indent=2)

    print(f"\n결과 파일: {output_path}")

    if stage_validation_result["passed"]:
        print("\n✅ Stage 검증 통과!")
    else:
        print("\n❌ Stage 검증 실패")


if __name__ == "__main__":
    asyncio.run(main())
