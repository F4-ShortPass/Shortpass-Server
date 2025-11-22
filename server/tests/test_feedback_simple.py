"""
간단한 피드백 RAG 테스트
목적: 피드백 저장 → 검색 → 재평가 플로우 검증
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.orm import Session
from openai import AsyncOpenAI
from db.database import get_db, engine
from services.feedback.feedback_manager import FeedbackManager
from models.feedback_memory import FeedbackMemory
from sqlalchemy import text
import os

# OpenAI 클라이언트
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def test_feedback_accumulation():
    """
    시나리오: 동일한 실수가 반복되는 상황을 시뮬레이션
    """
    print("=" * 80)
    print("피드백 누적 효과 테스트")
    print("=" * 80)

    db: Session = next(get_db())

    try:
        # pgvector extension 확인
        result = db.execute(text("SELECT * FROM pg_extension WHERE extname = 'vector'"))
        if not result.fetchone():
            print("⚠️  pgvector extension이 없습니다. 설치 필요:")
            print("   CREATE EXTENSION vector;")
            return

        print("✅ pgvector extension 활성화됨\n")

        feedback_manager = FeedbackManager(db=db, openai_client=openai_client)

        # ========================================
        # Step 1: 첫 번째 피드백 저장
        # ========================================
        print("[Step 1] 첫 번째 피드백 저장")
        print("-" * 80)

        feedback_1 = await feedback_manager.save_feedback(
            job_category="Sales",
            competency_name="interpersonal_skill",
            ai_score=70,
            ai_reasoning="지원자가 공격적인 말투를 사용하여 대인관계 역량이 부족해 보임",
            human_score=90,
            human_reasoning="영업 직무에서는 공격성이 아니라 적극성으로 해석해야 함. 고객 설득력이 뛰어남",
            use_llm_summary=False  # V2: 빠른 저장
        )

        print(f"✅ 피드백 #{feedback_1.id} 저장 완료")
        print(f"   - Mistake: {feedback_1.mistake_summary}")
        print(f"   - Correction: {feedback_1.correction_guideline[:100]}...")
        print()

        # ========================================
        # Step 2: 비슷한 상황의 두 번째 피드백 저장
        # ========================================
        print("[Step 2] 유사 상황의 두 번째 피드백 저장")
        print("-" * 80)

        feedback_2 = await feedback_manager.save_feedback(
            job_category="Sales",
            competency_name="interpersonal_skill",
            ai_score=65,
            ai_reasoning="지원자가 강압적인 태도로 고객을 대하는 것처럼 보임",
            human_score=85,
            human_reasoning="영업에서는 강한 어조가 자신감과 주도성을 의미함. 고객 니즈 파악 능력 우수",
            use_llm_summary=False
        )

        print(f"✅ 피드백 #{feedback_2.id} 저장 완료")
        print(f"   - Mistake: {feedback_2.mistake_summary}")
        print()

        # ========================================
        # Step 3: 세 번째 유사 케이스 (다른 역량)
        # ========================================
        print("[Step 3] 다른 역량의 피드백 저장")
        print("-" * 80)

        feedback_3 = await feedback_manager.save_feedback(
            job_category="Sales",
            competency_name="problem_solving",
            ai_score=75,
            ai_reasoning="지원자가 체계적인 문제 해결 프로세스를 설명하지 못함",
            human_score=80,
            human_reasoning="영업에서는 즉각적 대응이 중요. 현장 경험 기반의 직관적 해결 능력을 높게 평가",
            use_llm_summary=False
        )

        print(f"✅ 피드백 #{feedback_3.id} 저장 완료")
        print()

        # ========================================
        # Step 4: 유사 상황 검색 (interpersonal_skill)
        # ========================================
        print("[Step 4] 유사 피드백 검색 - interpersonal_skill")
        print("-" * 80)

        current_context = "지원자가 적극적이고 단호한 말투로 고객 응대 사례를 설명함"

        relevant_feedbacks = await feedback_manager.get_relevant_feedback(
            job_category="Sales",
            competency_name="interpersonal_skill",
            current_context=current_context,
            top_k=3,
            similarity_threshold=0.5,
            use_dynamic_threshold=True
        )

        print(f"🔍 검색 결과: {len(relevant_feedbacks)}개 발견")
        for i, fb in enumerate(relevant_feedbacks, 1):
            print(f"\n  [{i}] 유사도: {fb['similarity']:.2%}")
            print(f"      Mistake: {fb['mistake_summary']}")
            print(f"      Correction: {fb['correction_guideline'][:80]}...")
            print(f"      Score Change: {fb['ai_score']} → {fb['human_score']} (차이: +{fb['human_score'] - fb['ai_score']})")

        print()

        # ========================================
        # Step 5: 통계 확인
        # ========================================
        print("[Step 5] 피드백 통계")
        print("-" * 80)

        total_count = db.query(FeedbackMemory).count()
        interpersonal_count = db.query(FeedbackMemory).filter(
            FeedbackMemory.competency_name == "interpersonal_skill"
        ).count()

        print(f"  총 피드백: {total_count}개")
        print(f"  interpersonal_skill: {interpersonal_count}개")

        # 평균 점수 차이
        avg_improvement = db.execute(text("""
            SELECT AVG(human_score - ai_score) as avg_diff
            FROM feedback_memory
            WHERE competency_name = 'interpersonal_skill'
        """)).scalar()

        print(f"  평균 점수 개선: +{avg_improvement:.1f}점")
        print()

        # ========================================
        # Step 6: Few-shot 효과 시뮬레이션
        # ========================================
        print("[Step 6] Few-shot Prompting 시뮬레이션")
        print("-" * 80)

        if relevant_feedbacks:
            print("📋 AI에게 주입될 Few-shot Examples:\n")
            for i, fb in enumerate(relevant_feedbacks[:2], 1):
                print(f"  Example {i}:")
                print(f"    User: 'AI는 {fb['ai_score']}점으로 평가했지만 수정됨'")
                print(f"    Assistant: '초기 평가 {fb['ai_score']}점. {fb['mistake_summary']}'")
                print(f"    User: 'HR 교정: {fb['correction_guideline'][:60]}... 실제 {fb['human_score']}점'")
                print(f"    Assistant: '이해했습니다. {fb['human_score']}점으로 조정합니다.'\n")

            print("  💡 기대 효과:")
            print(f"     - AI가 '강한 어조 = 부정적' 패턴을 학습함")
            print(f"     - 영업 직무에서는 '강한 어조 = 적극성'으로 재해석")
            print(f"     - 예상 점수 향상: +{sum(fb['human_score'] - fb['ai_score'] for fb in relevant_feedbacks) / len(relevant_feedbacks):.1f}점")

        print()
        print("=" * 80)
        print("✅ 테스트 완료!")
        print("=" * 80)
        print("\n💡 다음 단계:")
        print("   1. CompetencyAgent로 실제 평가 실행")
        print("   2. use_feedback=True로 재평가")
        print("   3. 점수 변화 확인")

    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(test_feedback_accumulation())
