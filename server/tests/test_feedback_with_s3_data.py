"""
S3 실제 평가 데이터로 피드백 RAG 테스트
목적: 박서진(102번) 평가 결과 활용 → 피드백 저장 → 재평가
"""
import asyncio
import json
import boto3
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.orm import Session
from openai import AsyncOpenAI
from db.database import get_db
from services.feedback.feedback_manager import FeedbackManager
from ai.agents.competency_agent import CompetencyAgent
import os

# AWS S3
s3_client = boto3.client('s3', region_name=os.getenv('AWS_REGION', 'us-east-1'))
BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'linkbig-ht-06-f4')

# OpenAI
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def load_s3_evaluation(interview_id: int, timestamp: str):
    """S3에서 평가 결과 로드"""
    try:
        # Stage 1 Evidence 로드
        key = f"evaluations/{interview_id}/{timestamp}/stage1_evidence.json"
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        stage1_data = json.loads(response['Body'].read().decode('utf-8'))

        # Stage 4 Presentation (프론트엔드 포맷)
        key = f"evaluations/{interview_id}/{timestamp}/stage4_presentation_frontend.json"
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        stage4_data = json.loads(response['Body'].read().decode('utf-8'))

        return stage1_data, stage4_data

    except Exception as e:
        print(f"❌ S3 로드 실패: {e}")
        return None, None


async def test_feedback_with_real_data():
    """
    시나리오:
    1. S3에서 102번 평가 결과 로드
    2. 특정 역량의 점수를 "수정"하는 피드백 저장
    3. 동일한 트랜스크립트로 재평가 (use_feedback=True)
    4. 점수 변화 확인
    """
    print("=" * 80)
    print("S3 실제 데이터 기반 피드백 RAG 테스트")
    print("=" * 80)

    db: Session = next(get_db())

    try:
        # ========================================
        # Step 1: S3에서 102번 평가 결과 로드
        # ========================================
        print("\n[Step 1] S3에서 102번 평가 결과 로드")
        print("-" * 80)

        interview_id = 102
        timestamp = "20251124T083236"  # 102번의 실제 타임스탬프 (TEST_SUMMARY.md 참조)

        stage1_data, stage4_data = await load_s3_evaluation(interview_id, timestamp)

        if not stage1_data:
            print("⚠️  S3 데이터 로드 실패. 로컬 테스트 데이터를 사용하거나 timestamp를 확인하세요.")
            return

        print(f"✅ Stage 1 Evidence 로드 완료")

        # 첫 번째 역량 선택 (interpersonal_skill - 점수 82점으로 높음)
        competency_name = "interpersonal_skill"
        competency_data = stage1_data.get(competency_name, {})

        if not competency_data:
            print("⚠️  선택한 역량 데이터가 없습니다.")
            return

        original_score = competency_data.get('overall_score', 0)
        strengths = competency_data.get('strengths', [])
        original_reasoning = strengths[0] if strengths else "근거 없음"

        print(f"\n   선택된 역량: {competency_name}")
        print(f"   원본 점수: {original_score}점")
        print(f"   원본 강점: {original_reasoning[:100]}...")
        print()

        # ========================================
        # Step 2: HR이 점수를 수정하는 시나리오
        # ========================================
        print("[Step 2] HR 피드백 시뮬레이션 (점수 수정)")
        print("-" * 80)

        # 점수를 의도적으로 낮춤 (또는 높임)
        modified_score = max(0, original_score - 15)  # 15점 낮춤
        hr_reasoning = f"{competency_name} 역량을 재검토한 결과, 구체적 사례가 부족하여 {modified_score}점이 적절함. 추가 검증 필요."

        print(f"   HR 수정 점수: {modified_score}점 (원본 대비 {modified_score - original_score:+d}점)")
        print(f"   HR 수정 사유: {hr_reasoning}")

        feedback_manager = FeedbackManager(db=db, openai_client=openai_client)

        feedback = await feedback_manager.save_feedback(
            job_category="Sales",  # 102번의 직무
            competency_name=competency_name,
            ai_score=original_score,
            ai_reasoning=original_reasoning,
            human_score=modified_score,
            human_reasoning=hr_reasoning,
            evaluation_id=interview_id,
            use_llm_summary=False  # V2: 빠른 저장
        )

        print(f"\n✅ 피드백 저장 완료 (ID: {feedback.id})")
        print()

        # ========================================
        # Step 3: 유사 상황 검색
        # ========================================
        print("[Step 3] 유사 피드백 검색")
        print("-" * 80)

        # 트랜스크립트 요약 (실제 평가에서 사용된 컨텍스트)
        transcript_summary = original_reasoning[:300]

        relevant_feedbacks = await feedback_manager.get_relevant_feedback(
            job_category="Sales",
            competency_name=competency_name,
            current_context=transcript_summary,
            top_k=3,
            similarity_threshold=0.5,
            use_dynamic_threshold=True
        )

        print(f"🔍 검색 결과: {len(relevant_feedbacks)}개")
        for i, fb in enumerate(relevant_feedbacks, 1):
            print(f"  [{i}] 유사도: {fb['similarity']:.2%}")
            print(f"      점수 변화: {fb['ai_score']} → {fb['human_score']} ({fb['human_score'] - fb['ai_score']:+d}점)")
            print(f"      교정 가이드: {fb['correction_guideline'][:80]}...")
        print()

        # ========================================
        # Step 4: 재평가 시뮬레이션 (선택적)
        # ========================================
        print("[Step 4] 재평가 시뮬레이션")
        print("-" * 80)
        print("⚠️  실제 CompetencyAgent 재평가는 OpenAI API 비용이 발생합니다.")
        print("   계속하려면 아래 코드 주석을 해제하세요.\n")

        # ===== 재평가 코드 (주석 처리) =====
        # agent = CompetencyAgent(
        #     openai_client=openai_client,
        #     db_session=db,
        #     use_feedback=True,       # 피드백 활성화
        #     job_category="Sales"
        # )
        #
        # # 트랜스크립트 로드 (실제 면접 데이터 필요)
        # with open(f"test_data/transcript_{interview_id}.json", "r") as f:
        #     transcript = json.load(f)
        #
        # # 재평가 실행
        # new_result = await agent.evaluate(
        #     competency_name=competency_name,
        #     competency_display_name=competency_name,
        #     competency_category="common",
        #     prompt=original_reasoning,  # 동일한 프롬프트
        #     transcript=transcript
        # )
        #
        # print(f"✅ 재평가 완료")
        # print(f"   - 기존 점수: {original_score}점")
        # print(f"   - 새 점수: {new_result['score']}점")
        # print(f"   - 변화: {new_result['score'] - original_score:+d}점")

        print("💡 재평가를 건너뜁니다. 검색 결과만으로 Few-shot 효과 확인 가능.\n")

        # ========================================
        # Step 5: 누적 효과 확인
        # ========================================
        print("[Step 5] 피드백 누적 효과 분석")
        print("-" * 80)

        # 동일 역량에 대한 모든 피드백 조회
        from models.feedback_memory import FeedbackMemory

        all_feedbacks = db.query(FeedbackMemory).filter(
            FeedbackMemory.competency_name == competency_name
        ).all()

        print(f"  {competency_name} 역량 피드백 개수: {len(all_feedbacks)}개")

        if all_feedbacks:
            avg_score_change = sum(fb.human_score - fb.ai_score for fb in all_feedbacks) / len(all_feedbacks)
            print(f"  평균 점수 변화: {avg_score_change:+.1f}점")

            if avg_score_change > 0:
                print(f"  📈 AI가 일관되게 {competency_name}을(를) 낮게 평가하는 경향 발견!")
            elif avg_score_change < 0:
                print(f"  📉 AI가 일관되게 {competency_name}을(를) 높게 평가하는 경향 발견!")

        print()
        print("=" * 80)
        print("✅ 테스트 완료!")
        print("=" * 80)
        print("\n📊 요약:")
        print(f"   1. {interview_id}번 평가 데이터 로드 성공")
        print(f"   2. {competency_name} 역량 피드백 저장")
        print(f"   3. 유사 피드백 {len(relevant_feedbacks)}개 검색 완료")
        print(f"   4. 누적 피드백 {len(all_feedbacks)}개")
        print("\n💡 다음 단계:")
        print("   - 여러 지원자 데이터로 반복 테스트")
        print("   - 특정 역량에 5-10개 피드백 축적 후 재평가")
        print("   - Frontend에서 실제 HR 수정 워크플로우 테스트")

    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(test_feedback_with_real_data())
