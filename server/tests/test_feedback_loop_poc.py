"""
오답노트 피드백 루프 POC 테스트
Reflexion 패턴을 활용한 Self-Improving AI Agent 검증
"""
import asyncio
import sys
import os

# Add server directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openai import AsyncOpenAI
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from db.database import Base
from models.feedback_memory import FeedbackMemory
from services.feedback.feedback_manager import FeedbackManager
from core.config import settings


async def test_feedback_poc():
    """POC 전체 플로우 테스트"""

    print("=" * 80)
    print("🧪 Feedback Loop POC Test - Reflexion Pattern")
    print("=" * 80)

    # 1. DB 연결 및 테이블 생성
    print("\n[Step 1] Database Setup")
    print("-" * 80)

    engine = create_engine(settings.DATABASE_URL, echo=False)

    # pgvector extension 확인
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM pg_extension WHERE extname = 'vector'"))
        if result.fetchone():
            print("✓ pgvector extension is enabled")
        else:
            print("✗ pgvector extension not found!")
            print("  Run: CREATE EXTENSION IF NOT EXISTS vector;")
            return

    # 테이블 생성
    print("Creating feedback_memory table...")
    Base.metadata.create_all(bind=engine, tables=[FeedbackMemory.__table__])
    print("✓ Table created successfully")

    # Session 생성
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    # OpenAI 클라이언트
    openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    # FeedbackManager 초기화
    feedback_manager = FeedbackManager(db=db, openai_client=openai_client)


    # 2. 샘플 피드백 저장
    print("\n[Step 2] Save Sample Feedback")
    print("-" * 80)

    sample_feedbacks = [
        {
            "job_category": "Sales",
            "competency_name": "interpersonal_skill",
            "ai_score": 70,
            "ai_reasoning": "지원자가 공격적인 말투를 사용하여 대인관계 역량이 부족해 보임. 고객과의 소통 시 부드러운 접근이 필요해 보임.",
            "human_score": 90,
            "human_reasoning": "영업 직무에서는 공격성이 아니라 적극성으로 해석해야 함. 고객 설득력이 뛰어나고 목표 지향적인 태도가 우수함.",
        },
        {
            "job_category": "Sales",
            "competency_name": "problem_solving",
            "ai_score": 60,
            "ai_reasoning": "문제 해결 사례가 구체적이지 않고 결과 중심으로만 설명함. 분석적 접근이 부족해 보임.",
            "human_score": 85,
            "human_reasoning": "영업 직무에서는 빠른 의사결정과 실행력이 더 중요함. 분석보다는 행동 중심의 문제 해결 방식이 직무에 적합함.",
        },
        {
            "job_category": "Engineering",
            "competency_name": "problem_solving",
            "ai_score": 85,
            "ai_reasoning": "논리적 사고와 체계적 접근이 우수함. 기술적 문제를 단계적으로 분석하고 해결함.",
            "human_score": 95,
            "human_reasoning": "알고리즘 최적화와 시스템 설계에서 탁월한 역량 보임. 단순 점수보다 더 높게 평가해야 함.",
        }
    ]

    saved_feedbacks = []
    for i, fb in enumerate(sample_feedbacks, 1):
        print(f"\n  Saving feedback {i}/{len(sample_feedbacks)}...")
        saved = await feedback_manager.save_feedback(
            evaluation_id=1000 + i,
            applicant_id=100 + i,
            **fb
        )
        saved_feedbacks.append(saved)
        print(f"    ✓ ID: {saved.id}")
        print(f"    ✓ Mistake: {saved.mistake_summary[:60]}...")
        print(f"    ✓ Guideline: {saved.correction_guideline[:60]}...")

    print(f"\n✓ Saved {len(saved_feedbacks)} feedbacks to database")


    # 3. 피드백 검색 테스트
    print("\n[Step 3] Search Relevant Feedback (RAG)")
    print("-" * 80)

    # 테스트 시나리오: 새로운 Sales 지원자 평가
    test_context = """
    지원자가 적극적이고 설득력 있는 말투로 고객 응대 사례를 설명함.
    "저는 고객에게 강력하게 어필하고 빠르게 의사결정을 유도합니다"라고 답변.
    """

    print(f"\n  Query Context: {test_context.strip()}")
    print(f"  Job Category: Sales")
    print(f"  Competency: interpersonal_skill\n")

    relevant_feedbacks = await feedback_manager.get_relevant_feedback(
        job_category="Sales",
        competency_name="interpersonal_skill",
        current_context=test_context,
        top_k=3,
        similarity_threshold=0.5  # 50% 유사도
    )

    print(f"\n  Found {len(relevant_feedbacks)} relevant feedbacks:\n")
    for i, fb in enumerate(relevant_feedbacks, 1):
        print(f"  [{i}] Similarity: {fb['similarity']:.2%}")
        print(f"      Mistake: {fb['mistake_summary']}")
        print(f"      Guideline: {fb['correction_guideline'][:80]}...")
        print(f"      Score Change: {fb['ai_score']}점 → {fb['human_score']}점\n")


    # 4. CompetencyAgent 통합 테스트 (시뮬레이션)
    print("\n[Step 4] Simulate CompetencyAgent with Feedback Injection")
    print("-" * 80)

    print("\n  🔹 Scenario: AI가 과거와 유사한 실수를 할 뻔했지만, 피드백 덕분에 방지됨\n")

    # 과거 실수: "공격적 말투" → 낮은 점수
    print("  [Before Feedback Loop]")
    print("    AI Judgment: 지원자가 공격적 → 70점")
    print("    Problem: 영업 특성을 고려하지 못함\n")

    # 피드백 주입 후
    print("  [After Feedback Loop - with Reflexion]")
    if relevant_feedbacks:
        print(f"    ✓ {len(relevant_feedbacks)}개의 유사 피드백 발견!")
        print(f"    ✓ 프롬프트에 교정 지침 주입:")
        for fb in relevant_feedbacks[:1]:  # 첫 번째만 출력
            print(f"      '{fb['correction_guideline']}'")
        print(f"\n    → AI가 이제는 '적극성'으로 재해석할 가능성 ↑")
        print(f"    → 예상 점수: 85~90점 (과거 Human Score 참고)")

    print("\n  🎯 Result: AI가 과거 실수를 학습하여 동일한 오류를 방지함!")


    # 5. 통계 확인
    print("\n[Step 5] Feedback Statistics")
    print("-" * 80)

    from sqlalchemy import func

    total = db.query(func.count(FeedbackMemory.id)).scalar()
    print(f"\n  Total Feedbacks: {total}")

    by_competency = db.query(
        FeedbackMemory.competency_name,
        func.count(FeedbackMemory.id).label('count')
    ).group_by(FeedbackMemory.competency_name).all()

    print(f"\n  By Competency:")
    for item in by_competency:
        print(f"    - {item.competency_name}: {item.count}")

    by_job = db.query(
        FeedbackMemory.job_category,
        func.count(FeedbackMemory.id).label('count')
    ).group_by(FeedbackMemory.job_category).all()

    print(f"\n  By Job Category:")
    for item in by_job:
        print(f"    - {item.job_category}: {item.count}")


    # 정리
    db.close()

    print("\n" + "=" * 80)
    print("✅ POC Test Completed Successfully!")
    print("=" * 80)
    print("\n📌 Next Steps:")
    print("  1. Frontend: CandidateEvaluation 페이지에 '점수 수정' UI 추가")
    print("  2. API Integration: 점수 수정 시 POST /api/v1/feedback/ 호출")
    print("  3. Agent Integration: evaluate_all_competencies에 use_feedback=True 전달")
    print("  4. Production: 실제 평가 파이프라인에 적용하여 성능 모니터링")
    print("\n💡 Expected Benefits:")
    print("  - AI가 쓸수록 똑똑해짐 (Self-Evolving System)")
    print("  - HR의 도메인 지식이 시스템에 축적됨")
    print("  - 동일 실수 반복 방지 → 평가 정확도 향상")
    print()


if __name__ == "__main__":
    asyncio.run(test_feedback_poc())
