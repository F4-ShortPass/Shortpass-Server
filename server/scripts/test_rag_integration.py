"""
RAG 통합 테스트 스크립트

목적:
- CompetencyAgent RAG 통합 검증
- 임베딩 생성 및 RAG 검색 테스트
- 토큰 절감 효과 측정

실행 방법:
    python scripts/test_rag_integration.py
"""
import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path, override=True)

from openai import AsyncOpenAI
from sqlalchemy.orm import Session

from db.database import get_db
from models.interview import SessionTranscript, InterviewSession, InterviewStatus
from services.rag_embedding_service import generate_and_save_transcript_embeddings
from ai.agents.competency_agent import CompetencyAgent


async def create_test_transcripts(db: Session, transcript: dict) -> int:
    """
    테스트용 면접 세션 및 transcript 생성

    Returns:
        session_id: 생성된 세션 ID
    """
    print("\n" + "="*80)
    print("[Step 1] 테스트 데이터 생성")
    print("="*80)

    # 1. 테스트 세션 생성
    session = InterviewSession(
        applicant_id=101,
        company_id=1,
        status=InterviewStatus.COMPLETED
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    print(f"\n  ✓ 테스트 세션 생성: session_id={session.id}")

    # 2. Transcript segments를 SessionTranscript로 변환
    segments = transcript.get("segments", [])

    for seg in segments:
        # 질문과 답변을 결합
        combined_text = f"질문: {seg.get('question_text', '')}\\n답변: {seg.get('answer_text', '')}"

        st = SessionTranscript(
            session_id=session.id,
            persona_instance_id=None,
            turn=seg.get("segment_order"),
            text=combined_text,
            meta_json={
                "segment_id": seg.get("segment_id"),
                "interviewer_name": seg.get("interviewer_name"),
                "turn_type": seg.get("turn_type")
            }
        )
        db.add(st)

    db.commit()

    print(f"  ✓ {len(segments)}개 transcript 생성")

    return session.id


async def test_rag_evaluation(
    db: Session,
    session_id: int,
    transcript: dict
):
    """
    RAG 기반 평가 테스트
    """
    print("\n" + "="*80)
    print("[Step 3] RAG 평가 테스트")
    print("="*80)

    # OpenAI 클라이언트
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 1. RAG 없이 평가
    print("\n[1] RAG 비활성화 평가:")
    agent_no_rag = CompetencyAgent(
        openai_client=client,
        use_rag=False
    )

    # 간단한 프롬프트 (테스트용)
    test_prompt = f"""
다음 면접 대화를 분석하여 '문제 해결 능력'을 평가하세요.

면접 대화:
{json.dumps(transcript.get('segments', [])[:3], ensure_ascii=False, indent=2)}

다음 JSON 형식으로 응답하세요:
{{
    "overall_score": 75,
    "strengths": ["구체적 분석 프레임워크 사용"],
    "weaknesses": ["리스크 관리 미흡"],
    "key_observations": ["데이터 기반 접근"],
    "perspectives": {{
        "evidence_reasoning": "구조적 사고 능력 우수",
        "evidence_details": []
    }},
    "confidence": {{
        "overall_confidence": 0.8
    }}
}}
"""

    try:
        result_no_rag = await agent_no_rag.evaluate(
            competency_name="problem_solving",
            competency_display_name="문제해결력",
            competency_category="common",
            prompt=test_prompt,
            transcript=transcript
        )
        print(f"  ✓ 평가 완료: {result_no_rag.get('overall_score')}점")
    except Exception as e:
        print(f"  ✗ 평가 실패: {e}")
        result_no_rag = None


    # 2. RAG 활성화 평가
    print("\n[2] RAG 활성화 평가:")
    agent_with_rag = CompetencyAgent(
        openai_client=client,
        use_rag=True,
        rag_top_k=5,
        db_session=db
    )

    try:
        result_with_rag = await agent_with_rag.evaluate(
            competency_name="problem_solving",
            competency_display_name="문제해결력",
            competency_category="common",
            prompt=test_prompt,
            transcript=transcript,
            session_id=session_id
        )
        print(f"  ✓ 평가 완료: {result_with_rag.get('overall_score')}점")

        # RAG 메타데이터 확인
        rag_metadata = transcript.get("rag_metadata", {})
        if rag_metadata:
            print(f"  📊 RAG 통계:")
            print(f"    - 원본 segments: {rag_metadata.get('original_segment_count')}")
            print(f"    - 필터 segments: {rag_metadata.get('filtered_segment_count')}")
            print(f"    - 토큰 절감률: {rag_metadata.get('token_reduction_rate', 0)*100:.1f}%")

    except Exception as e:
        print(f"  ✗ 평가 실패: {e}")
        result_with_rag = None


    # 3. 결과 비교
    print("\n[3] 결과 비교:")
    if result_no_rag and result_with_rag:
        print(f"  RAG 없이: {result_no_rag.get('overall_score')}점")
        print(f"  RAG 사용: {result_with_rag.get('overall_score')}점")
        print(f"  점수 차이: {abs(result_no_rag.get('overall_score', 0) - result_with_rag.get('overall_score', 0))}점")


async def main():
    """메인 실행"""

    print("\n" + "="*80)
    print("  RAG 통합 테스트")
    print("="*80)

    # 1. 테스트 데이터 로드
    transcript_path = Path(__file__).parent.parent / "test_data" / "transcript_jiwon_101.json"

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    print(f"\n  테스트 데이터: {transcript_path}")
    print(f"  총 segments: {len(transcript.get('segments', []))}개")


    # 2. DB 세션
    db_generator = get_db()
    db = next(db_generator)

    try:
        # 3. 테스트 데이터 생성
        session_id = await create_test_transcripts(db, transcript)


        # 4. 임베딩 생성
        print("\n" + "="*80)
        print("[Step 2] 임베딩 생성")
        print("="*80)

        embedding_result = await generate_and_save_transcript_embeddings(
            db=db,
            session_id=session_id,
            force_regenerate=True
        )

        print(f"\n  ✓ 임베딩 생성 완료:")
        print(f"    - 총 transcript: {embedding_result['total_transcripts']}개")
        print(f"    - 생성된 embedding: {embedding_result['embeddings_generated']}개")


        # 5. RAG 평가 테스트
        await test_rag_evaluation(db, session_id, transcript)


        print("\n" + "="*80)
        print("  ✅ 테스트 완료!")
        print("="*80)

    finally:
        # Cleanup
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
