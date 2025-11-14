"""
RAG 임베딩 효과 검증 스크립트

목적:
- 기존 평가 결과를 활용하여 segment layer에서 RAG 효과 검증
- Segment별 임베딩 생성
- 역량별 RAG 검색 테스트
- 관련성 점수 비교

데이터:
- transcript_jiwon_101.json (면접 대화)
- evaluation_result_101.json (기존 평가 결과)
"""
import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path, override=True)

from openai import AsyncOpenAI


# 역량별 검색 쿼리 정의
COMPETENCY_SEARCH_QUERIES = {
    # Common Competencies (5개)
    "achievement_motivation": "목표 설정, 자발적 시작, 내적 동기, 뿌듯한 경험, 성취 욕구, 도전 추구, 프로젝트 완수, 자기주도적",
    "growth_potential": "학습 경험, 실패로부터 배움, 새로운 기술 습득, 자기계발, 피드백 수용, 성장 마인드, 개선 노력",
    "interpersonal_skill": "팀워크, 협업 경험, 갈등 해결, 커뮤니케이션, 관계 형성, 공감 능력, 설득력, 리스닝",
    "organizational_fit": "조직 문화, 가치관, 업무 스타일, 팀 적응, 회사 선택 이유, 업무 환경, 조직 생활",
    "problem_solving": "문제 해결 사례, 논리적 사고, 창의적 접근, 분석 능력, 복잡한 상황 대처, 의사결정",

    # Job Competencies (5개)
    "customer_journey_marketing": "고객 여정, VMD, 마케팅 전략, 브랜드 경험, 고객 행동, 매장 운영, 시각적 머천다이징",
    "md_data_analysis": "데이터 분석, 트렌드 분석, 매출 분석, 상품 기획, 판매 데이터, 재고 분석, SKU 관리, 피벗 테이블",
    "seasonal_strategy_kpi": "시즌 전략, KPI 설정, 목표 달성, 전략 수립, 성과 지표, 비즈니스 계획, 실행력",
    "stakeholder_collaboration": "이해관계자 협업, 부서간 협력, 협상, 조율, 커뮤니케이션, 파트너십",
    "value_chain_optimization": "소싱, 생산, 유통, 공급망, 원가 절감, 효율화, 물류, 벤더 관리",
}

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


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """코사인 유사도 계산"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


async def generate_embeddings(
    client: AsyncOpenAI,
    texts: List[str]
) -> List[List[float]]:
    """텍스트 리스트를 임베딩으로 변환"""

    print(f"  임베딩 생성 중... ({len(texts)}개 텍스트)")

    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    embeddings = [item.embedding for item in response.data]
    print(f"  ✓ 임베딩 생성 완료 (차원: {len(embeddings[0])})")

    return embeddings


async def create_segment_embeddings(
    client: AsyncOpenAI,
    transcript: Dict
) -> Dict[int, Dict]:
    """
    Segment별 임베딩 생성

    Returns:
        {
            segment_id: {
                "segment_id": int,
                "question": str,
                "answer": str,
                "combined_text": str,
                "embedding": List[float],
                "metadata": {...}
            }
        }
    """

    print("\n" + "="*80)
    print("[Step 1] Segment 임베딩 생성")
    print("="*80)

    segments = transcript.get("segments", [])

    # 임베딩용 텍스트 준비 (질문 + 답변)
    segment_texts = []
    segment_metadata = []

    for seg in segments:
        segment_id = seg["segment_id"]
        question = seg.get("question_text", "")
        answer = seg.get("answer_text", "")

        # 질문과 답변을 결합 (더 나은 의미 표현)
        combined_text = f"질문: {question}\n답변: {answer}"

        segment_texts.append(combined_text)
        segment_metadata.append({
            "segment_id": segment_id,
            "segment_order": seg.get("segment_order"),
            "question": question,
            "answer": answer,
            "combined_text": combined_text,
            "turn_type": seg.get("turn_type"),
            "interviewer_name": seg.get("interviewer_name"),
        })

    # 배치 임베딩 생성
    embeddings = await generate_embeddings(client, segment_texts)

    # 결과 조합
    segment_embeddings = {}
    for metadata, embedding in zip(segment_metadata, embeddings):
        segment_id = metadata["segment_id"]
        segment_embeddings[segment_id] = {
            **metadata,
            "embedding": embedding
        }

    print(f"\n  총 {len(segment_embeddings)}개 Segment 임베딩 생성 완료")

    return segment_embeddings


async def test_rag_search(
    client: AsyncOpenAI,
    competency_name: str,
    segment_embeddings: Dict[int, Dict],
    top_k: int = 5
) -> List[Dict]:
    """
    특정 역량에 대한 RAG 검색 테스트

    Returns:
        [
            {
                "segment_id": int,
                "similarity": float,
                "question": str,
                "answer": str,
                "rank": int
            }
        ]
    """

    # 1. 역량별 검색 쿼리
    search_query = COMPETENCY_SEARCH_QUERIES.get(
        competency_name,
        f"{competency_name} 관련 행동 사례"
    )

    # 2. 쿼리 임베딩
    query_embedding = await generate_embeddings(client, [search_query])
    query_vector = query_embedding[0]

    # 3. 모든 segment와 유사도 계산
    similarities = []
    for segment_id, seg_data in segment_embeddings.items():
        seg_embedding = seg_data["embedding"]
        similarity = cosine_similarity(query_vector, seg_embedding)

        similarities.append({
            "segment_id": segment_id,
            "segment_order": seg_data["segment_order"],
            "similarity": similarity,
            "question": seg_data["question"],
            "answer": seg_data["answer"],
            "turn_type": seg_data["turn_type"],
            "interviewer_name": seg_data["interviewer_name"],
        })

    # 4. 유사도 내림차순 정렬
    similarities.sort(key=lambda x: x["similarity"], reverse=True)

    # 5. Top-K 반환
    top_results = []
    for rank, result in enumerate(similarities[:top_k], 1):
        top_results.append({
            **result,
            "rank": rank
        })

    return top_results


async def compare_with_agent_evaluation(
    rag_results: Dict[str, List[Dict]],
    evaluation_result: Dict
) -> Dict:
    """
    RAG 검색 결과와 Agent 평가 결과 비교

    Args:
        rag_results: {competency_name: [top_k_segments]}
        evaluation_result: 기존 평가 결과

    Returns:
        비교 분석 결과
    """

    print("\n" + "="*80)
    print("[Step 3] RAG vs Agent 평가 비교")
    print("="*80)

    comparison = {}

    competency_results = evaluation_result.get("competency_results", {})

    for comp_name, rag_top_k in rag_results.items():
        comp_result = competency_results.get(comp_name, {})

        # Agent가 사용한 evidence_details 추출
        perspectives = comp_result.get("perspectives", {})
        evidence_details = perspectives.get("evidence_details", [])

        agent_segment_ids = set()
        for ev in evidence_details:
            seg_id = ev.get("segment_id")
            if seg_id:
                agent_segment_ids.add(seg_id)

        # RAG Top-K segment IDs
        rag_segment_ids = set([r["segment_id"] for r in rag_top_k])

        # 교집합 (겹치는 segment)
        overlap = agent_segment_ids.intersection(rag_segment_ids)

        # 비교 결과
        comparison[comp_name] = {
            "agent_segments": sorted(list(agent_segment_ids)),
            "rag_top_k_segments": sorted(list(rag_segment_ids)),
            "overlap_segments": sorted(list(overlap)),
            "overlap_count": len(overlap),
            "overlap_rate": len(overlap) / len(agent_segment_ids) if agent_segment_ids else 0.0,
            "agent_score": comp_result.get("overall_score"),
            "rag_top_k": rag_top_k[:3]  # 상위 3개만 저장
        }

    return comparison


async def main():
    """메인 실행"""

    print("\n" + "="*80)
    print("  RAG 임베딩 효과 검증 테스트")
    print("="*80)

    # 데이터 로드
    data_dir = Path(__file__).parent.parent / "test_data"
    transcript_path = data_dir / "transcript_jiwon_101.json"
    evaluation_result_path = data_dir / "evaluation_result_101.json"

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    with open(evaluation_result_path, "r", encoding="utf-8") as f:
        evaluation_result = json.load(f)

    print(f"\n  데이터 로드 완료:")
    print(f"  - Transcript: {len(transcript['segments'])}개 segment")
    print(f"  - 평가 결과: {len(evaluation_result.get('competency_results', {}))}개 역량")

    # OpenAI 클라이언트
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


    # Step 1: Segment 임베딩 생성
    segment_embeddings = await create_segment_embeddings(client, transcript)


    # Step 2: 역량별 RAG 검색 테스트
    print("\n" + "="*80)
    print("[Step 2] 역량별 RAG 검색 테스트")
    print("="*80)

    rag_results = {}

    # 10개 역량 테스트
    competencies_to_test = list(COMPETENCY_SEARCH_QUERIES.keys())

    for comp_name in competencies_to_test:
        comp_display = COMPETENCY_DISPLAY_NAMES.get(comp_name, comp_name)
        print(f"\n[{comp_display}]")
        print(f"  검색 쿼리: {COMPETENCY_SEARCH_QUERIES[comp_name][:60]}...")

        top_results = await test_rag_search(
            client,
            comp_name,
            segment_embeddings,
            top_k=8
        )

        rag_results[comp_name] = top_results

        # 상위 3개 출력
        print(f"  Top 3 segments:")
        for result in top_results[:3]:
            print(f"    {result['rank']}. Segment {result['segment_id']} (유사도: {result['similarity']:.3f})")
            print(f"       Q: {result['question'][:60]}...")
            print(f"       A: {result['answer'][:80]}...")


    # Step 3: Agent 평가와 비교
    comparison = await compare_with_agent_evaluation(rag_results, evaluation_result)


    # Step 4: 결과 분석 및 출력
    print("\n" + "="*80)
    print("[Step 4] RAG 효과 분석")
    print("="*80)

    total_overlap = 0
    total_agent_segments = 0

    print("\n역량별 RAG vs Agent 비교:")
    print("-"*80)

    for comp_name, comp_comparison in comparison.items():
        comp_display = COMPETENCY_DISPLAY_NAMES.get(comp_name, comp_name)

        agent_segs = comp_comparison["agent_segments"]
        rag_segs = comp_comparison["rag_top_k_segments"]
        overlap = comp_comparison["overlap_segments"]
        overlap_rate = comp_comparison["overlap_rate"]

        total_overlap += len(overlap)
        total_agent_segments += len(agent_segs)

        print(f"\n[{comp_display}]")
        print(f"  Agent 사용 segments: {agent_segs} ({len(agent_segs)}개)")
        print(f"  RAG Top-8 segments: {rag_segs[:8]} ({len(rag_segs)}개)")
        print(f"  겹치는 segments: {overlap} ({len(overlap)}개)")
        print(f"  ✓ 일치율: {overlap_rate*100:.1f}%")

        if overlap_rate >= 0.7:
            print(f"  → 평가: 🟢 RAG가 Agent 판단과 70% 이상 일치")
        elif overlap_rate >= 0.5:
            print(f"  → 평가: 🟡 RAG가 Agent 판단과 50% 이상 일치")
        else:
            print(f"  → 평가: 🔴 RAG가 Agent 판단과 50% 미만 일치 (개선 필요)")

    # 전체 평균
    avg_overlap_rate = total_overlap / total_agent_segments if total_agent_segments > 0 else 0.0

    print("\n" + "-"*80)
    print(f"[전체 평균]")
    print(f"  전체 일치율: {avg_overlap_rate*100:.1f}%")
    print(f"  총 Agent segments: {total_agent_segments}개")
    print(f"  총 겹치는 segments: {total_overlap}개")

    if avg_overlap_rate >= 0.6:
        print(f"\n  ✅ 결론: RAG가 Agent 판단을 {avg_overlap_rate*100:.1f}% 재현 가능!")
        print(f"         → RAG 도입 시 평가 정확도 유지 가능")
    else:
        print(f"\n  ⚠️  결론: RAG 일치율 {avg_overlap_rate*100:.1f}% (개선 필요)")
        print(f"         → 검색 쿼리 또는 Top-K 조정 필요")


    # 결과 저장
    output_path = data_dir / "rag_test_result.json"

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "transcript_file": "transcript_jiwon_101.json",
        "evaluation_file": "evaluation_result_101.json",
        "segment_count": len(segment_embeddings),
        "competencies_tested": len(competencies_to_test),
        "avg_overlap_rate": avg_overlap_rate,
        "comparison": comparison,
        "rag_results": {
            comp_name: [
                {
                    "segment_id": r["segment_id"],
                    "similarity": r["similarity"],
                    "rank": r["rank"],
                    "question": r["question"][:100],
                    "answer": r["answer"][:150]
                }
                for r in results
            ]
            for comp_name, results in rag_results.items()
        }
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n  결과 저장: {output_path}")

    print("\n" + "="*80)
    print("  테스트 완료!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
