"""
RAG 임베딩 생성 서비스 (OpenAI text-embedding-3-small)

목적:
- SessionTranscript의 텍스트를 임베딩 벡터로 변환
- OpenAI text-embedding-3-small 사용 (1536차원, $0.02/1M tokens)
- 배치 처리로 효율적 임베딩 생성
- 역량 평가를 위한 RAG 검색 지원
"""
import os
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
from openai import AsyncOpenAI
import numpy as np

from models.interview import SessionTranscript


# OpenAI 클라이언트 초기화
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 임베딩 모델 설정
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536


async def generate_embeddings(
    texts: List[str],
    batch_size: int = 100
) -> List[List[float]]:
    """
    텍스트 리스트를 임베딩으로 변환

    Args:
        texts: 임베딩할 텍스트 리스트
        batch_size: 배치 크기 (OpenAI API 제한: 2048개)

    Returns:
        임베딩 벡터 리스트 (각 벡터는 1536차원)
    """
    if not texts:
        return []

    all_embeddings = []

    # 배치 처리
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        response = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch
        )

        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


async def generate_and_save_transcript_embeddings(
    db: Session,
    session_id: int,
    force_regenerate: bool = False
) -> Dict:
    """
    특정 세션의 모든 transcript에 대해 임베딩 생성 및 저장

    Args:
        db: SQLAlchemy 세션
        session_id: 인터뷰 세션 ID
        force_regenerate: True면 기존 임베딩 무시하고 재생성

    Returns:
        {
            "session_id": int,
            "total_transcripts": int,
            "embeddings_generated": int,
            "embeddings_skipped": int
        }
    """
    # 1. 해당 세션의 transcripts 조회
    query = db.query(SessionTranscript).filter(
        SessionTranscript.session_id == session_id
    )

    if not force_regenerate:
        # 임베딩이 없는 것만 조회
        query = query.filter(SessionTranscript.embedding.is_(None))

    transcripts = query.order_by(SessionTranscript.turn).all()

    if not transcripts:
        return {
            "session_id": session_id,
            "total_transcripts": 0,
            "embeddings_generated": 0,
            "embeddings_skipped": 0
        }

    # 2. 텍스트 추출
    texts = [t.text for t in transcripts]
    transcript_ids = [t.id for t in transcripts]

    # 3. 임베딩 생성
    embeddings = await generate_embeddings(texts)

    # 4. DB 업데이트 (배치)
    for transcript_id, embedding in zip(transcript_ids, embeddings):
        # pgvector는 Python 리스트를 자동으로 vector 타입으로 변환
        db.execute(
            text("""
                UPDATE session_transcripts
                SET embedding = :embedding
                WHERE id = :transcript_id
            """),
            {
                "embedding": embedding,
                "transcript_id": transcript_id
            }
        )

    db.commit()

    # 5. 통계 반환
    total_count = db.query(SessionTranscript).filter(
        SessionTranscript.session_id == session_id
    ).count()

    return {
        "session_id": session_id,
        "total_transcripts": total_count,
        "embeddings_generated": len(embeddings),
        "embeddings_skipped": total_count - len(embeddings)
    }


async def search_relevant_transcripts(
    db: Session,
    session_id: int,
    query_text: str,
    top_k: int = 5,
    similarity_threshold: float = 0.0
) -> List[Dict]:
    """
    특정 세션에서 쿼리와 유사한 transcript 검색 (RAG)

    Args:
        db: SQLAlchemy 세션
        session_id: 인터뷰 세션 ID
        query_text: 검색 쿼리 (예: "문제 해결 능력, 논리적 사고")
        top_k: 반환할 상위 결과 개수
        similarity_threshold: 최소 유사도 (0~1, cosine similarity)

    Returns:
        [
            {
                "transcript_id": int,
                "turn": int,
                "text": str,
                "similarity": float,
                "meta_json": dict
            },
            ...
        ]
    """
    # 1. 쿼리 임베딩 생성
    query_embeddings = await generate_embeddings([query_text])
    query_vector = query_embeddings[0]

    # 2. 벡터 유사도 검색 (cosine similarity)
    # pgvector의 <=> 연산자는 cosine distance (1 - cosine_similarity)
    # 따라서 similarity = 1 - distance
    result = db.execute(
        text("""
            SELECT
                id,
                turn,
                text,
                meta_json,
                1 - (embedding <=> :query_vector) AS similarity
            FROM session_transcripts
            WHERE session_id = :session_id
              AND embedding IS NOT NULL
              AND 1 - (embedding <=> :query_vector) >= :threshold
            ORDER BY embedding <=> :query_vector
            LIMIT :top_k
        """),
        {
            "query_vector": query_vector,
            "session_id": session_id,
            "threshold": similarity_threshold,
            "top_k": top_k
        }
    )

    rows = result.fetchall()

    # 3. 결과 포맷팅
    results = []
    for row in rows:
        results.append({
            "transcript_id": row[0],
            "turn": row[1],
            "text": row[2],
            "meta_json": row[3],
            "similarity": float(row[4])
        })

    return results


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    코사인 유사도 계산 (로컬 계산용)

    Args:
        vec1: 벡터 1
        vec2: 벡터 2

    Returns:
        유사도 (0~1)
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
