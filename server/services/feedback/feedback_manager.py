# server/services/feedback/feedback_manager.py
"""
Feedback Manager - 오답노트 피드백 루프 핵심 서비스
HR의 평가 수정을 저장하고, 다음 평가 시 유사한 실수를 방지하기 위한 RAG 검색
"""
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
from openai import AsyncOpenAI
from models.feedback_memory import FeedbackMemory


class FeedbackManager:
    """피드백 관리 서비스 (RAG 기반 오답노트)"""

    def __init__(self, db: Session, openai_client: AsyncOpenAI):
        self.db = db
        self.openai_client = openai_client

    async def save_feedback(
        self,
        job_category: str,
        competency_name: str,
        ai_score: int,
        ai_reasoning: str,
        human_score: int,
        human_reasoning: str,
        evaluation_id: Optional[int] = None,
        applicant_id: Optional[int] = None,
        use_llm_summary: bool = False  # 🆕 V2: 기본적으로 LLM 사용 안함
    ) -> FeedbackMemory:
        """
        HR의 평가 수정을 오답노트로 저장 (V2: Simple First)

        Args:
            job_category: 직무 카테고리 (예: 'Sales', 'Engineering')
            competency_name: 역량 이름 (예: 'problem_solving')
            ai_score: AI가 매긴 점수
            ai_reasoning: AI의 판단 근거
            human_score: HR이 수정한 점수
            human_reasoning: HR의 수정 사유
            evaluation_id: 원본 평가 ID
            applicant_id: 지원자 ID
            use_llm_summary: LLM으로 요약 생성 (기본: False, 빠른 저장)

        Returns:
            저장된 FeedbackMemory 객체
        """
        # 🆕 V2: LLM 요약을 선택적으로만 사용
        if use_llm_summary:
            # 기존 방식 (LLM 2회 호출)
            mistake_summary = await self._summarize_mistake(
                ai_reasoning=ai_reasoning,
                human_reasoning=human_reasoning,
                ai_score=ai_score,
                human_score=human_score
            )

            correction_guideline = await self._generate_correction_guideline(
                competency_name=competency_name,
                mistake_summary=mistake_summary,
                human_reasoning=human_reasoning
            )
        else:
            # 🆕 Simple First: 템플릿 사용 (LLM 호출 0회)
            score_diff = human_score - ai_score
            mistake_summary = f"AI 점수 {ai_score}점을 {human_score}점으로 조정 (차이: {score_diff:+d}점)"

            # HR의 수정 사유를 그대로 사용
            correction_guideline = human_reasoning

        # 임베딩 생성 (HR 수정 사유로 생성 - 더 정확)
        embedding_vector = await self._get_embedding(human_reasoning)

        # DB 저장
        feedback = FeedbackMemory(
            job_category=job_category,
            competency_name=competency_name,
            mistake_summary=mistake_summary,
            ai_score=ai_score,
            ai_reasoning=ai_reasoning,
            human_score=human_score,
            correction_guideline=correction_guideline,
            embedding=embedding_vector,
            evaluation_id=evaluation_id,
            applicant_id=applicant_id
        )

        self.db.add(feedback)
        self.db.commit()
        self.db.refresh(feedback)

        print(f"✅ Feedback saved: {competency_name} ({job_category}) - {mistake_summary[:50]}...")

        return feedback

    async def get_relevant_feedback(
        self,
        job_category: str,
        competency_name: str,
        current_context: str,
        top_k: int = 3,
        similarity_threshold: float = 0.5,  # 🆕 V2: 기본값 낮춤
        use_dynamic_threshold: bool = True  # 🆕 V2: 동적 threshold
    ) -> List[Dict]:
        """
        현재 평가 상황과 유사한 과거 피드백 검색 (V2: Dynamic Threshold)

        Args:
            job_category: 직무 카테고리
            competency_name: 역량 이름
            current_context: 현재 평가 컨텍스트 (트랜스크립트 요약 등)
            top_k: 반환할 최대 개수
            similarity_threshold: 최소 유사도 (0~1)
            use_dynamic_threshold: 동적으로 threshold 조정 (기본: True)

        Returns:
            관련 피드백 목록 (유사도 높은 순)
        """
        # 1. 현재 컨텍스트 임베딩
        query_vector = await self._get_embedding(current_context)

        # 🆕 V2: Dynamic Threshold - 결과가 없으면 점진적으로 낮춤
        if use_dynamic_threshold:
            return await self._search_with_dynamic_threshold(
                query_vector=query_vector,
                job_category=job_category,
                competency_name=competency_name,
                top_k=top_k,
                initial_threshold=similarity_threshold
            )
        else:
            # 기존 방식 (고정 threshold)
            return await self._search_with_threshold(
                query_vector=query_vector,
                job_category=job_category,
                competency_name=competency_name,
                top_k=top_k,
                threshold=similarity_threshold
            )

    async def _search_with_dynamic_threshold(
        self,
        query_vector: List[float],
        job_category: str,
        competency_name: str,
        top_k: int,
        initial_threshold: float
    ) -> List[Dict]:
        """
        🆕 V2: 동적 threshold로 검색 (결과 없으면 점진적으로 완화)
        """
        thresholds = [initial_threshold, 0.5, 0.3, 0.1, 0.0]

        for threshold in thresholds:
            results = await self._search_with_threshold(
                query_vector=query_vector,
                job_category=job_category,
                competency_name=competency_name,
                top_k=top_k,
                threshold=threshold
            )

            if results:
                if threshold < initial_threshold:
                    print(f"  ⚙️  Threshold relaxed to {threshold} (found {len(results)} results)")
                return results

        # 최후의 수단: 무조건 1개 반환
        print(f"  ⚠️  No results found, returning top 1 without threshold")
        return await self._search_with_threshold(
            query_vector=query_vector,
            job_category=job_category,
            competency_name=competency_name,
            top_k=1,
            threshold=0.0
        )

    async def _search_with_threshold(
        self,
        query_vector: List[float],
        job_category: str,
        competency_name: str,
        top_k: int,
        threshold: float
    ) -> List[Dict]:
        """
        특정 threshold로 검색
        """
        # pgvector 코사인 유사도 검색
        sql = text("""
            SELECT
                id,
                job_category,
                competency_name,
                mistake_summary,
                correction_guideline,
                ai_score,
                human_score,
                (1 - (embedding <=> CAST(:query_vector AS vector))) as similarity
            FROM feedback_memory
            WHERE job_category = :job_category
              AND competency_name = :competency_name
              AND (1 - (embedding <=> CAST(:query_vector AS vector))) >= :threshold
            ORDER BY embedding <=> CAST(:query_vector AS vector)
            LIMIT :top_k
        """)

        # query_vector를 문자열로 변환 (pgvector 형식: '[0.1, 0.2, ...]')
        query_vector_str = "[" + ",".join(map(str, query_vector)) + "]"

        result = self.db.execute(
            sql,
            {
                "query_vector": query_vector_str,
                "job_category": job_category,
                "competency_name": competency_name,
                "threshold": threshold,  # 🆕 수정: 파라미터 이름 수정
                "top_k": top_k
            }
        )

        # 결과 변환
        feedbacks = []
        for row in result:
            feedbacks.append({
                "id": row.id,
                "job_category": row.job_category,
                "competency_name": row.competency_name,
                "mistake_summary": row.mistake_summary,
                "correction_guideline": row.correction_guideline,
                "ai_score": row.ai_score,
                "human_score": row.human_score,
                "similarity": float(row.similarity)
            })

        return feedbacks

    async def _summarize_mistake(
        self,
        ai_reasoning: str,
        human_reasoning: str,
        ai_score: int,
        human_score: int
    ) -> str:
        """AI의 실수를 한 문장으로 요약"""
        prompt = f"""
AI와 HR의 평가를 비교하여, AI가 어떤 맥락을 놓쳐서 실수했는지 **한 문장**으로 요약하세요.

AI 판단 (점수: {ai_score}점):
{ai_reasoning}

HR 수정 (점수: {human_score}점):
{human_reasoning}

요약 (한 문장, 30자 이내):
"""

        response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini",  # 빠르고 저렴한 모델
            messages=[
                {"role": "system", "content": "당신은 평가 차이를 분석하는 전문가입니다. 간결하게 요약하세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=100
        )

        summary = response.choices[0].message.content.strip()
        return summary

    async def _generate_correction_guideline(
        self,
        competency_name: str,
        mistake_summary: str,
        human_reasoning: str
    ) -> str:
        """교정 가이드라인 생성"""
        prompt = f"""
역량: {competency_name}
실수 요약: {mistake_summary}
HR의 올바른 판단: {human_reasoning}

위 정보를 바탕으로, 다음 평가 시 AI가 참고할 수 있는 **교정 가이드라인**을 작성하세요.
형식: "~할 때는 ~로 해석해야 합니다" (1-2문장)
"""

        response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 평가 가이드라인을 작성하는 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=200
        )

        guideline = response.choices[0].message.content.strip()
        return guideline

    async def _get_embedding(self, text: str) -> List[float]:
        """텍스트를 OpenAI Embedding으로 변환"""
        response = await self.openai_client.embeddings.create(
            model="text-embedding-3-small",  # 1536 차원
            input=text
        )

        return response.data[0].embedding
