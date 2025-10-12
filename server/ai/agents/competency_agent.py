"""
Competency Agent (RAG 통합 버전)
10개 역량 병렬 평가 + RAG 지원

수정 내용:
    1. LLM 응답 후 필수 필드 검증
    2. key_observations 누락 시 자동 생성
    3. RAG 기반 관련 segment 선별 (85% 토큰 절감)
    4. Pydantic Structured Output 도입
"""

import asyncio
from asyncio import Semaphore
import json
import hashlib
import os
from typing import Dict, Optional, List, Any
from datetime import datetime
from openai import AsyncOpenAI, RateLimitError, APIStatusError
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field


# ==================== Pydantic Models ====================

class ConfidenceDetail(BaseModel):
    """신뢰도 상세 정보"""
    level: str = Field(description="신뢰도 수준: high/medium/low")
    score: float = Field(ge=0, le=1, description="신뢰도 점수 (0~1)")
    reasoning: str = Field(description="신뢰도 판단 근거")


class SegmentEvidence(BaseModel):
    """세그먼트별 증거 상세"""
    segment_id: int = Field(description="세그먼트 ID")
    text: str = Field(description="인용문 텍스트")
    relevance_note: str = Field(description="관련성 설명")
    quality_score: float = Field(ge=0, le=1, description="증거 품질 점수")
    char_index: Optional[int] = Field(description="문자 시작 위치", default=None)
    char_length: Optional[int] = Field(description="문자 길이", default=None)


class PerspectivesDetail(BaseModel):
    """다각도 평가 정보"""
    evidence_reasoning: str = Field(description="근거 기반 평가 논리")
    behavioral_indicators: List[str] = Field(description="관찰된 행동 지표", default_factory=list)
    development_suggestions: List[str] = Field(description="개발 제안사항", default_factory=list)
    evidence_details: List[SegmentEvidence] = Field(
        description="세그먼트별 증거 상세 목록 (면접 transcript의 segment_id, 인용문, 관련성, 품질 점수)",
        default_factory=list
    )


class SegmentReference(BaseModel):
    """세그먼트 참조 정보"""
    segment_id: int = Field(description="세그먼트 ID")
    relevance_score: float = Field(ge=0, le=1, description="관련성 점수")
    key_quote: str = Field(description="핵심 인용문")


class CompetencyEvaluationResult(BaseModel):
    """역량 평가 결과 (Structured Output)"""
    competency_name: str = Field(description="역량 이름 (영문)")
    overall_score: int = Field(ge=0, le=100, description="종합 점수 (0-100)")
    strengths: List[str] = Field(description="강점 목록 (2-5개)", min_length=1)
    weaknesses: List[str] = Field(description="약점 목록 (1-3개)", min_length=1)
    key_observations: List[str] = Field(description="핵심 관찰사항 (3-5개)", min_length=3)
    perspectives: PerspectivesDetail = Field(description="다각도 평가")
    confidence: ConfidenceDetail = Field(description="평가 신뢰도")
    segment_references: List[SegmentReference] = Field(
        description="참조한 세그먼트 정보",
        default_factory=list
    )


class CompetencyAgent:
    """역량 평가 Agent (RAG 통합)"""

    # 필수 필드 정의
    REQUIRED_FIELDS = {
        "competency_name": str,
        "overall_score": int,
        "strengths": list,
        "weaknesses": list,
        "key_observations": list,
        "perspectives": dict,
        "confidence": dict
    }

    # RAG 검색 쿼리 (역량별)
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
    
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        max_concurrent: int = 5,
        max_retries: int = 5,
        db_session: Optional[Session] = None,
        use_rag: bool = False,
        rag_top_k: int = 8,
        use_feedback: bool = False,  # 🆕 피드백 루프 활성화
        job_category: Optional[str] = None  # 🆕 직무 카테고리 (피드백 검색용)
    ):
        self.client = openai_client
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.semaphore = Semaphore(max_concurrent)
        self.cache = {}
        self.max_retries = max_retries
        self.db_session = db_session
        self.use_rag = use_rag
        self.rag_top_k = rag_top_k
        self.use_feedback = use_feedback
        self.job_category = job_category
    
    def _get_cache_key(self, competency_name: str, transcript: Dict) -> str:
        """캐시 키 생성"""
        transcript_str = json.dumps(transcript, sort_keys=True, ensure_ascii=False)
        transcript_hash = hashlib.md5(transcript_str.encode()).hexdigest()
        return f"{competency_name}:{transcript_hash}"
    
    
    def _validate_and_fix_response(
        self, 
        result: Dict, 
        competency_name: str
    ) -> Dict:
        """
        LLM 응답 검증 및 필수 필드 보강
        
        Args:
            result: LLM이 반환한 JSON
            competency_name: 역량 이름
        
        Returns:
            검증 및 보강된 JSON
        """
        
        print(f"  [검증] {competency_name} 응답 검증 중...")
        
        # 1. 필수 필드 존재 여부 확인
        missing_fields = []
        for field, field_type in self.REQUIRED_FIELDS.items():
            if field not in result:
                missing_fields.append(field)
                print(f"    ⚠️  필수 필드 누락: {field}")
        
        
        # 2. key_observations 누락 시 자동 생성
        if "key_observations" not in result or not result.get("key_observations"):
            print(f"    🔧 key_observations 자동 생성 중...")
            
            # strengths, weaknesses, perspectives에서 핵심 관찰 추출
            key_obs = self._generate_key_observations(result, competency_name)
            result["key_observations"] = key_obs
            
            print(f"    ✅ key_observations 생성 완료 ({len(key_obs)}개)")
        else:
            print(f"    ✅ key_observations 존재 ({len(result['key_observations'])}개)")
        
        
        # 3. 빈 리스트 필드 경고
        if not result.get("strengths"):
            print(f"    ⚠️  strengths 비어있음")
        
        if not result.get("weaknesses"):
            print(f"    ⚠️  weaknesses 비어있음")
        
        
        # 4. 점수 범위 검증
        score = result.get("overall_score", 0)
        if not (0 <= score <= 100):
            print(f"    ⚠️  overall_score 범위 오류: {score} → 50으로 조정")
            result["overall_score"] = 50
        
        
        return result
    
    
    def _generate_key_observations(
        self,
        result: Dict,
        competency_name: str
    ) -> list:
        """
        key_observations 자동 생성

        전략:
            1. strengths/weaknesses에서 상위 3개 추출
            2. perspectives.evidence_reasoning에서 핵심 문장 추출
            3. 최소 3개 보장
        """

        key_obs = []

        # 1. Strengths에서 추출 (상위 2개)
        strengths = result.get("strengths", [])
        if strengths:
            key_obs.extend(strengths[:2])


        # 2. Weaknesses에서 추출 (상위 1개)
        weaknesses = result.get("weaknesses", [])
        if weaknesses:
            key_obs.append(weaknesses[0])


        # 3. Evidence reasoning에서 핵심 문장 추출
        perspectives = result.get("perspectives", {})
        evidence_reasoning = perspectives.get("evidence_reasoning", "")

        if evidence_reasoning:
            # "따라서", "전반적으로" 같은 키워드 뒤 문장 추출
            import re
            # "따라서 X점 산정" 같은 결론 문장 찾기
            conclusion_match = re.search(r'(따라서|전반적으로|종합하면)[^.]+\.', evidence_reasoning)
            if conclusion_match:
                conclusion = conclusion_match.group(0).strip()
                if conclusion not in key_obs:
                    key_obs.append(conclusion)


        # 4. 최소 3개 보장 (부족하면 기본 메시지 추가)
        if len(key_obs) < 3:
            score = result.get("overall_score", 0)

            # 점수 대역별 기본 관찰
            if score >= 75:
                key_obs.append(f"{competency_name} 역량이 신입 기준 우수한 수준")
            elif score >= 60:
                key_obs.append(f"{competency_name} 역량이 신입 기준 양호한 수준")
            elif score >= 50:
                key_obs.append(f"{competency_name} 역량이 신입 기준 평균 수준")
            else:
                key_obs.append(f"{competency_name} 역량이 신입 기준 미흡한 수준")


        # 5. 중복 제거 및 최대 5개로 제한
        key_obs = list(dict.fromkeys(key_obs))[:5]

        return key_obs


    async def _filter_transcript_with_rag(
        self,
        competency_name: str,
        transcript: Dict,
        session_id: int
    ) -> Dict:
        """
        RAG를 사용하여 관련 segment만 필터링한 transcript 반환

        Args:
            competency_name: 역량 이름
            transcript: 전체 transcript
            session_id: 세션 ID (DB 조회용)

        Returns:
            필터링된 transcript (관련 segment만 포함)
        """
        if not self.use_rag or not self.db_session:
            return transcript

        try:
            # RAG 검색 쿼리 가져오기
            search_query = self.COMPETENCY_SEARCH_QUERIES.get(
                competency_name,
                f"{competency_name} 관련 행동 사례"
            )

            # RAG 검색 실행
            from services.rag_embedding_service import search_relevant_transcripts

            relevant_transcripts = await search_relevant_transcripts(
                db=self.db_session,
                session_id=session_id,
                query_text=search_query,
                top_k=self.rag_top_k,
                similarity_threshold=0.3  # 최소 30% 유사도
            )

            if not relevant_transcripts:
                print(f"  ⚠️  [{competency_name}] RAG 검색 결과 없음 → 전체 transcript 사용")
                return transcript

            # 관련 segment ID 추출
            relevant_segment_ids = set([t["turn"] for t in relevant_transcripts])

            # Transcript 필터링
            filtered_segments = [
                seg for seg in transcript.get("segments", [])
                if seg.get("segment_order") in relevant_segment_ids
            ]

            # 필터링된 transcript 생성
            filtered_transcript = {
                **transcript,
                "segments": filtered_segments,
                "rag_metadata": {
                    "competency_name": competency_name,
                    "search_query": search_query,
                    "original_segment_count": len(transcript.get("segments", [])),
                    "filtered_segment_count": len(filtered_segments),
                    "token_reduction_rate": 1 - (len(filtered_segments) / max(len(transcript.get("segments", [])), 1))
                }
            }

            print(f"  🔍 [{competency_name}] RAG 필터링: {len(transcript.get('segments', []))}개 → {len(filtered_segments)}개 segment")

            return filtered_transcript

        except Exception as e:
            print(f"  ⚠️  [{competency_name}] RAG 필터링 실패: {e} → 전체 transcript 사용")
            return transcript


    async def _get_feedback_examples(
        self,
        competency_name: str,
        transcript_summary: str
    ) -> List[Dict]:
        """
        🆕 V2: Few-shot examples로 사용할 피드백 검색

        Args:
            competency_name: 역량 이름
            transcript_summary: 현재 트랜스크립트 요약

        Returns:
            Few-shot examples 리스트
        """
        if not self.use_feedback or not self.db_session or not self.job_category:
            return []

        try:
            from services.feedback.feedback_manager import FeedbackManager

            feedback_manager = FeedbackManager(db=self.db_session, openai_client=self.client)

            # 🆕 V2: Dynamic threshold 사용
            feedbacks = await feedback_manager.get_relevant_feedback(
                job_category=self.job_category,
                competency_name=competency_name,
                current_context=transcript_summary,
                top_k=2,  # Few-shot은 2개면 충분
                similarity_threshold=0.5,
                use_dynamic_threshold=True  # 🆕 동적 조정
            )

            if feedbacks:
                print(f"  💡 [{competency_name}] Few-shot examples: {len(feedbacks)}개")

            return feedbacks

        except Exception as e:
            print(f"  ⚠️  [{competency_name}] 피드백 검색 실패: {e}")
            return []

    def _build_messages_with_feedback(
        self,
        prompt: str,
        feedbacks: List[Dict]
    ) -> List[Dict]:
        """
        🆕 V2: Few-shot prompting으로 메시지 구성

        Args:
            prompt: 원본 프롬프트
            feedbacks: 피드백 목록

        Returns:
            Few-shot examples가 포함된 messages
        """
        messages = [
            {
                "role": "system",
                "content": "You are an expert HR evaluator. Learn from past corrections to improve your assessments."
            }
        ]

        # 🆕 V2: Few-shot examples 추가 (최대 2개)
        for fb in feedbacks[:2]:
            # AI의 잘못된 평가 예시
            messages.append({
                "role": "user",
                "content": f"Evaluate competency. Context: Similar to cases where AI scored {fb['ai_score']} but was corrected."
            })
            messages.append({
                "role": "assistant",
                "content": f"Initial assessment: {fb['ai_score']} points. {fb['mistake_summary']}"
            })

            # HR의 교정 피드백
            messages.append({
                "role": "user",
                "content": f"HR Correction: {fb['correction_guideline']}. Actual score should be {fb['human_score']}."
            })
            messages.append({
                "role": "assistant",
                "content": f"Understood. I will adjust my evaluation approach. Corrected score: {fb['human_score']} points."
            })

        # 실제 평가 프롬프트
        messages.append({
            "role": "user",
            "content": prompt
        })

        return messages
    
    
    async def evaluate(
        self,
        competency_name: str,
        competency_display_name: str,
        competency_category: str,
        prompt: str,
        transcript: Dict,
        session_id: Optional[int] = None
    ) -> Dict:
        """역량 평가 실행 (V2: Few-shot Prompting + 조건부 캐시)"""

        # 🆕 V2: 피드백 사용 시 캐시 비활성화
        cache_enabled = not self.use_feedback

        # 캐시 확인
        if cache_enabled:
            cache_key = self._get_cache_key(competency_name, transcript)
            if cache_key in self.cache:
                print(f"[캐시 히트] {competency_name}")
                return self.cache[cache_key]

        # Rate Limiting
        async with self.semaphore:
            print(f"[평가 시작] {competency_name}")

            # RAG 필터링 (선택적)
            if self.use_rag and session_id:
                transcript = await self._filter_transcript_with_rag(
                    competency_name=competency_name,
                    transcript=transcript,
                    session_id=session_id
                )

            # 🆕 V2: Few-shot examples 가져오기
            feedback_examples = []
            if self.use_feedback:
                # 트랜스크립트 요약 생성 (간단히 첫 3개 segment 요약)
                transcript_summary = ""
                if transcript.get("segments"):
                    segments = transcript["segments"][:3]
                    transcript_summary = " ".join([
                        f"{seg.get('question_text', '')} {seg.get('answer_text', '')}"
                        for seg in segments
                    ])[:500]  # 500자로 제한

                feedback_examples = await self._get_feedback_examples(
                    competency_name=competency_name,
                    transcript_summary=transcript_summary
                )

            # 🆕 V2: Few-shot prompting으로 메시지 구성
            if feedback_examples:
                messages = self._build_messages_with_feedback(prompt, feedback_examples)
            else:
                # 피드백 없으면 기존 방식
                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert HR evaluator. Evaluate competencies based on interview transcripts."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]

            try:
                # OpenAI 호출 (재시도 포함) - Pydantic Structured Output
                for attempt in range(self.max_retries):
                    try:
                        # 🆕 V2: Few-shot messages 사용
                        response = await self.client.beta.chat.completions.parse(
                            model=self.model,
                            messages=messages,
                            temperature=0.0,
                            max_tokens=4000,
                            response_format=CompetencyEvaluationResult
                        )

                        # Pydantic 모델로 자동 파싱됨
                        parsed_result = response.choices[0].message.parsed

                        # Pydantic 모델을 Dict로 변환
                        result = parsed_result.model_dump()

                        # 🆕 Confidence 검증 (낮으면 재시도)
                        confidence_score = result.get("confidence", {}).get("score", 1.0)
                        confidence_level = result.get("confidence", {}).get("level", "high")

                        if confidence_score < 0.6 and attempt < self.max_retries - 1:
                            print(f"  ⚠️  [{competency_name}] Low confidence ({confidence_score:.2f}) - retrying ({attempt+1}/{self.max_retries})")
                            # 재시도를 위해 continue
                            await asyncio.sleep(1)
                            continue

                        # 메타 정보 추가
                        result["competency_name"] = competency_name
                        result["competency_display_name"] = competency_display_name
                        result["competency_category"] = competency_category
                        result["evaluated_at"] = datetime.now().isoformat()

                        # 🆕 V2: 조건부 캐싱 (피드백 사용 시 캐시 안함)
                        if cache_enabled:
                            cache_key = self._get_cache_key(competency_name, transcript)
                            self.cache[cache_key] = result

                        feedback_note = f" (Few-shot: {len(feedback_examples)}개)" if feedback_examples else ""
                        print(f"[평가 완료] {competency_name}: {result.get('overall_score', 0)}점 (Confidence: {confidence_level}, {confidence_score:.2f}){feedback_note}")

                        return result
                        
                    except RateLimitError as e:
                        await self._handle_rate_limit(e, attempt, competency_name)
                        continue
                    except APIStatusError as e:
                        if e.status_code == 429:
                            await self._handle_rate_limit(e, attempt, competency_name)
                            continue
                        raise
                    except json.JSONDecodeError as e:
                        if attempt < self.max_retries - 1:
                            print(f"[재시도 {attempt+1}/{self.max_retries}] {competency_name}: JSON 파싱 오류 → 백오프 후 재시도")
                            await asyncio.sleep(1 + attempt)
                        else:
                            raise
                    except Exception as e:
                        if attempt < self.max_retries - 1:
                            print(f"[재시도 {attempt+1}/{self.max_retries}] {competency_name}: {e}")
                            await asyncio.sleep(1 + attempt)
                        else:
                            raise
                            
            except Exception as e:
                raise RuntimeError(f"[{competency_name}] 평가 실패: {e}")

    async def _handle_rate_limit(self, error: Exception, attempt: int, competency_name: str):
        """429 오류 대응: retry-after 또는 지수 백오프 기반 대기"""
        retry_after = None
        response = getattr(error, "response", None)
        if response:
            retry_after = response.headers.get("retry-after") or response.headers.get("Retry-After")
        if retry_after:
            try:
                wait_seconds = float(retry_after)
            except ValueError:
                wait_seconds = None
        else:
            wait_seconds = None
        if wait_seconds is None:
            import re
            match = re.search(r"try again in ([0-9.]+)s", str(error))
            if match:
                wait_seconds = float(match.group(1))
        if wait_seconds is None:
            wait_seconds = min(30, 2 ** attempt * 2)
        print(f"[대기] {competency_name} rate limit 감지 → {wait_seconds:.1f}s 후 재시도 ({attempt+1}/{self.max_retries})")
        await asyncio.sleep(wait_seconds)


async def evaluate_all_competencies(
    agent: CompetencyAgent,
    transcript: Dict,
    prompts: Dict[str, str],
    session_id: Optional[int] = None,
    job_competencies: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Dict]:
    """10개 역량 배치 평가 (RAG 지원 + 동적 역량)"""

    print("=" * 60)
    print("10개 역량 배치 평가 시작")
    if agent.use_rag:
        print(f"  🔍 RAG 모드 활성화 (Top-K: {agent.rag_top_k})")
    print("=" * 60)

    # Common Competencies (5개 고정)
    common_competency_configs = [
        ("achievement_motivation", "성취/동기 역량", "common"),
        ("growth_potential", "성장 잠재력", "common"),
        ("interpersonal_skill", "대인관계 역량", "common"),
        ("organizational_fit", "조직 적합성", "common"),
        ("problem_solving", "문제해결력", "common"),
    ]

    # Job Competencies (5개 동적)
    job_competency_configs = []

    if job_competencies and len(job_competencies) >= 5:
        # JD에서 추출한 역량 사용
        print(f"  ✓ Using {len(job_competencies)} job competencies from JD")
        for i, comp in enumerate(job_competencies[:5]):  # 최대 5개
            # competency 구조: {"name": "...", "category": "...", "score": ..., "description": "..."}
            comp_name = comp.get("name", f"job_comp_{i+1}")
            # 영어 key로 변환 (snake_case)
            comp_key = comp_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("&", "and")
            job_competency_configs.append((
                comp_key,
                comp_name,
                "job"
            ))
    else:
        # 기본 직무 역량 사용 (Fallback)
        print("  ⚠ Using default job competencies (no JD competencies found)")
        job_competency_configs = [
            ("customer_journey_marketing", "고객 여정 설계 및 VMD·마케팅 통합 전략", "job"),
            ("md_data_analysis", "매출·트렌드 데이터 분석 및 상품 기획", "job"),
            ("seasonal_strategy_kpi", "시즌 전략 수립 및 비즈니스 문제해결", "job"),
            ("stakeholder_collaboration", "유관부서 협업 및 이해관계자 협상", "job"),
            ("value_chain_optimization", "소싱·생산·유통 밸류체인 최적화", "job"),
        ]

    # 전체 역량 설정 (공통 5개 + 직무 5개)
    competency_configs = common_competency_configs + job_competency_configs

    # 병렬 평가 실행
    tasks = [
        agent.evaluate(name, display, category, prompts[name], transcript, session_id)
        for name, display, category in competency_configs
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    print("=" * 60)
    print("배치 평가 완료")
    print("=" * 60)
    
    # 결과 매핑
    result_dict = {}
    for (name, _, _), result in zip(competency_configs, results):
        if isinstance(result, Exception):
            print(f"[오류] {name}: {str(result)}")
            result_dict[name] = {
                "error": str(result),
                "overall_score": 0,
                "confidence": {
                    "overall_confidence": 0.3
                },
                "key_observations": [f"{name} 평가 실패"]  # 🆕 에러 시에도 필드 보장
            }
        else:
            result_dict[name] = result
    
    return result_dict
