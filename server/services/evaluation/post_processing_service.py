"""
후처리 서비스

지민 에이전트가 산출한 Aggregator 결과를 입력으로,
프론트/Swagger에서 요구하는 추가 필드(긍/부 키워드, 추천 질문, 전체 요약)를 생성합니다.

설계 원칙 (간단 Rule 기반, LLM 옵션 없음):
- strengths/weaknesses, 점수, 역량명을 조합해 키워드/추천질문을 만든다.
- 항상 최소 1개 이상의 값을 반환하도록 안전한 기본값을 둔다.
- 구조는 server/test_data/evaluation_response_sample.json의 analysis_summary 블록을 따른다.
"""

from typing import Dict, List, Any, Tuple


class PostProcessingService:
    def __init__(
        self,
        positive_top_k: int = 4,
        negative_top_k: int = 4,
        question_top_k: int = 3,
        low_score_threshold: float = 75.0
    ):
        self.positive_top_k = positive_top_k
        self.negative_top_k = negative_top_k
        self.question_top_k = question_top_k
        self.low_score_threshold = low_score_threshold

    def build_analysis_summary(
        self,
        aggregated_competencies: Dict[str, Dict[str, Any]],
        final_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Aggregator 결과와 Final Result를 받아 후처리 필드를 생성한다.
        Returns:
            {
              "aggregator_summary": "...",
              "overall_applicant_summary": "...",
              "positive_keywords": [...],
              "negative_keywords": [...],
              "recommended_questions": [...]
            }
        """
        strengths, weaknesses = self._collect_strengths_weaknesses(aggregated_competencies)
        positive_keywords = self._build_positive_keywords(strengths, aggregated_competencies)
        negative_keywords = self._build_negative_keywords(weaknesses)
        recommended_questions = self._build_recommended_questions(aggregated_competencies, weaknesses)
        overall_applicant_summary = self._build_overall_summary(
            strengths, weaknesses, final_result
        )

        aggregator_summary = final_result.get("overall_evaluation_summary", "")

        return {
            "aggregator_summary": aggregator_summary,
            "overall_applicant_summary": overall_applicant_summary,
            "positive_keywords": positive_keywords,
            "negative_keywords": negative_keywords,
            "recommended_questions": recommended_questions,
        }

    def _collect_strengths_weaknesses(
        self, aggregated_competencies: Dict[str, Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        strengths: List[str] = []
        weaknesses: List[str] = []
        for comp in aggregated_competencies.values():
            comp_data = comp or {}
            comp_strengths = comp_data.get("strengths") or []
            comp_weaknesses = comp_data.get("weaknesses") or []

            if not isinstance(comp_strengths, list):
                comp_strengths = [comp_strengths]
            if not isinstance(comp_weaknesses, list):
                comp_weaknesses = [comp_weaknesses]

            strengths.extend(comp_strengths)
            weaknesses.extend(comp_weaknesses)
        # 중복 제거, 순서 유지
        strengths = list(dict.fromkeys(strengths))
        weaknesses = list(dict.fromkeys(weaknesses))
        return strengths, weaknesses

    def _build_positive_keywords(
        self,
        strengths: List[str],
        aggregated_competencies: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        keywords: List[str] = []
        keywords.extend(strengths[: self.positive_top_k])

        # 높은 점수 역량명을 키워드로 추가
        high_scores = sorted(
            [
                (comp_name, (comp_data or {}).get("overall_score", 0))
                for comp_name, comp_data in aggregated_competencies.items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        for comp_name, score in high_scores[:2]:
            keywords.append(f"{comp_name}_strong_{int(score)}")

        keywords = [kw for kw in keywords if kw]
        keywords = list(dict.fromkeys(keywords))[: self.positive_top_k]
        if not keywords:
            keywords = ["강점 미포착"]
        return keywords

    def _build_negative_keywords(self, weaknesses: List[str]) -> List[str]:
        keywords = weaknesses[: self.negative_top_k]
        keywords = [kw for kw in keywords if kw]
        keywords = list(dict.fromkeys(keywords))[: self.negative_top_k]
        if not keywords:
            keywords = ["명확한 약점 없음"]
        return keywords

    def _build_recommended_questions(
        self,
        aggregated_competencies: Dict[str, Dict[str, Any]],
        weaknesses: List[str],
    ) -> List[str]:
        # 점수가 낮은 역량을 우선으로 질문을 만든다.
        low_comps = sorted(
            [
                (comp_name, (comp_data or {}).get("overall_score", 0), comp_data or {})
                for comp_name, comp_data in aggregated_competencies.items()
                if (comp_data or {}).get("overall_score", 100) < self.low_score_threshold
            ],
            key=lambda x: x[1],
        )

        questions: List[str] = []
        for comp_name, score, comp_data in low_comps:
            weaknesses_raw = comp_data.get("weaknesses") or []
            if not isinstance(weaknesses_raw, list):
                weaknesses_raw = [weaknesses_raw]
            weakness_snippet = weaknesses_raw[0] if weaknesses_raw else "추가 설명 필요"
            questions.append(
                f"{comp_name}({int(score)}점) 보완 사례를 구체적 지표와 함께 설명해주세요: {weakness_snippet}"
            )

        # 약점 키워드 기반 보강 질문
        for wk in weaknesses[:2]:
            questions.append(f"위 내용과 관련해 더 구체적 사례와 지표를 알려주세요: {wk}")

        questions = [q for q in questions if q]
        questions = list(dict.fromkeys(questions))[: self.question_top_k]
        if not questions:
            questions = ["추가로 확인할 질문이 필요하지 않습니다."]
        return questions

    def _build_overall_summary(
        self,
        strengths: List[str],
        weaknesses: List[str],
        final_result: Dict[str, Any],
    ) -> str:
        final_score = final_result.get("final_score")
        confidence = final_result.get("avg_confidence")

        top_strengths = "; ".join(strengths[:2]) if strengths else "강점 표현 없음"
        top_weakness = weaknesses[0] if weaknesses else "약점 표현 없음"

        parts = [
            f"종합 점수 {final_score}점, 평균 신뢰도 {confidence} 기준 평가.",
            f"주요 강점: {top_strengths}.",
            f"보완 필요: {top_weakness}.",
        ]
        return " ".join(parts)
