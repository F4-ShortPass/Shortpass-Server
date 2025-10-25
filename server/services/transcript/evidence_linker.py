"""
평가 점수와 근거 연결 서비스

요구사항:
- 특정 역량 점수가 어떤 QA에서 나왔는지 추적
- 점수 → 질문 → 답변 → 하이라이트 → 평가 이유의 체인 구성
- HR이 "왜 이 점수인가?"를 즉시 확인 가능하도록
"""

from typing import Dict, List, Any, Optional
from services.transcript.highlight_extractor import HighlightExtractor


class EvidenceLinker:
    """
    점수와 근거를 연결하는 서비스

    사용 시나리오:
    1. HR이 "데이터 분석력 90점" 클릭
    2. 해당 점수를 받은 이유를 QA 근거로 보여줌
    3. 질문-답변-하이라이트-평가 이유 체인 제공
    """

    def __init__(self):
        self.highlight_extractor = HighlightExtractor()


    def link_score_to_evidence(
        self,
        competency: str,
        score: int,
        transcript: List[Dict[str, Any]],
        highlighted_sentences: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        점수와 근거를 연결

        Args:
            competency: 역량명 (예: "data_driven")
            score: 점수 (0-100)
            transcript: 전체 면접 transcript
            highlighted_sentences: 이미 추출된 하이라이트 (선택)

        Returns:
            {
                "competency": "data_driven",
                "competency_name": "데이터 기반 의사결정",
                "score": 90,
                "evidence_chain": [
                    {
                        "qa_id": "q3",
                        "question": "데이터를 활용한 경험은?",
                        "answer_excerpt": "6개월치 광고 데이터를 수집...",
                        "highlight": "피벗 테이블을 만들고, 채널별 전환율과 ROI를 계산",
                        "keywords": ["피벗", "전환율", "ROI"],
                        "reasoning": "구체적인 데이터 분석 도구와 지표 활용"
                    }
                ],
                "score_breakdown": {
                    "positive_factors": ["구체적 도구 활용", "정량적 지표 제시"],
                    "negative_factors": ["실무 프로젝트 경험 제한적"],
                    "overall_reasoning": "인턴십 수준의 데이터 분석 경험이 있으나..."
                }
            }
        """
        competency_names = {
            "strategic_thinking": "전략적 사고",
            "data_driven": "데이터 기반 의사결정",
            "communication": "커뮤니케이션",
            "problem_solving": "문제해결력",
            "industry_knowledge": "산업 이해도",
            "learning_attitude": "학습 태도"
        }

        comp_name = competency_names.get(competency, competency)

        # 1. 해당 역량을 평가한 QA 찾기
        relevant_qas = [
            qa for qa in transcript
            if competency in qa.get("target_competencies", [])
        ]

        # 2. Evidence Chain 구성
        evidence_chain = []

        for qa in relevant_qas:
            qa_id = qa.get("id", "unknown")
            question = qa.get("question_text", "")
            answer = qa.get("answer_text", "")

            # 하이라이트 추출
            if highlighted_sentences is None:
                highlight_result = self.highlight_extractor.extract_evidence_from_transcript(
                    question=question,
                    answer=answer,
                    competency=competency,
                    score=score
                )
                highlights = highlight_result["highlighted_sentences"]
                keywords = highlight_result["keywords"]
            else:
                highlights = highlighted_sentences
                keywords = []

            # Top 하이라이트 선택
            top_highlight = highlights[0] if highlights else None

            if top_highlight:
                evidence_chain.append({
                    "qa_id": qa_id,
                    "question": question[:150] + "..." if len(question) > 150 else question,
                    "answer_excerpt": answer[:200] + "..." if len(answer) > 200 else answer,
                    "highlight": top_highlight["text"],
                    "keywords": keywords[:5],
                    "reasoning": self._generate_reasoning_for_highlight(
                        competency, score, top_highlight["text"], keywords
                    )
                })

        # 3. 점수 분해 분석
        score_breakdown = self._analyze_score_breakdown(
            competency, score, evidence_chain
        )

        return {
            "competency": competency,
            "competency_name": comp_name,
            "score": score,
            "evidence_chain": evidence_chain,
            "score_breakdown": score_breakdown
        }


    def _generate_reasoning_for_highlight(
        self,
        competency: str,
        score: int,
        highlight_text: str,
        keywords: List[str]
    ) -> str:
        """하이라이트 문장에 대한 평가 이유 생성"""

        # 키워드 기반 reasoning
        keyword_str = ", ".join([f"'{k}'" for k in keywords[:3]])

        if score >= 80:
            return f"{keyword_str} 등 핵심 키워드를 사용하여 명확한 역량 보유를 입증"
        elif score >= 70:
            return f"{keyword_str} 등이 언급되었으나 실무 경험의 깊이는 제한적"
        else:
            return f"관련 키워드가 일부 있으나 구체성과 실무 경험 부족"


    def _analyze_score_breakdown(
        self,
        competency: str,
        score: int,
        evidence_chain: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """점수 분해 분석 (왜 이 점수를 받았는지)"""

        positive_factors = []
        negative_factors = []

        # 점수 구간별 기본 분석
        if score >= 90:
            positive_factors.append("해당 역량에서 매우 우수한 실무 경험 보유")
            positive_factors.append("구체적이고 명확한 사례 제시")
        elif score >= 80:
            positive_factors.append("해당 역량에서 우수한 수준")
            positive_factors.append("실무 적용 가능한 경험 있음")
        elif score >= 70:
            positive_factors.append("기본적인 역량 보유")
            negative_factors.append("실무 경험의 깊이가 다소 제한적")
        elif score >= 60:
            positive_factors.append("역량에 대한 이해는 있음")
            negative_factors.append("실무 경험 부족")
            negative_factors.append("구체적 사례 제시 미흡")
        else:
            negative_factors.append("해당 역량에서 명확한 경험 확인 어려움")
            negative_factors.append("추가 학습 및 경험 필요")

        # Evidence chain 기반 분석
        total_keywords = set()
        for evidence in evidence_chain:
            total_keywords.update(evidence.get("keywords", []))

        if len(total_keywords) > 5:
            positive_factors.append(f"다양한 관련 키워드 사용 ({len(total_keywords)}개)")
        elif len(total_keywords) < 3:
            negative_factors.append("관련 키워드 언급 부족")

        # Overall reasoning
        overall = self._generate_overall_reasoning(competency, score, positive_factors, negative_factors)

        return {
            "positive_factors": positive_factors,
            "negative_factors": negative_factors,
            "overall_reasoning": overall
        }


    def _generate_overall_reasoning(
        self,
        competency: str,
        score: int,
        positive_factors: List[str],
        negative_factors: List[str]
    ) -> str:
        """종합 평가 이유 생성"""

        competency_names = {
            "data_driven": "데이터 기반 의사결정",
            "strategic_thinking": "전략적 사고",
            "communication": "커뮤니케이션",
            "problem_solving": "문제해결력",
            "industry_knowledge": "산업 이해도",
            "learning_attitude": "학습 태도"
        }

        comp_name = competency_names.get(competency, competency)

        overall = f"{comp_name} 역량에서 {score}점을 받은 이유는 다음과 같습니다:\n\n"

        if positive_factors:
            overall += "✅ 긍정 요소:\n"
            for factor in positive_factors:
                overall += f"  - {factor}\n"

        if negative_factors:
            overall += "\n⚠️ 보완 필요:\n"
            for factor in negative_factors:
                overall += f"  - {factor}\n"

        return overall.strip()


    def create_evidence_summary_for_all_competencies(
        self,
        transcript: List[Dict[str, Any]],
        competency_scores: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        전체 역량에 대한 근거 요약

        Args:
            transcript: 전체 면접 transcript
            competency_scores: {"data_driven": 85, ...}

        Returns:
            {
                "data_driven": {
                    "score": 85,
                    "evidence_chain": [...],
                    "score_breakdown": {...}
                },
                ...
            }
        """
        summary = {}

        for competency, score in competency_scores.items():
            evidence = self.link_score_to_evidence(
                competency=competency,
                score=score,
                transcript=transcript
            )
            summary[competency] = evidence

        return summary
