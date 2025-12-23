"""
평가 근거 추출 서비스

멘토 요구사항:
"알고리즘 산출 과정이 아니라, 실제 면접 내용을 보여줘야 신뢰를 얻는다"
"어떤 질문에 어떤 응답을 했는지 그 자체를 근거로"

이 서비스는 Transcript에서 평가에 사용된 구체적인 증거를 추출합니다.
"""

from typing import Dict, List, Any, Optional, Tuple
import re
from dataclasses import dataclass


@dataclass
class HighlightRange:
    """하이라이트 범위"""
    start: int
    end: int
    text: str
    keyword: str
    sentiment: str  # 'positive' or 'negative'


@dataclass
class CompetencyEvidence:
    """역량별 평가 근거"""
    competency: str
    score: float
    evidence_sentences: List[str]
    positive_keywords: List[str]
    negative_keywords: List[str]
    highlight_ranges: List[HighlightRange]
    justification: str


class EvidenceExtractor:
    """
    Transcript에서 평가 근거를 추출하는 서비스

    핵심 기능:
    1. 답변에서 역량 평가에 사용된 문장 추출
    2. 긍정/부정 키워드 식별
    3. 하이라이팅 범위 계산
    4. 평가 근거 설명 생성
    """

    # 역량별 핵심 키워드 정의
    COMPETENCY_KEYWORDS = {
        "strategic_thinking": {
            "positive": [
                "전략", "장기적", "비전", "경쟁우위", "차별화",
                "시장", "분석", "포지셔닝", "로드맵", "우선순위",
                "강점", "약점", "기회", "위협", "SWOT"
            ],
            "negative": [
                "단기", "즉흥적", "계획 없이", "생각 안 해봤", "잘 모르"
            ]
        },
        "data_driven": {
            "positive": [
                "데이터", "분석", "지표", "측정", "ROI", "전환율",
                "피벗", "SQL", "Python", "엑셀", "시각화",
                "근거", "수치", "정량", "통계"
            ],
            "negative": [
                "감으로", "느낌상", "추측", "아마도", "생각에는"
            ]
        },
        "communication": {
            "positive": [
                "소통", "협업", "설득", "경청", "피드백",
                "조율", "합의", "논의", "공유", "보고",
                "객관적", "논리", "근거"
            ],
            "negative": [
                "혼자", "무시", "강요", "일방적", "듣지 않고"
            ]
        },
        "problem_solving": {
            "positive": [
                "문제", "해결", "대안", "옵션", "리스크",
                "우선순위", "판단", "분석", "원인", "개선",
                "창의적", "혁신"
            ],
            "negative": [
                "포기", "회피", "방치", "막막", "모르겠"
            ]
        },
        "industry_knowledge": {
            "positive": [
                "전기차", "배터리", "자율주행", "부품",
                "시장", "트렌드", "경쟁사", "기술",
                "업계", "산업"
            ],
            "negative": [
                "잘 모름", "들은 것", "확실하지 않", "생각 안 해"
            ]
        },
        "learning_attitude": {
            "positive": [
                "배우", "성장", "학습", "공부", "발전",
                "개선", "노력", "열정", "도전", "목표"
            ],
            "negative": [
                "충분", "완벽", "더 배울 것 없", "이미"
            ]
        }
    }

    # 역량 한글 매핑
    COMPETENCY_KOR = {
        "strategic_thinking": "전략적 사고",
        "data_driven": "데이터 기반 의사결정",
        "communication": "커뮤니케이션",
        "problem_solving": "문제해결력",
        "industry_knowledge": "산업 이해도",
        "learning_attitude": "학습 태도"
    }


    def extract_evidence_for_competency(
        self,
        answer_text: str,
        competency: str,
        score: float,
        question_text: Optional[str] = None
    ) -> CompetencyEvidence:
        """
        특정 역량 평가에 사용된 증거 추출

        Args:
            answer_text: 지원자 답변
            competency: 평가 역량 (예: 'data_driven')
            score: 평가 점수 (0-100)
            question_text: 질문 내용 (선택)

        Returns:
            CompetencyEvidence 객체
        """
        # 1. 키워드 정의 가져오기
        keywords_config = self.COMPETENCY_KEYWORDS.get(
            competency,
            {"positive": [], "negative": []}
        )

        # 2. 문장 단위로 분리
        sentences = self._split_sentences(answer_text)

        # 3. 각 문장에서 키워드 매칭
        evidence_sentences = []
        positive_keywords = []
        negative_keywords = []

        for sentence in sentences:
            pos_matches, neg_matches = self._find_keywords_in_sentence(
                sentence,
                keywords_config
            )

            if pos_matches or neg_matches:
                evidence_sentences.append(sentence)
                positive_keywords.extend(pos_matches)
                negative_keywords.extend(neg_matches)

        # 4. 하이라이트 범위 계산
        highlight_ranges = self._calculate_highlight_ranges(
            answer_text,
            positive_keywords,
            negative_keywords
        )

        # 5. 평가 근거 설명 생성
        justification = self._generate_justification(
            competency=competency,
            score=score,
            evidence_sentences=evidence_sentences,
            positive_keywords=positive_keywords,
            negative_keywords=negative_keywords
        )

        return CompetencyEvidence(
            competency=competency,
            score=score,
            evidence_sentences=evidence_sentences,
            positive_keywords=list(set(positive_keywords)),
            negative_keywords=list(set(negative_keywords)),
            highlight_ranges=highlight_ranges,
            justification=justification
        )


    def extract_all_evidences(
        self,
        qa_pairs: List[Dict[str, Any]],
        competency_scores: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        전체 QA에서 모든 역량별 증거 추출

        Args:
            qa_pairs: [{"question": "...", "answer": "...", "competencies": [...]}]
            competency_scores: {"data_driven": 85, "communication": 90, ...}

        Returns:
            역량별 증거 리스트
        """
        all_evidences = []

        for competency, score in competency_scores.items():
            # 해당 역량을 평가한 QA 찾기
            relevant_answers = []

            for qa in qa_pairs:
                if competency in qa.get("target_competencies", []):
                    relevant_answers.append({
                        "question": qa["question"],
                        "answer": qa["answer"]
                    })

            # 모든 관련 답변 합치기
            combined_answer = "\n".join([a["answer"] for a in relevant_answers])

            if combined_answer.strip():
                evidence = self.extract_evidence_for_competency(
                    answer_text=combined_answer,
                    competency=competency,
                    score=score
                )

                all_evidences.append({
                    "competency": competency,
                    "competency_name": self.COMPETENCY_KOR.get(competency, competency),
                    "score": score,
                    "evidence_sentences": evidence.evidence_sentences[:3],  # Top 3
                    "positive_keywords": evidence.positive_keywords[:5],  # Top 5
                    "negative_keywords": evidence.negative_keywords[:3],
                    "justification": evidence.justification,
                    "relevant_qa": relevant_answers
                })

        return all_evidences


    def extract_strengths_weaknesses(
        self,
        competency_scores: Dict[str, float],
        evidences: List[Dict[str, Any]],
        top_n: int = 3
    ) -> Dict[str, Any]:
        """
        강점/약점 Top N 추출

        Args:
            competency_scores: 역량별 점수
            evidences: 역량별 증거
            top_n: 상위 몇 개 (기본 3)

        Returns:
            {
                "strengths": [
                    {
                        "competency": "data_driven",
                        "competency_name": "데이터 기반 의사결정",
                        "score": 90,
                        "summary": "엑셀, 피벗테이블 등을 활용한 데이터 분석 경험",
                        "evidence": "6개월치 광고 데이터를 수집해서..."
                    }
                ],
                "weaknesses": [...]
            }
        """
        # 점수 기준 정렬
        sorted_scores = sorted(
            competency_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 강점: 상위 N개 (70점 이상)
        strengths = []
        for comp, score in sorted_scores[:top_n]:
            if score >= 70:
                evidence_data = next(
                    (e for e in evidences if e["competency"] == comp),
                    None
                )

                if evidence_data:
                    strengths.append({
                        "competency": comp,
                        "competency_name": self.COMPETENCY_KOR.get(comp, comp),
                        "score": score,
                        "summary": self._generate_strength_summary(evidence_data),
                        "evidence": evidence_data["evidence_sentences"][0] if evidence_data["evidence_sentences"] else "",
                        "keywords": evidence_data["positive_keywords"][:3]
                    })

        # 약점: 하위 N개 (60점 이하)
        weaknesses = []
        for comp, score in sorted_scores[-top_n:]:
            if score <= 60:
                evidence_data = next(
                    (e for e in evidences if e["competency"] == comp),
                    None
                )

                weaknesses.append({
                    "competency": comp,
                    "competency_name": self.COMPETENCY_KOR.get(comp, comp),
                    "score": score,
                    "summary": self._generate_weakness_summary(comp, score),
                    "improvement_suggestion": self._generate_improvement_suggestion(comp),
                    "follow_up_question": self._generate_follow_up_question(comp)
                })

        return {
            "strengths": strengths,
            "weaknesses": weaknesses
        }


    # ==================== Private Helper Methods ====================

    def _split_sentences(self, text: str) -> List[str]:
        """문장 단위로 분리"""
        # 한국어 문장 분리 (마침표, 느낌표, 물음표 기준)
        sentences = re.split(r'[.!?]\s+', text)
        return [s.strip() for s in sentences if s.strip()]


    def _find_keywords_in_sentence(
        self,
        sentence: str,
        keywords_config: Dict[str, List[str]]
    ) -> Tuple[List[str], List[str]]:
        """문장에서 키워드 찾기"""
        positive_matches = []
        negative_matches = []

        sentence_lower = sentence.lower()

        for keyword in keywords_config.get("positive", []):
            if keyword.lower() in sentence_lower:
                positive_matches.append(keyword)

        for keyword in keywords_config.get("negative", []):
            if keyword.lower() in sentence_lower:
                negative_matches.append(keyword)

        return positive_matches, negative_matches


    def _calculate_highlight_ranges(
        self,
        text: str,
        positive_keywords: List[str],
        negative_keywords: List[str]
    ) -> List[HighlightRange]:
        """하이라이트 범위 계산"""
        ranges = []

        for keyword in set(positive_keywords):
            for match in re.finditer(re.escape(keyword), text, re.IGNORECASE):
                ranges.append(HighlightRange(
                    start=match.start(),
                    end=match.end(),
                    text=match.group(),
                    keyword=keyword,
                    sentiment='positive'
                ))

        for keyword in set(negative_keywords):
            for match in re.finditer(re.escape(keyword), text, re.IGNORECASE):
                ranges.append(HighlightRange(
                    start=match.start(),
                    end=match.end(),
                    text=match.group(),
                    keyword=keyword,
                    sentiment='negative'
                ))

        # 시작 위치 기준 정렬
        ranges.sort(key=lambda r: r.start)
        return ranges


    def _generate_justification(
        self,
        competency: str,
        score: float,
        evidence_sentences: List[str],
        positive_keywords: List[str],
        negative_keywords: List[str]
    ) -> str:
        """평가 근거 설명 생성"""
        comp_name = self.COMPETENCY_KOR.get(competency, competency)

        # 점수 수준 판단
        if score >= 90:
            level = "매우 뛰어난"
        elif score >= 80:
            level = "우수한"
        elif score >= 70:
            level = "양호한"
        elif score >= 60:
            level = "보통의"
        else:
            level = "부족한"

        # 키워드 요약
        pos_keywords_str = ", ".join([f"'{k}'" for k in positive_keywords[:3]])

        # 근거 문장
        evidence_snippet = ""
        if evidence_sentences:
            evidence_snippet = f"\n\n특히 다음 발언이 핵심 근거입니다:\n\"{evidence_sentences[0][:100]}...\""

        justification = f"""
{comp_name} 역량에서 {level} 평가({score:.0f}점)를 받았습니다.

답변에서 {pos_keywords_str} 등의 키워드가 발견되어
해당 역량을 보유하고 있음을 확인할 수 있습니다.
{evidence_snippet}
        """.strip()

        # 부정 키워드가 있으면 언급
        if negative_keywords:
            neg_str = ", ".join([f"'{k}'" for k in negative_keywords[:2]])
            justification += f"\n\n다만, {neg_str} 등의 표현이 있어 추가 확인이 필요할 수 있습니다."

        return justification


    def _generate_strength_summary(self, evidence_data: Dict) -> str:
        """강점 요약 생성"""
        keywords = evidence_data.get("positive_keywords", [])[:3]
        if keywords:
            return f"{', '.join(keywords)} 등을 활용한 경험이 뛰어남"
        return "해당 역량에서 우수한 역량 보유"


    def _generate_weakness_summary(self, competency: str, score: float) -> str:
        """약점 요약 생성"""
        comp_name = self.COMPETENCY_KOR.get(competency, competency)
        return f"{comp_name} 영역에서 실무 경험 부족 ({score:.0f}점)"


    def _generate_improvement_suggestion(self, competency: str) -> str:
        """개선 제안 생성"""
        suggestions = {
            "strategic_thinking": "전략 수립 프로젝트 참여 또는 사례 연구 학습 권장",
            "data_driven": "Python, SQL 등 데이터 분석 도구 실무 경험 확대 필요",
            "communication": "다양한 이해관계자와의 협업 경험 쌓기",
            "problem_solving": "복잡한 문제 상황에서의 의사결정 경험 필요",
            "industry_knowledge": "자동차/전기차 산업 트렌드 및 기술 학습 필요",
            "learning_attitude": "구체적인 학습 계획 및 실행력 강화"
        }
        return suggestions.get(competency, "해당 영역의 실무 경험 확대 권장")


    def _generate_follow_up_question(self, competency: str) -> str:
        """후속 질문 생성"""
        questions = {
            "strategic_thinking": "5년 후 이 산업의 모습을 어떻게 예상하시나요? 그에 대한 대응 전략은?",
            "data_driven": "실제 데이터 분석 프로젝트에서 어떤 인사이트를 도출한 경험이 있나요?",
            "communication": "이해관계가 상충하는 상황에서 합의를 이끌어낸 경험을 구체적으로 말씀해주세요.",
            "problem_solving": "기존 방식으로 해결 불가능한 문제를 어떻게 돌파했는지 사례를 들어주세요.",
            "industry_knowledge": "우리 회사의 주요 경쟁사와 비교했을 때 강점/약점은 무엇이라고 생각하나요?",
            "learning_attitude": "최근 3개월간 새롭게 학습한 것과 그것을 어떻게 적용했는지 말씀해주세요."
        }
        return questions.get(competency, "해당 영역에서의 구체적인 경험을 더 자세히 말씀해주세요.")
