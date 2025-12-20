"""
Transcript 하이라이트 추출 서비스

요구사항:
- 답변에서 평가에 사용된 문장 추출
- 문장의 시작/끝 인덱스 반환 (하이라이팅용)
- 키워드 밀도 기반 relevance 점수 계산
"""

from typing import Dict, List, Any, Tuple
import re
from dataclasses import dataclass, asdict


@dataclass
class HighlightedSentence:
    """하이라이트된 문장 정보"""
    text: str
    start: int  # 전체 답변에서의 시작 위치
    end: int    # 전체 답변에서의 끝 위치
    relevance: float  # 0.0 ~ 1.0 (역량 관련도)
    matched_keywords: List[str]  # 매칭된 키워드들


class HighlightExtractor:
    """
    Transcript에서 하이라이트 대상 문장 추출

    사용 시나리오:
    1. 지원자 답변 텍스트 입력
    2. 평가 역량 및 점수 입력
    3. 역량 관련 문장 자동 하이라이트
    4. 프론트엔드에서 <mark> 태그로 시각화
    """

    # 역량별 핵심 키워드 (evidence_extractor와 동일)
    COMPETENCY_KEYWORDS = {
        "strategic_thinking": [
            "전략", "장기적", "비전", "경쟁우위", "차별화",
            "시장", "분석", "포지셔닝", "로드맵", "우선순위",
            "강점", "약점", "기회", "위협", "SWOT", "경쟁사"
        ],
        "data_driven": [
            "데이터", "분석", "지표", "측정", "ROI", "전환율",
            "피벗", "SQL", "Python", "엑셀", "시각화",
            "근거", "수치", "정량", "통계", "Pandas"
        ],
        "communication": [
            "소통", "협업", "설득", "경청", "피드백",
            "조율", "합의", "논의", "공유", "보고",
            "객관적", "논리", "근거", "팀"
        ],
        "problem_solving": [
            "문제", "해결", "대안", "옵션", "리스크",
            "우선순위", "판단", "분석", "원인", "개선",
            "창의적", "혁신", "접근"
        ],
        "industry_knowledge": [
            "전기차", "배터리", "자율주행", "부품",
            "시장", "트렌드", "경쟁사", "기술",
            "업계", "산업", "현대모비스", "보쉬", "덴소"
        ],
        "learning_attitude": [
            "배우", "성장", "학습", "공부", "발전",
            "개선", "노력", "열정", "도전", "목표",
            "자세", "태도"
        ]
    }


    def extract_evidence_from_transcript(
        self,
        question: str,
        answer: str,
        competency: str,
        score: int
    ) -> Dict[str, Any]:
        """
        답변에서 특정 역량 평가에 사용된 문장 추출

        Args:
            question: 면접 질문
            answer: 지원자 답변
            competency: 평가 역량 (예: "data_driven")
            score: 해당 역량 점수 (0-100)

        Returns:
            {
                "highlighted_sentences": [
                    {
                        "text": "Python으로 코호트 분석",
                        "start": 45,
                        "end": 65,
                        "relevance": 0.95,
                        "matched_keywords": ["Python", "분석"]
                    }
                ],
                "keywords": ["Python", "Pandas", "코호트 분석"],
                "justification": "구체적인 분석 도구 활용..."
            }
        """
        # 1. 문장 분리
        sentences = self._split_sentences_with_positions(answer)

        # 2. 역량별 키워드 가져오기
        keywords = self.COMPETENCY_KEYWORDS.get(competency, [])

        # 3. 각 문장의 relevance 계산
        highlighted = []
        all_matched_keywords = set()

        for sent_text, start, end in sentences:
            matched_keywords = self._find_keywords_in_text(sent_text, keywords)

            if matched_keywords:
                # Relevance: 매칭된 키워드 수 / 전체 키워드 수
                relevance = min(len(matched_keywords) / len(keywords), 1.0)

                # Relevance 가중치 부여 (점수가 높으면 relevance도 높게)
                if score >= 80:
                    relevance = min(relevance * 1.2, 1.0)

                highlighted.append(HighlightedSentence(
                    text=sent_text,
                    start=start,
                    end=end,
                    relevance=round(relevance, 2),
                    matched_keywords=matched_keywords
                ))

                all_matched_keywords.update(matched_keywords)

        # 4. Relevance 높은 순으로 정렬
        highlighted.sort(key=lambda x: x.relevance, reverse=True)

        # 5. 평가 근거 생성
        justification = self._generate_justification(
            competency=competency,
            score=score,
            highlighted_sentences=[h.text for h in highlighted[:3]],
            keywords=list(all_matched_keywords)
        )

        return {
            "highlighted_sentences": [asdict(h) for h in highlighted],
            "keywords": sorted(list(all_matched_keywords)),
            "justification": justification,
            "total_highlighted_chars": sum(len(h.text) for h in highlighted),
            "coverage_percentage": round(
                sum(len(h.text) for h in highlighted) / len(answer) * 100, 1
            ) if answer else 0
        }


    def _split_sentences_with_positions(
        self,
        text: str
    ) -> List[Tuple[str, int, int]]:
        """
        문장 분리 + 위치 정보 반환

        Returns:
            [(sentence_text, start_index, end_index), ...]
        """
        # 한국어 문장 분리 (마침표, 느낌표, 물음표 기준)
        pattern = r'([^.!?]+[.!?])'
        matches = re.finditer(pattern, text)

        sentences = []
        for match in matches:
            sentence = match.group(0).strip()
            start = match.start()
            end = match.end()
            sentences.append((sentence, start, end))

        # 마지막 문장 처리 (마침표 없는 경우)
        if sentences:
            last_end = sentences[-1][2]
            if last_end < len(text):
                remaining = text[last_end:].strip()
                if remaining:
                    sentences.append((remaining, last_end, len(text)))
        elif text.strip():
            sentences.append((text.strip(), 0, len(text)))

        return sentences


    def _find_keywords_in_text(
        self,
        text: str,
        keywords: List[str]
    ) -> List[str]:
        """텍스트에서 키워드 찾기"""
        text_lower = text.lower()
        matched = []

        for keyword in keywords:
            if keyword.lower() in text_lower:
                matched.append(keyword)

        return matched


    def _generate_justification(
        self,
        competency: str,
        score: int,
        highlighted_sentences: List[str],
        keywords: List[str]
    ) -> str:
        """평가 근거 설명 생성"""
        competency_names = {
            "strategic_thinking": "전략적 사고",
            "data_driven": "데이터 기반 의사결정",
            "communication": "커뮤니케이션",
            "problem_solving": "문제해결력",
            "industry_knowledge": "산업 이해도",
            "learning_attitude": "학습 태도"
        }

        comp_name = competency_names.get(competency, competency)

        # 점수 수준
        if score >= 90:
            level = "매우 우수한"
        elif score >= 80:
            level = "우수한"
        elif score >= 70:
            level = "양호한"
        elif score >= 60:
            level = "보통의"
        else:
            level = "부족한"

        # 키워드 나열
        keyword_str = ", ".join([f"'{k}'" for k in keywords[:5]])

        # 근거 문장
        evidence_str = ""
        if highlighted_sentences:
            evidence_str = f"\n\n특히 다음 발언이 핵심 근거입니다:\n\"{highlighted_sentences[0][:100]}...\""

        justification = f"""
{comp_name} 역량에서 {level} 평가({score}점)를 받았습니다.

답변에서 {keyword_str} 등의 키워드가 발견되어
해당 역량을 명확히 보여주고 있습니다.
{evidence_str}
        """.strip()

        return justification


    def extract_all_highlights_from_qa_pairs(
        self,
        qa_pairs: List[Dict[str, Any]],
        competency_scores: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        전체 QA에서 모든 하이라이트 추출

        Args:
            qa_pairs: [{"question": "...", "answer": "...", "target_competencies": [...]}, ...]
            competency_scores: {"data_driven": 85, ...}

        Returns:
            {
                "highlights_by_qa": {
                    "q1": {
                        "data_driven": {
                            "highlighted_sentences": [...],
                            "keywords": [...]
                        }
                    }
                },
                "global_keywords": {
                    "data_driven": ["Python", "데이터", ...]
                }
            }
        """
        highlights_by_qa = {}
        global_keywords = {comp: set() for comp in competency_scores.keys()}

        for qa in qa_pairs:
            qa_id = qa.get("id", "unknown")
            qa_highlights = {}

            for competency in qa.get("target_competencies", []):
                if competency in competency_scores:
                    result = self.extract_evidence_from_transcript(
                        question=qa["question"],
                        answer=qa["answer"],
                        competency=competency,
                        score=competency_scores[competency]
                    )

                    qa_highlights[competency] = result
                    global_keywords[competency].update(result["keywords"])

            if qa_highlights:
                highlights_by_qa[qa_id] = qa_highlights

        # Set to list
        global_keywords = {
            comp: sorted(list(keywords))
            for comp, keywords in global_keywords.items()
        }

        return {
            "highlights_by_qa": highlights_by_qa,
            "global_keywords": global_keywords
        }
