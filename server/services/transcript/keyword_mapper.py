"""
키워드 매핑 서비스

요구사항:
- Transcript에서 역량별 키워드 출현 빈도 분석
- 키워드가 나타난 문맥(context) 저장
- 시각화용 데이터 구조 제공 (프론트엔드 태그 클라우드용)
"""

from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
import re


class KeywordMapper:
    """
    Transcript에서 역량별 키워드 매핑 및 분석

    사용 시나리오:
    1. 전체 면접 transcript 입력
    2. 역량별로 관련 키워드 자동 추출
    3. 빈도 및 문맥 분석
    4. 프론트엔드에서 태그 클라우드로 시각화
    """

    # 역량별 핵심 키워드 (확장 가능)
    COMPETENCY_KEYWORDS = {
        "strategic_thinking": [
            "전략", "장기적", "비전", "경쟁우위", "차별화",
            "시장", "분석", "포지셔닝", "로드맵", "우선순위",
            "강점", "약점", "기회", "위협", "SWOT", "경쟁사",
            "글로벌", "진출", "확대", "성장", "목표"
        ],
        "data_driven": [
            "데이터", "분석", "지표", "측정", "ROI", "전환율",
            "피벗", "SQL", "Python", "엑셀", "시각화",
            "근거", "수치", "정량", "통계", "Pandas",
            "쿼리", "대시보드", "리포트", "인사이트"
        ],
        "communication": [
            "소통", "협업", "설득", "경청", "피드백",
            "조율", "합의", "논의", "공유", "보고",
            "객관적", "논리", "근거", "팀", "프로젝트",
            "의견", "대화", "회의", "발표"
        ],
        "problem_solving": [
            "문제", "해결", "대안", "옵션", "리스크",
            "우선순위", "판단", "분석", "원인", "개선",
            "창의적", "혁신", "접근", "방법", "전환",
            "돌파", "극복", "해결책"
        ],
        "industry_knowledge": [
            "전기차", "배터리", "자율주행", "부품",
            "시장", "트렌드", "경쟁사", "기술",
            "업계", "산업", "현대모비스", "보쉬", "덴소",
            "콘티넨탈", "완성차", "공급망", "ESG"
        ],
        "learning_attitude": [
            "배우", "성장", "학습", "공부", "발전",
            "개선", "노력", "열정", "도전", "목표",
            "자세", "태도", "경험", "실무", "익히"
        ]
    }


    def map_keywords_to_competencies(
        self,
        transcript: List[Dict[str, Any]],
        competencies: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Transcript에서 역량별 키워드 매핑

        Args:
            transcript: [{"question": "...", "answer": "...", "id": "q1"}, ...]
            competencies: ["data_driven", "communication", ...]

        Returns:
            {
                "data_driven": [
                    {
                        "keyword": "Python",
                        "count": 3,
                        "context": ["문장1", "문장2", "문장3"],
                        "qa_ids": ["q3", "q4"]
                    },
                    {
                        "keyword": "데이터",
                        "count": 5,
                        "context": ["...", "..."],
                        "qa_ids": ["q3", "q5"]
                    }
                ]
            }
        """
        keyword_map = {}

        for competency in competencies:
            # 해당 역량의 키워드 목록
            keywords = self.COMPETENCY_KEYWORDS.get(competency, [])

            # 키워드별 카운터 및 문맥 저장
            keyword_data = defaultdict(lambda: {
                "count": 0,
                "context": [],
                "qa_ids": set()
            })

            # Transcript 순회
            for qa in transcript:
                answer = qa.get("answer", "")
                qa_id = qa.get("id", "unknown")

                # 문장 단위로 분리
                sentences = self._split_sentences(answer)

                for sentence in sentences:
                    for keyword in keywords:
                        if keyword.lower() in sentence.lower():
                            keyword_data[keyword]["count"] += sentence.lower().count(keyword.lower())
                            keyword_data[keyword]["context"].append(sentence[:100])  # 문맥 100자
                            keyword_data[keyword]["qa_ids"].add(qa_id)

            # 결과 정리 (빈도순 정렬)
            result = []
            for keyword, data in keyword_data.items():
                if data["count"] > 0:
                    result.append({
                        "keyword": keyword,
                        "count": data["count"],
                        "context": list(set(data["context"]))[:3],  # 중복 제거 후 Top 3
                        "qa_ids": list(data["qa_ids"])
                    })

            # 빈도 높은 순으로 정렬
            result.sort(key=lambda x: x["count"], reverse=True)

            keyword_map[competency] = result

        return keyword_map


    def get_top_keywords_summary(
        self,
        keyword_map: Dict[str, List[Dict[str, Any]]],
        top_n: int = 10
    ) -> Dict[str, List[str]]:
        """
        역량별 Top N 키워드 요약

        Args:
            keyword_map: map_keywords_to_competencies 결과
            top_n: 상위 몇 개

        Returns:
            {
                "data_driven": ["데이터", "Python", "분석", ...],
                "communication": ["소통", "협업", ...]
            }
        """
        summary = {}

        for competency, keywords in keyword_map.items():
            top_keywords = [k["keyword"] for k in keywords[:top_n]]
            summary[competency] = top_keywords

        return summary


    def generate_tag_cloud_data(
        self,
        keyword_map: Dict[str, List[Dict[str, Any]]],
        competency: str
    ) -> List[Dict[str, Any]]:
        """
        태그 클라우드용 데이터 생성 (프론트엔드 시각화용)

        Args:
            keyword_map: map_keywords_to_competencies 결과
            competency: 특정 역량

        Returns:
            [
                {"text": "Python", "value": 10, "color": "#1f77b4"},
                {"text": "데이터", "value": 15, "color": "#ff7f0e"}
            ]
        """
        keywords = keyword_map.get(competency, [])

        # 색상 팔레트
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

        tag_cloud_data = []
        for i, kw_data in enumerate(keywords[:20]):  # Top 20
            tag_cloud_data.append({
                "text": kw_data["keyword"],
                "value": kw_data["count"],
                "color": colors[i % len(colors)]
            })

        return tag_cloud_data


    def analyze_keyword_trends(
        self,
        transcript: List[Dict[str, Any]],
        keyword: str
    ) -> Dict[str, Any]:
        """
        특정 키워드의 출현 패턴 분석

        Args:
            transcript: 전체 면접 transcript
            keyword: 분석할 키워드

        Returns:
            {
                "keyword": "데이터",
                "total_count": 8,
                "appearances": [
                    {"qa_id": "q3", "count": 3, "context": "..."},
                    {"qa_id": "q5", "count": 2, "context": "..."}
                ],
                "first_mention": "q3",
                "last_mention": "q8"
            }
        """
        appearances = []
        first_mention = None
        last_mention = None

        for qa in transcript:
            qa_id = qa.get("id", "unknown")
            answer = qa.get("answer", "")

            count = answer.lower().count(keyword.lower())

            if count > 0:
                # 키워드가 포함된 문장 추출
                sentences = self._split_sentences(answer)
                context_sentences = [
                    s for s in sentences
                    if keyword.lower() in s.lower()
                ]

                appearances.append({
                    "qa_id": qa_id,
                    "count": count,
                    "context": context_sentences[0][:150] if context_sentences else ""
                })

                if first_mention is None:
                    first_mention = qa_id
                last_mention = qa_id

        return {
            "keyword": keyword,
            "total_count": sum(a["count"] for a in appearances),
            "appearances": appearances,
            "first_mention": first_mention,
            "last_mention": last_mention
        }


    # ==================== Helper Methods ====================

    def _split_sentences(self, text: str) -> List[str]:
        """문장 단위로 분리"""
        sentences = re.split(r'[.!?]\s+', text)
        return [s.strip() for s in sentences if s.strip()]


    def get_competency_keyword_overlap(
        self,
        keyword_map: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        역량 간 키워드 중복 분석

        Returns:
            {
                "overlapping_keywords": {
                    ("data_driven", "problem_solving"): ["분석", "판단"],
                    ...
                },
                "unique_keywords": {
                    "data_driven": ["Python", "SQL"],
                    ...
                }
            }
        """
        all_keywords = {}
        for comp, keywords in keyword_map.items():
            all_keywords[comp] = set(k["keyword"] for k in keywords)

        # 중복 찾기
        overlapping = {}
        competencies = list(all_keywords.keys())

        for i, comp1 in enumerate(competencies):
            for comp2 in competencies[i+1:]:
                overlap = all_keywords[comp1] & all_keywords[comp2]
                if overlap:
                    overlapping[(comp1, comp2)] = list(overlap)

        # 고유 키워드 찾기
        unique = {}
        for comp, keywords in all_keywords.items():
            others = set()
            for other_comp, other_keywords in all_keywords.items():
                if other_comp != comp:
                    others.update(other_keywords)

            unique[comp] = list(keywords - others)

        return {
            "overlapping_keywords": overlapping,
            "unique_keywords": unique
        }
