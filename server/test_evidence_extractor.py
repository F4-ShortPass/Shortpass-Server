"""
Evidence Extractor 테스트 스크립트

scripts.txt의 실제 면접 데이터를 사용하여
평가 근거 추출 기능을 테스트합니다.
"""

from services.evaluation.evidence_extractor import EvidenceExtractor


def test_박서연_interview():
    """
    박서연 지원자의 실제 면접 transcript를 기반으로
    평가 근거 추출 테스트
    """

    # scripts.txt에서 추출한 실제 QA
    qa_pairs = [
        {
            "question": "현대모비스가 글로벌 전기차 부품 시장에서 시장점유율을 확대하려고 합니다. 우리가 경쟁 우위를 확보하기 위해 어떤 전략적 접근을 해야 한다고 생각하시나요?",
            "answer": """
먼저 경쟁사 대비 우리의 강점과 약점을 분석하는 게 중요하다고 생각합니다.
현대모비스는 현대차그룹과의 긴밀한 협업 관계가 있어서 초기 검증이 빠르다는 장점이 있고요,
배터리 관리 시스템 같은 전기차 핵심 부품 기술력도 보유하고 있습니다.
반면 글로벌 브랜드 인지도나 다양한 완성차 업체와의 네트워크는 경쟁사 대비 약하다고 봅니다.
그래서 저는 두 가지 방향을 제안하고 싶은데요,
첫째는 기술 차별화입니다. 특히 차세대 배터리 기술이나 통합 열관리 시스템 같은 영역에서 독보적인 기술력을 확보하고,
둘째는 지역별 맞춤 전략입니다. 유럽은 규제가 강하니까 친환경 인증과 ESG 스토리를 강조하고,
중국 시장은 현지 파트너십을 통해 빠르게 점유율을 높이는 전략이 필요합니다.
            """.strip(),
            "target_competencies": ["strategic_thinking"]
        },
        {
            "question": "과거에 데이터를 활용해 비즈니스 문제를 해결했던 경험을 구체적으로 말씀해주시겠습니까?",
            "answer": """
이전 인턴십에서 마케팅 캠페인 효율성을 분석한 적이 있습니다.
당시 회사가 여러 채널에 광고비를 쓰고 있었는데 어떤 채널이 가장 효과적인지 명확하지 않았거든요.
저는 6개월치 광고 데이터를 수집해서 엑셀로 피벗 테이블을 만들고, 채널별 전환율과 ROI를 계산했습니다.
그 결과 인스타그램 광고가 전환율은 높지만 단가가 비싸고, 네이버 블로그는 전환율은 낮지만 비용 대비 효율이 좋다는 걸 발견했습니다.
이 인사이트를 바탕으로 예산을 재배분하는 제안서를 만들었고, 실제로 다음 분기 마케팅 전략에 반영되었습니다.
            """.strip(),
            "target_competencies": ["data_driven"]
        },
        {
            "question": "Python이나 R 같은 분석 도구는 사용해보신 적이 있나요?",
            "answer": """
Python은 학교 수업에서 기초 문법을 배웠고, Pandas 라이브러리를 사용해서 간단한 데이터 전처리는 해봤습니다.
하지만 실무 프로젝트에서 직접 활용한 경험은 많지 않습니다.
대신 SQL은 데이터베이스 수업에서 배워서 기본적인 쿼리 작성은 가능합니다.
            """.strip(),
            "target_competencies": ["data_driven"]
        },
        {
            "question": "만약 현대모비스에서 신규 시장 진입을 검토하는데 데이터가 부족하거나 불확실한 상황이라면 어떻게 접근하시겠습니까?",
            "answer": """
데이터가 부족하면 일단 시장조사 보고서나 업계 리포트를 찾아보고,
경쟁사 공시자료 같은 2차 자료를 최대한 활용할 것 같습니다.
그래도 부족하면 전문가 인터뷰를 진행하거나, 작은 규모로 파일럿 테스트를 해볼 수도 있을 것 같아요.
하지만 솔직히 실무에서 그런 상황을 직접 겪어본 적은 없어서 정확히 어떻게 해야 할지는 조금 막막합니다.
            """.strip(),
            "target_competencies": ["data_driven", "problem_solving"]
        },
        {
            "question": "과거에 다른 팀이나 이해관계자와 협업하면서 어려웠던 경험과 그것을 어떻게 해결했는지 말씀해주세요.",
            "answer": """
학교 팀 프로젝트에서 팀원들 간 의견 차이로 힘들었던 적이 있습니다.
제가 제안한 주제를 다른 팀원이 너무 어렵다고 반대했거든요.
처음에는 제 의견을 계속 설득하려고 했는데 분위기만 안 좋아지더라고요.
그래서 접근을 바꿔서, 각자가 제안한 주제의 장단점을 표로 정리해서 객관적으로 비교했습니다.
그랬더니 팀원들도 납득하고, 결국 두 아이디어를 절충한 새로운 방향을 찾을 수 있었습니다.
이 경험을 통해 감정보다는 논리와 데이터로 소통하는 게 중요하다는 걸 배웠습니다.
            """.strip(),
            "target_competencies": ["communication"]
        },
        {
            "question": "현재 자동차 산업, 특히 전기차 시장의 주요 트렌드를 어떻게 이해하고 계신가요?",
            "answer": """
제가 이해하기로는 전기차 시장이 빠르게 성장하고 있고, 배터리 기술이 핵심 경쟁력이라고 알고 있습니다.
특히 배터리 주행거리와 충전 속도가 중요한 이슈이고, 최근에는 자율주행 기술도 같이 발전하고 있다고 들었습니다.
또 중국 기업들이 가격 경쟁력으로 시장을 빠르게 장악하고 있는 것도 위협 요인인 것 같습니다.
            """.strip(),
            "target_competencies": ["industry_knowledge"]
        },
        {
            "question": "현대모비스 같은 부품 공급사 입장에서는 어떤 기회와 위협이 있다고 보시나요?",
            "answer": """
기회는 전기차로 전환되면서 새로운 부품 수요가 생긴다는 점일 것 같습니다.
배터리 관리 시스템이나 전력 변환 장치 같은 것들이요.
위협은... 솔직히 깊이 생각해본 적은 없는데,
아마도 기존 내연기관 부품 매출이 줄어드는 게 위협일 것 같고,
새로운 전기차 부품 시장에서 경쟁이 치열해지는 것도 위협 요소일 것 같습니다.
확실하지는 않지만요.
            """.strip(),
            "target_competencies": ["industry_knowledge"]
        },
        {
            "question": "만약 현대모비스에 입사하게 된다면 첫 1년 동안 어떤 목표를 세우고 어떻게 성장하고 싶으신가요?",
            "answer": """
첫 1년은 정말 빠르게 배우는 시간으로 만들고 싶습니다.
먼저 자동차 산업과 현대모비스의 사업 구조를 깊이 이해하는 게 우선이라고 생각합니다.
선배님들의 프로젝트를 보조하면서 실무 프로세스를 익히고,
데이터 분석 도구도 더 능숙하게 다룰 수 있도록 공부하고 싶습니다.
특히 Python이나 SQL 같은 도구를 실무에서 활용할 수 있는 수준까지 끌어올리고 싶어요.
그리고 가능하다면 작은 프로젝트라도 직접 리드해보면서 성과를 내고 싶습니다.
제가 부족한 부분이 많다는 걸 알고 있지만, 배우려는 자세와 열정만큼은 누구보다 강하다고 자신합니다.
            """.strip(),
            "target_competencies": ["learning_attitude"]
        }
    ]

    # 역량별 가정 점수 (실제로는 AI 평가 결과)
    competency_scores = {
        "strategic_thinking": 82,
        "data_driven": 75,
        "communication": 78,
        "problem_solving": 70,
        "industry_knowledge": 55,
        "learning_attitude": 85
    }

    # Evidence Extractor 초기화
    extractor = EvidenceExtractor()

    print("=" * 80)
    print("박서연 지원자 면접 평가 근거 추출 테스트")
    print("=" * 80)

    # 1. 전체 증거 추출
    print("\n[1] 전체 역량별 증거 추출\n")
    evidences = extractor.extract_all_evidences(qa_pairs, competency_scores)

    for evidence in evidences:
        print(f"\n--- {evidence['competency_name']} ({evidence['score']}점) ---")
        print(f"\n긍정 키워드: {', '.join(evidence['positive_keywords'])}")
        if evidence['negative_keywords']:
            print(f"부정 키워드: {', '.join(evidence['negative_keywords'])}")

        print(f"\n증거 문장 (Top 3):")
        for i, sent in enumerate(evidence['evidence_sentences'][:3], 1):
            print(f"  {i}. {sent[:100]}...")

        print(f"\n평가 근거:\n{evidence['justification']}")
        print("-" * 80)

    # 2. 강점/약점 추출
    print("\n[2] 강점/약점 분석\n")
    strengths_weaknesses = extractor.extract_strengths_weaknesses(
        competency_scores=competency_scores,
        evidences=evidences,
        top_n=3
    )

    print("강점 (Top 3):")
    for i, strength in enumerate(strengths_weaknesses['strengths'], 1):
        print(f"\n  {i}. {strength['competency_name']} ({strength['score']}점)")
        print(f"     요약: {strength['summary']}")
        print(f"     키워드: {', '.join(strength['keywords'])}")
        print(f"     증거: {strength['evidence'][:80]}...")

    print("\n\n약점 (개선 필요):")
    for i, weakness in enumerate(strengths_weaknesses['weaknesses'], 1):
        print(f"\n  {i}. {weakness['competency_name']} ({weakness['score']}점)")
        print(f"     요약: {weakness['summary']}")
        print(f"     개선 제안: {weakness['improvement_suggestion']}")
        print(f"     후속 질문: {weakness['follow_up_question']}")

    # 3. 특정 역량 상세 분석 (데이터 기반 의사결정)
    print("\n\n[3] 특정 역량 상세 분석: 데이터 기반 의사결정\n")

    # 데이터 기반 답변만 추출
    data_driven_answers = [
        qa["answer"] for qa in qa_pairs
        if "data_driven" in qa["target_competencies"]
    ]
    combined_answer = "\n".join(data_driven_answers)

    data_evidence = extractor.extract_evidence_for_competency(
        answer_text=combined_answer,
        competency="data_driven",
        score=competency_scores["data_driven"]
    )

    print(f"점수: {data_evidence.score}점")
    print(f"\n긍정 키워드: {data_evidence.positive_keywords}")
    print(f"부정 키워드: {data_evidence.negative_keywords}")

    print(f"\n하이라이트 범위 (키워드 위치):")
    for i, highlight in enumerate(data_evidence.highlight_ranges[:10], 1):
        print(f"  {i}. [{highlight.start}:{highlight.end}] '{highlight.text}' ({highlight.sentiment})")

    print(f"\n평가 근거:\n{data_evidence.justification}")

    print("\n" + "=" * 80)
    print("테스트 완료!")
    print("=" * 80)


if __name__ == "__main__":
    test_박서연_interview()
