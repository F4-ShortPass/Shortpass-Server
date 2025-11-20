"""
약점 기반 후속 질문 생성 서비스

요구사항:
- 약점으로 파악된 역량에 대한 심층 면접 질문 자동 생성
- 직무 설명(JD)과 연계하여 실무 중심 질문 생성
- LLM 기반 질문 생성 (선택적)
"""

from typing import Dict, List, Any, Optional
from ai.utils.llm_client import LLMClient


class FollowUpQuestionGenerator:
    """
    약점 기반 후속 질문 생성기

    사용 시나리오:
    1. 평가 완료 후 약점 역량 파악
    2. 해당 역량에 대한 추가 검증 질문 생성
    3. HR이 2차 면접 또는 전화 인터뷰에서 활용
    """

    # 역량별 템플릿 질문 (LLM 없이도 사용 가능)
    TEMPLATE_QUESTIONS = {
        "strategic_thinking": [
            "우리 회사가 향후 3년 내 가장 주목해야 할 시장 변화는 무엇이라고 생각하나요? 그 이유는?",
            "경쟁사 대비 우리의 차별화 포인트를 어떻게 강화할 수 있을까요?",
            "전략 수립 시 가장 중요하게 고려해야 할 요소 3가지는 무엇인가요?"
        ],
        "data_driven": [
            "데이터가 부족한 상황에서 의사결정을 내려야 한다면 어떻게 접근하시겠습니까?",
            "과거 데이터 분석 프로젝트에서 가장 큰 인사이트를 얻은 경험을 구체적으로 말씀해주세요.",
            "SQL, Python 등 데이터 분석 도구를 실무에서 어떻게 활용한 경험이 있나요?"
        ],
        "communication": [
            "이해관계가 상충하는 부서 간 협업 상황에서 합의를 이끌어낸 경험이 있나요?",
            "상급자와 의견이 다를 때 어떻게 소통하시나요? 구체적 사례를 들어주세요.",
            "복잡한 기술 내용을 비전문가에게 설명해야 했던 경험과 그 방법을 말씀해주세요."
        ],
        "problem_solving": [
            "기존 방식으로 해결 불가능한 문제를 어떻게 돌파했는지 사례를 들어주세요.",
            "예상치 못한 위기 상황에서 빠르게 대응했던 경험을 구체적으로 말씀해주세요.",
            "창의적인 문제 해결 방법을 적용했던 사례와 그 결과를 공유해주세요."
        ],
        "industry_knowledge": [
            "우리 산업의 최근 3년간 가장 큰 변화는 무엇이라고 생각하나요?",
            "주요 경쟁사와 비교했을 때 우리 회사의 강점과 약점은 무엇인가요?",
            "향후 5년간 이 산업이 어떻게 변화할 것으로 예상하시나요?"
        ],
        "learning_attitude": [
            "최근 6개월 내 새롭게 학습한 것과 그것을 어떻게 적용했는지 말씀해주세요.",
            "업무에 필요한 새로운 기술이나 지식을 어떻게 습득하시나요?",
            "실패했던 경험에서 무엇을 배웠고 어떻게 개선했나요?"
        ]
    }


    def __init__(self, use_llm: bool = False):
        """
        Args:
            use_llm: True면 LLM 사용, False면 템플릿 질문 사용
        """
        self.use_llm = use_llm
        if use_llm:
            self.llm_client = LLMClient()


    def generate_follow_up_questions(
        self,
        weaknesses: List[Dict[str, Any]],
        job_description: Optional[str] = None,
        transcript: Optional[List[Dict[str, Any]]] = None,
        max_questions: int = 5
    ) -> List[Dict[str, str]]:
        """
        약점 기반 후속 질문 생성

        Args:
            weaknesses: [{"competency": "data_driven", "score": 55, "summary": "..."}]
            job_description: 직무 설명 (LLM 사용 시)
            transcript: 면접 transcript (LLM 사용 시)
            max_questions: 최대 질문 수

        Returns:
            [
                {
                    "question": "데이터가 부족한 상황에서...",
                    "reason": "데이터 기반 의사결정 역량이 55점으로 부족하여 검증 필요",
                    "related_weakness": "data_driven",
                    "difficulty": "medium"
                }
            ]
        """
        if self.use_llm and job_description and transcript:
            return self._generate_with_llm(weaknesses, job_description, transcript, max_questions)
        else:
            return self._generate_with_template(weaknesses, max_questions)


    def _generate_with_template(
        self,
        weaknesses: List[Dict[str, Any]],
        max_questions: int
    ) -> List[Dict[str, str]]:
        """템플릿 기반 질문 생성 (빠르고 안정적)"""

        questions = []

        competency_names = {
            "strategic_thinking": "전략적 사고",
            "data_driven": "데이터 기반 의사결정",
            "communication": "커뮤니케이션",
            "problem_solving": "문제해결력",
            "industry_knowledge": "산업 이해도",
            "learning_attitude": "학습 태도"
        }

        for weakness in weaknesses[:max_questions]:
            competency = weakness.get("competency", "")
            score = weakness.get("score", 0)
            summary = weakness.get("summary", "")

            comp_name = competency_names.get(competency, competency)

            # 템플릿 질문 가져오기
            template_questions = self.TEMPLATE_QUESTIONS.get(competency, [])

            if template_questions:
                # 점수에 따라 난이도 조절
                if score < 50:
                    selected_question = template_questions[0]  # 기본 질문
                    difficulty = "basic"
                elif score < 65:
                    selected_question = template_questions[1] if len(template_questions) > 1 else template_questions[0]
                    difficulty = "medium"
                else:
                    selected_question = template_questions[2] if len(template_questions) > 2 else template_questions[-1]
                    difficulty = "advanced"

                questions.append({
                    "question": selected_question,
                    "reason": f"{comp_name} 역량이 {score}점으로 평가되어 추가 검증이 필요합니다. {summary}",
                    "related_weakness": competency,
                    "difficulty": difficulty
                })

        return questions


    async def _generate_with_llm(
        self,
        weaknesses: List[Dict[str, Any]],
        job_description: str,
        transcript: List[Dict[str, Any]],
        max_questions: int
    ) -> List[Dict[str, str]]:
        """LLM 기반 맞춤형 질문 생성"""

        questions = []

        for weakness in weaknesses[:max_questions]:
            competency = weakness.get("competency", "")
            score = weakness.get("score", 0)
            summary = weakness.get("summary", "")

            # LLM 프롬프트 구성
            prompt = self._build_llm_prompt(
                competency=competency,
                score=score,
                summary=summary,
                job_description=job_description,
                transcript=transcript
            )

            try:
                # LLM 호출
                response = await self.llm_client.ainvoke(prompt)

                # 응답 파싱
                question_data = self._parse_llm_response(response, competency, score, summary)
                questions.append(question_data)

            except Exception as e:
                print(f"LLM 질문 생성 실패: {e}, 템플릿 사용")
                # LLM 실패 시 템플릿으로 fallback
                fallback = self._generate_with_template([weakness], 1)
                if fallback:
                    questions.append(fallback[0])

        return questions


    def _build_llm_prompt(
        self,
        competency: str,
        score: int,
        summary: str,
        job_description: str,
        transcript: List[Dict[str, Any]]
    ) -> str:
        """LLM용 프롬프트 구성"""

        competency_names = {
            "strategic_thinking": "전략적 사고",
            "data_driven": "데이터 기반 의사결정",
            "communication": "커뮤니케이션",
            "problem_solving": "문제해결력",
            "industry_knowledge": "산업 이해도",
            "learning_attitude": "학습 태도"
        }

        comp_name = competency_names.get(competency, competency)

        # Transcript 요약 (최근 3개 QA)
        recent_qa = transcript[-3:] if len(transcript) >= 3 else transcript
        transcript_summary = "\n".join([
            f"Q: {qa.get('question_text', '')[:80]}...\nA: {qa.get('answer_text', '')[:100]}..."
            for qa in recent_qa
        ])

        prompt = f"""
당신은 HR 면접 전문가입니다.
지원자의 약점을 보완하거나 추가 검증하기 위한 심층 면접 질문을 생성하세요.

<약점 정보>
- 역량: {comp_name}
- 점수: {score}점 (100점 만점)
- 평가 요약: {summary}

<직무 설명>
{job_description[:500]}

<최근 답변 내용>
{transcript_summary}

<요구사항>
1. 약점을 구체적으로 검증할 수 있는 실무 중심 질문 1개 생성
2. 질문은 구체적이고 상황 기반이어야 함
3. "경험이 있나요?" 같은 단순 질문 지양
4. 실제 업무 시나리오를 제시하는 질문 선호

<응답 형식>
질문: [생성된 질문]
이유: [왜 이 질문이 필요한지 1-2문장]
난이도: [basic/medium/advanced]
"""

        return prompt.strip()


    def _parse_llm_response(
        self,
        response: str,
        competency: str,
        score: int,
        summary: str
    ) -> Dict[str, str]:
        """LLM 응답 파싱"""

        import re

        # 질문 추출
        question_match = re.search(r'질문:\s*(.+?)(?=\n이유:|$)', response, re.DOTALL)
        question = question_match.group(1).strip() if question_match else response

        # 이유 추출
        reason_match = re.search(r'이유:\s*(.+?)(?=\n난이도:|$)', response, re.DOTALL)
        reason = reason_match.group(1).strip() if reason_match else summary

        # 난이도 추출
        difficulty_match = re.search(r'난이도:\s*(basic|medium|advanced)', response, re.IGNORECASE)
        difficulty = difficulty_match.group(1).lower() if difficulty_match else "medium"

        return {
            "question": question,
            "reason": reason,
            "related_weakness": competency,
            "difficulty": difficulty
        }


    def get_question_categories(self) -> List[str]:
        """지원 가능한 질문 카테고리 반환"""
        return list(self.TEMPLATE_QUESTIONS.keys())


    def get_sample_questions_by_competency(
        self,
        competency: str
    ) -> List[str]:
        """특정 역량의 샘플 질문 반환"""
        return self.TEMPLATE_QUESTIONS.get(competency, [])
