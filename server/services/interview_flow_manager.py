# server/services/interview_flow_manager.py
"""
면접 플로우 매니저
공통 질문 → 3개 페르소나 분기 → 종료 플로우를 관리합니다.
"""

import json
import boto3
from typing import List, Dict, Optional
from enum import Enum
from models.persona import Persona
from core.config import settings


class InterviewStage(str, Enum):
    """면접 단계"""
    COMMON = "common"          # 공통 질문
    BRANCHED = "branched"      # 분기 (3명 동시)
    FINISHED = "finished"      # 종료


class InterviewFlowManager:
    """면접 플로우 관리"""

    # 공통 질문 템플릿
    COMMON_QUESTIONS = [
        "먼저 간단히 자기소개 부탁드립니다.",
        "지원 동기를 말씀해주세요.",
        "자신의 강점을 보여줄 수 있는 대표 경험을 하나 소개해주세요."
    ]

    def __init__(self, personas: List[Persona], applicant_name: str = "지원자"):
        """
        Args:
            personas: 3명의 면접관 페르소나
            applicant_name: 지원자 이름
        """
        self.personas = personas
        self.applicant_name = applicant_name
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=settings.AWS_REGION)
        self.model_id = settings.BEDROCK_MODEL_ID

        # 대화 히스토리 (각 페르소나별)
        self.conversation_histories = {
            persona.persona_id: [] for persona in personas
        }

        # 공통 대화 히스토리
        self.common_history = []

        # 현재 상태
        self.stage = InterviewStage.COMMON
        self.common_question_index = 0
        self.branch_question_count = {persona.persona_id: 0 for persona in personas}

    def get_next_common_question(self) -> Optional[str]:
        """
        다음 공통 질문 가져오기

        Returns:
            질문 문자열 또는 None (공통 질문 종료)
        """
        if self.common_question_index >= len(self.COMMON_QUESTIONS):
            return None

        question = self.COMMON_QUESTIONS[self.common_question_index]
        self.common_question_index += 1
        return question

    def add_common_qa(self, question: str, answer: str):
        """
        공통 질문/답변을 히스토리에 추가

        Args:
            question: 질문
            answer: 답변
        """
        self.common_history.append({
            "role": "assistant",
            "content": question
        })
        self.common_history.append({
            "role": "user",
            "content": answer
        })

    def generate_follow_up_question(
        self,
        persona: Persona,
        user_answer: str
    ) -> str:
        """
        페르소나 기반 꼬리 질문 생성

        Args:
            persona: 면접관 페르소나
            user_answer: 사용자 답변

        Returns:
            생성된 질문
        """
        # 해당 페르소나의 대화 히스토리
        history = self.conversation_histories[persona.persona_id]

        # 프롬프트 구성
        prompt_messages = [
            {
                "role": "user",
                "content": f"""{persona.system_prompt}

[공통 질문 단계에서 나눈 대화]:
{self._format_history(self.common_history)}

[당신({persona.display_name})과의 대화 이력]:
{self._format_history(history)}

[방금 지원자의 답변]:
{user_answer}

위 답변을 바탕으로, 당신의 페르소나({persona.archetype.value})에 맞는 **다음 꼬리 질문**을 하나만 생성해주세요.
질문만 생성하고 다른 설명은 절대 붙이지 마세요.
"""
            }
        ]

        try:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 256,
                "messages": prompt_messages,
                "temperature": 0.7
            })

            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json"
            )

            response_body = json.loads(response['body'].read().decode('utf-8'))
            question = response_body['content'][0]['text'].strip()

            return question

        except Exception as e:
            print(f"꼬리 질문 생성 에러: {e}")
            return f"{persona.focus_keywords[0]}에 대해 좀 더 구체적으로 설명해주세요."

    def _format_history(self, history: List[Dict]) -> str:
        """대화 히스토리를 문자열로 포맷"""
        if not history:
            return "(없음)"

        formatted = []
        for msg in history:
            role = "면접관" if msg["role"] == "assistant" else "지원자"
            formatted.append(f"{role}: {msg['content']}")

        return "\n".join(formatted)

    def process_branched_answer(
        self,
        persona: Persona,
        answer: str
    ) -> str:
        """
        분기 단계에서 답변 처리 및 다음 질문 생성

        Args:
            persona: 현재 질문한 페르소나
            answer: 지원자 답변

        Returns:
            다음 질문
        """
        # 히스토리에 추가
        self.conversation_histories[persona.persona_id].append({
            "role": "user",
            "content": answer
        })

        # 질문 카운트 증가
        self.branch_question_count[persona.persona_id] += 1

        # 다음 질문 생성
        next_question = self.generate_follow_up_question(persona, answer)

        # 히스토리에 질문 추가
        self.conversation_histories[persona.persona_id].append({
            "role": "assistant",
            "content": next_question
        })

        return next_question

    def should_finish_interview(self) -> bool:
        """
        면접 종료 여부 판단

        Returns:
            종료 여부
        """
        # 각 페르소나가 2-3개씩 질문했으면 종료
        min_questions = 2
        return all(
            count >= min_questions
            for count in self.branch_question_count.values()
        )

    def generate_final_comments(self) -> Dict[str, str]:
        """
        각 면접관별 최종 코멘트 생성

        Returns:
            {persona_id: comment} 딕셔너리
        """
        comments = {}

        for persona in self.personas:
            history = self.conversation_histories[persona.persona_id]

            if not history:
                comments[persona.persona_id] = f"{persona.display_name}: 감사합니다."
                continue

            # LLM으로 짧은 코멘트 생성
            prompt = f"""{persona.system_prompt}

당신과 지원자의 대화:
{self._format_history(history)}

위 대화를 바탕으로, 지원자에게 짧은 마무리 코멘트를 해주세요. (50자 이내)
긍정적이고 건설적인 피드백을 포함하세요.
"""

            try:
                body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 128,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7
                })

                response = self.bedrock_runtime.invoke_model(
                    modelId=self.model_id,
                    body=body,
                    contentType="application/json",
                    accept="application/json"
                )

                response_body = json.loads(response['body'].read().decode('utf-8'))
                comment = response_body['content'][0]['text'].strip()

                comments[persona.persona_id] = comment

            except Exception as e:
                print(f"최종 코멘트 생성 에러: {e}")
                comments[persona.persona_id] = "좋은 답변 감사합니다. 수고하셨습니다."

        return comments

    def start_branched_stage(self):
        """분기 단계 시작"""
        self.stage = InterviewStage.BRANCHED
        print("\n=== 분기 단계 시작 ===")
        print(f"{len(self.personas)}명의 면접관이 동시에 질문합니다.")

    def finish_interview(self):
        """면접 종료"""
        self.stage = InterviewStage.FINISHED
        print("\n=== 면접 종료 ===")
