"""
페르소나 생성기
CompanyProfile을 기반으로 면접관 페르소나를 생성합니다.
"""

import json
import boto3
from typing import List
from models.company_profile import CompanyProfile
from models.persona import Persona, ArchetypeEnum
from core.config import settings


class PersonaGenerator:
    """면접관 페르소나 생성"""

    # 아키타입별 기본 스타일 가이드
    ARCHETYPE_STYLES = {
        ArchetypeEnum.ANALYTICAL: {
            "name": "분석형",
            "description": "논리적이고 분석적인 면접관. 모든 답변에 대해 '왜'와 '어떻게'를 깊게 파고드는 꼬리 질문을 던집니다.",
            "base_prompt": """당신은 논리적이고 분석적인 면접관입니다.
지원자의 답변에 대해 '왜(Why)'와 '어떻게(How)'를 5단계까지 깊게 파고들며 꼬리 질문을 던지세요.
감정적인 공감보다 사실과 데이터 검증에 집중합니다.
답변의 논리적 일관성과 구체적인 근거를 중요하게 생각합니다."""
        },
        ArchetypeEnum.SUPPORTIVE: {
            "name": "친화형",
            "description": "따뜻하고 지지적인 면접관. 지원자가 편안하게 자신의 경험을 이야기할 수 있도록 격려합니다.",
            "base_prompt": """당신은 따뜻하고 지지적인 면접관입니다.
지원자가 편안하게 자신의 경험을 모두 이야기할 수 있도록 긍정적인 피드백과 격려를 사용하세요.
지원자의 잠재력과 성장 과정을 중요하게 생각합니다.
경청하며 공감하는 태도로 질문합니다."""
        },
        ArchetypeEnum.STRESS_TESTER: {
            "name": "압박형",
            "description": "빠르고 도전적인 면접관. 예상치 못한 상황을 가정하여 문제 해결 능력을 테스트합니다.",
            "base_prompt": """당신은 빠르고 도전적인 면접관입니다.
일부러 예상치 못한 상황을 가정하고(예: '지금 서비스가 다운된다면?'),
지원자의 문제 해결 능력과 압박감 속에서의 대처 방식을 테스트합니다.
날카로운 질문으로 지원자의 한계를 파악하려 합니다."""
        }
    }

    def __init__(self):
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=settings.AWS_REGION)
        self.model_id = settings.BEDROCK_MODEL_ID

    def generate_system_prompt(
        self,
        company_profile: CompanyProfile,
        archetype: ArchetypeEnum
    ) -> str:
        """
        Bedrock을 사용하여 페르소나의 System Prompt 생성

        Args:
            company_profile: 기업 프로필
            archetype: 면접관 아키타입

        Returns:
            생성된 시스템 프롬프트
        """
        style_info = self.ARCHETYPE_STYLES[archetype]
        base_prompt = style_info["base_prompt"]

        # LLM에게 페르소나 프롬프트 생성 요청
        prompt = f"""
당신은 AI 면접관의 '페르소나'를 생성하는 시스템 프롬프트 엔지니어입니다.

[회사 정보]:
- 회사명: {company_profile.company_name}
- 채용 직무: {company_profile.job_title}
- 필요 역량: {', '.join(company_profile.key_skills)}
- 기업 문화/인재상: {company_profile.culture_summary}

[면접관 아키타입]:
{base_prompt}

위 정보를 조합하여, 이 AI 면접관이 사용할 최종 '시스템 프롬프트(System Prompt)'를 200자 내외로 생성해 주세요.
이 프롬프트는 면접의 '톤 앤 매너'와 '핵심 질문 방향'을 결정해야 합니다.

시스템 프롬프트만 작성하고, 다른 설명은 붙이지 마세요.
"""

        try:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 512,
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
            system_prompt = response_body['content'][0]['text'].strip()

            return system_prompt

        except Exception as e:
            print(f"System Prompt 생성 에러: {e}")
            # 기본 프롬프트 반환
            return f"{base_prompt}\n\n당신은 {company_profile.company_name}의 면접관입니다. {', '.join(company_profile.key_skills)} 역량을 중심으로 질문하세요."

    def generate_welcome_message(
        self,
        company_profile: CompanyProfile,
        archetype: ArchetypeEnum
    ) -> str:
        """
        첫인사 메시지 생성

        Args:
            company_profile: 기업 프로필
            archetype: 면접관 아키타입

        Returns:
            환영 메시지
        """
        style_info = self.ARCHETYPE_STYLES[archetype]

        # 간단한 템플릿 기반 생성
        templates = {
            ArchetypeEnum.ANALYTICAL: f"안녕하세요. {company_profile.company_name}의 {company_profile.job_title} 면접을 진행하겠습니다. 답변은 구체적이고 논리적으로 부탁드립니다.",
            ArchetypeEnum.SUPPORTIVE: f"반갑습니다! {company_profile.company_name} 면접에 오신 것을 환영합니다. 편안하게 경험을 나눠주세요.",
            ArchetypeEnum.STRESS_TESTER: f"{company_profile.company_name} 면접을 시작하겠습니다. 빠른 판단력을 확인하고 싶습니다. 준비되셨나요?"
        }

        return templates.get(archetype, f"안녕하세요. {company_profile.company_name} 면접을 시작하겠습니다.")

    def create_persona(
        self,
        company_profile: CompanyProfile,
        archetype: ArchetypeEnum,
        persona_id: str
    ) -> Persona:
        """
        단일 페르소나 생성

        Args:
            company_profile: 기업 프로필
            archetype: 면접관 아키타입
            persona_id: 페르소나 ID

        Returns:
            Persona 객체
        """
        print(f"\n페르소나 생성 중: {company_profile.company_name} ({archetype.value})")

        # System Prompt 생성
        system_prompt = self.generate_system_prompt(company_profile, archetype)

        # 환영 메시지 생성
        welcome_message = self.generate_welcome_message(company_profile, archetype)

        # 스타일 설명
        style_description = self.ARCHETYPE_STYLES[archetype]["description"]

        # Persona 객체 생성
        persona = Persona(
            persona_id=persona_id,
            company_id=company_profile.company_id,
            company_name=company_profile.company_name,
            archetype=archetype,
            system_prompt=system_prompt,
            welcome_message=welcome_message,
            style_description=style_description,
            focus_keywords=company_profile.key_skills
        )

        print(f"✓ 페르소나 생성 완료: {persona.display_name}")
        return persona

    def create_personas_from_profiles(
        self,
        company_profiles: List[CompanyProfile]
    ) -> List[Persona]:
        """
        여러 CompanyProfile로부터 페르소나 생성
        각 기업마다 하나의 아키타입을 할당합니다.

        Args:
            company_profiles: 기업 프로필 리스트

        Returns:
            Persona 리스트
        """
        personas = []

        # 3개 기업에 3개 아키타입 할당
        archetypes = [
            ArchetypeEnum.ANALYTICAL,
            ArchetypeEnum.SUPPORTIVE,
            ArchetypeEnum.STRESS_TESTER
        ]

        for idx, profile in enumerate(company_profiles[:3]):  # 최대 3개만
            archetype = archetypes[idx % 3]
            persona_id = f"interviewer_{idx + 1}"

            persona = self.create_persona(profile, archetype, persona_id)
            personas.append(persona)

        print(f"\n총 {len(personas)}개 페르소나 생성 완료")
        return personas
