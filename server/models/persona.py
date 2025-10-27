"""
면접관 페르소나 데이터 모델
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class ArchetypeEnum(str, Enum):
    """면접관 아키타입"""
    ANALYTICAL = "analytical"      # 분석형
    SUPPORTIVE = "supportive"      # 친화형
    STRESS_TESTER = "stress_tester"  # 압박형


@dataclass
class Persona:
    """면접관 페르소나"""

    persona_id: str                    # 페르소나 ID (예: "interviewer_1")
    company_id: str                    # 소속 기업 ID
    company_name: str                  # 기업명
    archetype: ArchetypeEnum           # 아키타입
    system_prompt: str                 # LLM 시스템 프롬프트
    welcome_message: str               # 첫인사
    style_description: str             # 스타일 설명 (UI용)
    focus_keywords: List[str]          # 집중 키워드

    def __str__(self):
        return f"Persona({self.company_name} - {self.archetype.value})"

    def to_dict(self):
        """딕셔너리로 변환"""
        return {
            "persona_id": self.persona_id,
            "company_id": self.company_id,
            "company_name": self.company_name,
            "archetype": self.archetype.value,
            "system_prompt": self.system_prompt,
            "welcome_message": self.welcome_message,
            "style_description": self.style_description,
            "focus_keywords": self.focus_keywords
        }

    @property
    def display_name(self):
        """UI에 표시할 이름"""
        archetype_names = {
            ArchetypeEnum.ANALYTICAL: "분석형",
            ArchetypeEnum.SUPPORTIVE: "친화형",
            ArchetypeEnum.STRESS_TESTER: "압박형"
        }
        return f"면접관 ({self.company_name} - {archetype_names[self.archetype]})"
