# server/services/competency_service.py
"""
역량 분류 및 분석 서비스 (전처리 데이터 기반)
"""
import json
from typing import Dict, List, Optional, Any


class CompetencyService:
    """
    전처리된 JD 페르소나 데이터로부터 역량을 조회하는 서비스
    (Bedrock 호출 제거 → 메모리 캐시 기반)
    """

    # 고정 공통 역량 (하드코딩)
    COMMON_COMPETENCIES = [
        "고객지향",
        "도전정신",
        "협동",
        "팀워크",
        "목표지향",
        "책임감"
    ]

    def __init__(self):
        pass  # LLMClient 제거 (더 이상 Bedrock 호출 안 함)

    async def analyze_jd_competencies(self, jd_text: str = None) -> Dict[str, Any]:
        """
        전처리된 JD 역량 데이터를 반환 (Bedrock 호출 제거)

        Args:
            jd_text: (사용하지 않음, 하위 호환성 유지를 위해 남겨둠)

        Returns:
            Dict: 공통 역량과 직무 역량 정보
        """
        try:
            # main.py의 PERSONA_DATA_CACHE에서 데이터 조회
            from main import PERSONA_DATA_CACHE

            if PERSONA_DATA_CACHE is None:
                print("⚠️  경고: 페르소나 데이터가 로드되지 않았습니다. 기본값 반환.")
                return self._get_default_competencies()

            # 캐시된 데이터 사용
            return {
                "common_competencies": PERSONA_DATA_CACHE.get("common_competencies", self.COMMON_COMPETENCIES),
                "job_competencies": PERSONA_DATA_CACHE.get("job_competencies", []),
                "core_competencies": PERSONA_DATA_CACHE.get("core_competencies", []),
                "analysis_summary": f"{PERSONA_DATA_CACHE.get('company_name')} - {PERSONA_DATA_CACHE.get('job_title')} 직무의 역량 분석 결과 (사전 처리됨)"
            }

        except Exception as e:
            print(f"Error loading cached competencies: {e}")
            # 기본값 반환
            return self._get_default_competencies()

    def _get_default_competencies(self) -> Dict[str, Any]:
        """기본 역량 반환 (폴백)"""
        return {
            "common_competencies": self.COMMON_COMPETENCIES,
            "job_competencies": [
                "문제해결력", "커뮤니케이션", "창의적 사고",
                "기술적 이해", "리더십", "분석적 사고"
            ],
            "core_competencies": [
                "글로벌 시장 분석", "프로젝트 관리", "협상력",
                "공급망 이해", "리스크 관리"
            ],
            "analysis_summary": "페르소나 데이터가 없어 기본 역량으로 설정되었습니다."
        }

    def _build_competency_analysis_prompt(self, jd_text: str) -> str:
        """
        역량 분석용 프롬프트 생성
        """
        return f"""
다음 채용공고(JD)를 분석하여 해당 직무에 필요한 핵심 역량 6개를 추출해주세요.

<채용공고>
{jd_text}
</채용공고>

요구사항:
1. 직무 관련 핵심 역량 6개를 정확히 추출하세요
2. 각 역량은 2-4글자의 명사형으로 표현하세요
3. 기술적 역량, 소프트 스킬, 업무 역량을 균형있게 포함하세요

응답 형식 (JSON):
{{
  "job_competencies": [
    "데이터분석",
    "문제해결력",
    "커뮤니케이션",
    "창의적 사고",
    "기술적 이해",
    "리더십"
  ],
  "analysis_summary": "이 직무는 데이터 기반 의사결정과 팀 협업이 중요한 역할입니다..."
}}

반드시 JSON 형식으로만 응답해주세요.
"""

    def _parse_competency_response(self, response: str) -> Dict[str, Any]:
        """
        LLM 응답을 파싱하여 구조화된 데이터로 변환
        """
        try:
            # JSON 추출 시도
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)

                # job_competencies 검증
                if "job_competencies" in result and len(result["job_competencies"]) >= 6:
                    # 정확히 6개만 사용
                    result["job_competencies"] = result["job_competencies"][:6]
                    return result

            # 파싱 실패 시 기본값
            raise ValueError("Invalid response format")

        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            # 기본값 반환
            return {
                "job_competencies": [
                    "문제해결력", "커뮤니케이션", "창의적 사고",
                    "기술적 이해", "리더십", "분석적 사고"
                ],
                "analysis_summary": "응답 파싱 실패로 기본 역량이 설정되었습니다."
            }

    async def generate_persona_data(
        self,
        jd_text: str = None,
        job_competencies: List[str] = None,
        company_questions: List[str] = None
    ) -> Dict[str, Any]:
        """
        페르소나 데이터 조회 (전처리된 데이터 사용, Bedrock 호출 제거)

        Args:
            jd_text: (사용하지 않음, 하위 호환성 유지)
            job_competencies: (사용하지 않음)
            company_questions: 기업에서 입력한 필수 질문들

        Returns:
            Dict: 페르소나 정보가 포함된 완전한 JSON
        """
        try:
            # main.py의 PERSONA_DATA_CACHE에서 데이터 조회
            from main import PERSONA_DATA_CACHE

            if PERSONA_DATA_CACHE is None:
                print("⚠️  경고: 페르소나 데이터가 로드되지 않았습니다. 기본값 반환.")
                return self._get_default_persona_data(job_competencies or [], company_questions or [])

            # 캐시된 데이터 사용
            result = {
                "company": PERSONA_DATA_CACHE.get("company_name", "Unknown Company"),
                "common_competencies": PERSONA_DATA_CACHE.get("common_competencies", self.COMMON_COMPETENCIES),
                "job_competencies": PERSONA_DATA_CACHE.get("job_competencies", []),
                "core_competencies": PERSONA_DATA_CACHE.get("core_competencies", []),
                "persona_summary": PERSONA_DATA_CACHE.get("persona_summary", []),
                "system_prompt": PERSONA_DATA_CACHE.get("system_prompt", "")
            }

            # 기업 질문이 제공되면 추가
            if company_questions:
                result["core_questions"] = company_questions

            return result

        except Exception as e:
            print(f"Error loading persona data: {e}")
            # 기본 페르소나 반환
            return self._get_default_persona_data(job_competencies or [], company_questions or [])

    def _build_persona_generation_prompt(
        self,
        jd_text: str,
        job_competencies: List[str],
        company_questions: List[str]
    ) -> str:
        """
        페르소나 생성용 프롬프트
        """
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(company_questions)])
        competencies_text = ", ".join(job_competencies)

        return f"""
다음 정보를 바탕으로 면접관 페르소나 2개를 생성해주세요.

<채용공고>
{jd_text}
</채용공고>

<직무 역량>
{competencies_text}
</직무 역량>

<기업 필수 질문>
{questions_text}
</기업 필수 질문>

요구사항:
1. 서로 다른 평가 초점을 가진 2개의 면접관 페르소나 생성
2. 각 페르소나는 직무 역량 중 2-3개를 중점적으로 평가
3. 실제 면접에서 사용할 수 있는 구체적인 예시 질문 포함

응답 형식 (JSON):
{{
  "company": "회사명 추출",
  "persona_summary": [
    {{
      "type": "논리형 면접관",
      "focus": "문제해결력과 분석적 사고를 중점 평가",
      "target_competencies": ["문제해결력", "분석적 사고"],
      "example_question": "프로젝트에서 예상치 못한 문제를 어떻게 해결했나요?"
    }},
    {{
      "type": "커뮤니케이션형 면접관",
      "focus": "협업 및 소통 능력 평가",
      "target_competencies": ["커뮤니케이션", "리더십"],
      "example_question": "의견 충돌이 있었을 때, 어떻게 조율했나요?"
    }}
  ]
}}

반드시 JSON 형식으로만 응답해주세요.
"""

    def _parse_persona_response(self, response: str) -> Dict[str, Any]:
        """
        페르소나 생성 응답 파싱
        """
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)

                # 필수 필드 검증
                if ("company" in result and
                    "persona_summary" in result and
                    len(result["persona_summary"]) >= 2):
                    return result

            raise ValueError("Invalid persona response format")

        except Exception as e:
            print(f"Failed to parse persona response: {e}")
            return {}

    def _get_default_persona_data(
        self,
        job_competencies: List[str],
        company_questions: List[str]
    ) -> Dict[str, Any]:
        """
        기본 페르소나 데이터 반환
        """
        return {
            "company": "Unknown Company",
            "common_competencies": self.COMMON_COMPETENCIES,
            "job_competencies": job_competencies,
            "core_questions": company_questions,
            "persona_summary": [
                {
                    "type": "역량평가형 면접관",
                    "focus": "직무 역량과 경험을 중점 평가",
                    "target_competencies": job_competencies[:3],
                    "example_question": "해당 직무와 관련된 경험을 구체적으로 설명해주세요."
                },
                {
                    "type": "문화적합성형 면접관",
                    "focus": "조직 적합성과 소프트 스킬 평가",
                    "target_competencies": job_competencies[3:],
                    "example_question": "우리 조직에 어떻게 기여할 수 있을지 말씀해주세요."
                }
            ]
        }

    def get_competency_visualization_data(
        self,
        job_competencies: List[str]
    ) -> Dict[str, Any]:
        """
        육각형 그래프용 데이터 생성

        Args:
            job_competencies: 직무 역량 리스트

        Returns:
            Dict: 시각화용 데이터
        """
        return {
            "common_competencies": {
                "title": "공통 역량 (고정값)",
                "items": self.COMMON_COMPETENCIES,
                "color": "#3B82F6"  # 파란색
            },
            "job_competencies": {
                "title": "직무 역량 (JD 추출)",
                "items": job_competencies,
                "color": "#10B981"  # 녹색
            },
            "chart_config": {
                "type": "hexagon",
                "max_value": 5,
                "grid_lines": 5
            }
        }