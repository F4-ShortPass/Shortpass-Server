# ai/agents/rag_agent.py
"""
RAG Agent - JD 파싱 및 역량 정보 검색
- JD에서 필수/우대 기술 추출
- 도메인 요구사항 추출
- 동적 평가 기준 생성 (최대 5개)
- 역량별 가중치 계산
"""

import logging
import json
from typing import Dict, Any, List, Optional
from ai.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class RAGAgent:
    """
    RAG Agent for JD parsing and competency retrieval
    """

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def parse_jd(
        self,
        job_description: str,
        job_title: str
    ) -> Dict[str, Any]:
        """
        JD를 파싱하여 평가에 필요한 모든 정보 추출

        Args:
            job_description: 채용공고 전체 텍스트
            job_title: 직무명

        Returns:
            {
                "required_skills": ["Python", "AWS", ...],
                "preferred_skills": ["K8s", "GraphQL", ...],
                "domain_requirements": ["이커머스", "결제 시스템"],
                "dynamic_evaluation_criteria": [
                    "Python 숙련도",
                    "AWS 인프라 운영",
                    ...
                ],
                "competency_weights": {
                    "job_expertise": 0.40,
                    "analytical": 0.15,
                    ...
                },
                "position_type": "backend_senior",
                "seniority_level": "senior",
                "team_size": "5-10명",
                "main_responsibilities": ["API 개발", ...]
            }
        """
        prompt = self._build_jd_parsing_prompt(job_description, job_title)

        if not job_description or not job_description.strip():
            logger.error("Empty job description provided to RAGAgent")
            raise ValueError("Job description cannot be empty")

        logger.info(f"Starting JD parsing for job_title='{job_title}' (text_length={len(job_description)})")

        try:
            response = await self.llm_client.ainvoke(prompt)

            # JSON 파싱
            response_text = response.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            parsed_data = json.loads(response_text)
            logger.info(f"Successfully parsed JD for job_title='{job_title}'")

            # 검증 및 기본값 설정
            validated = self._validate_parsed_data(parsed_data)
            logger.debug(f"Validated data: {list(validated.keys())}")
            return validated

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed for job_title='{job_title}': {str(e)}", exc_info=True)
            logger.warning("Falling back to default parsed data")
            raise ValueError(f"LLM returned invalid JSON: {str(e)}")
        except Exception as e:
            logger.error(f"JD parsing failed for job_title='{job_title}': {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to parse JD: {str(e)}")

    def _build_jd_parsing_prompt(
        self,
        job_description: str,
        job_title: str
    ) -> str:
        """JD 파싱 프롬프트 생성"""
        return f"""당신은 채용공고를 분석하는 전문가입니다.
아래 채용공고에서 면접 평가에 필요한 정보를 추출하세요.

<채용공고>
직무명: {job_title}

{job_description}
</채용공고>

<추출_항목>
1. **필수 기술 (required_skills)**
   - 공고에서 "필수", "반드시", "required" 등으로 명시된 기술
   - 최대 10개
   - 예: ["Python", "FastAPI", "PostgreSQL", "AWS", "Docker"]

2. **우대 기술 (preferred_skills)**
   - 공고에서 "우대", "선호", "preferred", "plus" 등으로 명시된 기술
   - 최대 5개
   - 예: ["Kubernetes", "GraphQL", "Redis"]

3. **도메인 요구사항 (domain_requirements)**
   - 특정 산업/도메인 지식 요구사항
   - 예: ["이커머스", "결제 시스템", "핀테크"]
   - 없으면 빈 리스트

4. **동적 평가 기준 (dynamic_evaluation_criteria)**
   - 이 직무에서 가장 중요한 5개 평가 항목
   - 면접에서 깊이 있게 평가해야 할 역량
   - 반드시 5개 (더 많거나 적으면 안 됨)
   - 예: ["Python 숙련도", "AWS 인프라 운영", "컨테이너 오케스트레이션", "이커머스 도메인 지식", "실시간 문제해결 능력"]

5. **역량별 가중치 (competency_weights)**
   - 6개 역량의 중요도를 0-1 사이 값으로 (합계 1.0)
   - job_expertise: 직무 전문성 (기술 스택, 도메인 지식)
   - analytical: 분석적 사고
   - execution: 실행력
   - relationship: 관계 형성
   - resilience: 회복탄력성
   - influence: 영향력

   직무 유형에 따라 가중치 조정:
   - 시니어 기술직: job_expertise 높게 (0.35-0.45)
   - 주니어 기술직: job_expertise 중간, execution 높게
   - 리드/매니저: relationship, influence 높게
   - PM/기획: analytical 높게

6. **포지션 타입 (position_type)**
   - 직무 분류: backend, frontend, fullstack, devops, data, pm, designer 등

7. **시니어리티 레벨 (seniority_level)**
   - junior, mid, senior, lead, principal 중 하나

8. **주요 업무 (main_responsibilities)**
   - 핵심 업무 3-5개
   - 예: ["RESTful API 설계 및 개발", "데이터베이스 설계 및 최적화", "AWS 인프라 관리"]
</추출_항목>

<출력_형식>
오직 유효한 JSON만 반환하세요. 마크다운, 설명 없음.

{{
  "required_skills": ["Python", "FastAPI", "PostgreSQL", "AWS", "Docker", "REST API", "Git", "Linux"],
  "preferred_skills": ["Kubernetes", "GraphQL", "Redis", "Elasticsearch"],
  "domain_requirements": ["이커머스", "대용량 트래픽 처리"],
  "dynamic_evaluation_criteria": [
    "Python 숙련도",
    "AWS 인프라 운영",
    "컨테이너 오케스트레이션 (K8s)",
    "이커머스 도메인 지식",
    "실시간 문제해결 능력"
  ],
  "competency_weights": {{
    "job_expertise": 0.40,
    "analytical": 0.15,
    "execution": 0.20,
    "relationship": 0.10,
    "resilience": 0.05,
    "influence": 0.10
  }},
  "position_type": "backend",
  "seniority_level": "senior",
  "main_responsibilities": [
    "RESTful API 설계 및 개발",
    "데이터베이스 설계 및 최적화",
    "AWS 인프라 구축 및 운영",
    "코드 리뷰 및 기술 멘토링"
  ]
}}
</출력_형식>

<중요_규칙>
1. dynamic_evaluation_criteria는 반드시 정확히 5개
2. competency_weights의 합은 반드시 1.0
3. required_skills와 preferred_skills는 중복 없이
4. 공고에 없는 내용은 추측하지 말고 빈 리스트/기본값
5. 기술 이름은 정확하게 (예: "python" ✗ → "Python" ✓)
</중요_규칙>

위 형식에 맞춰 JSON만 반환하세요.
"""

    def _validate_parsed_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """파싱된 데이터 검증 및 기본값 설정"""

        # 필수 필드 확인
        validated = {
            "required_skills": data.get("required_skills", []),
            "preferred_skills": data.get("preferred_skills", []),
            "domain_requirements": data.get("domain_requirements", []),
            "dynamic_evaluation_criteria": data.get("dynamic_evaluation_criteria", []),
            "competency_weights": data.get("competency_weights", {}),
            "position_type": data.get("position_type", "unknown"),
            "seniority_level": data.get("seniority_level", "mid"),
            "main_responsibilities": data.get("main_responsibilities", [])
        }

        # dynamic_evaluation_criteria 검증 (정확히 5개)
        if len(validated["dynamic_evaluation_criteria"]) != 5:
            logger.warning(f"dynamic_evaluation_criteria has {len(validated['dynamic_evaluation_criteria'])} items, adjusting to 5")
            criteria = validated["dynamic_evaluation_criteria"]
            if len(criteria) < 5:
                # 부족하면 기본 항목 추가
                default_criteria = [
                    "기술 전문성",
                    "문제 해결 능력",
                    "커뮤니케이션 스킬",
                    "팀워크",
                    "성장 가능성"
                ]
                while len(criteria) < 5:
                    criteria.append(default_criteria[len(criteria)])
                logger.debug(f"Added {5 - len(validated['dynamic_evaluation_criteria'])} default criteria")
            else:
                # 초과하면 상위 5개만
                criteria = criteria[:5]
                logger.debug(f"Trimmed criteria to 5 items")
            validated["dynamic_evaluation_criteria"] = criteria

        # competency_weights 검증 (합계 1.0)
        weights = validated["competency_weights"]
        required_keys = ["job_expertise", "analytical", "execution", "relationship", "resilience", "influence"]

        # 누락된 키 확인
        if not all(key in weights for key in required_keys):
            missing_keys = [k for k in required_keys if k not in weights]
            logger.warning(f"competency_weights missing keys: {missing_keys}, using default weights")
            validated["competency_weights"] = self._get_default_weights()
        else:
            # 합계 검증
            total = sum(weights.values())
            if abs(total - 1.0) > 0.01:
                logger.warning(f"competency_weights sum is {total:.3f}, normalizing to 1.0")
                validated["competency_weights"] = {
                    key: val / total for key, val in weights.items()
                }
                logger.debug(f"Normalized weights: {validated['competency_weights']}")

        return validated

    def _get_default_weights(self) -> Dict[str, float]:
        """기본 가중치"""
        return {
            "job_expertise": 0.30,
            "analytical": 0.15,
            "execution": 0.20,
            "relationship": 0.15,
            "resilience": 0.10,
            "influence": 0.10
        }

    def _get_default_parsed_data(self) -> Dict[str, Any]:
        """파싱 실패 시 기본 데이터"""
        return {
            "required_skills": [],
            "preferred_skills": [],
            "domain_requirements": [],
            "dynamic_evaluation_criteria": [
                "기술 전문성",
                "문제 해결 능력",
                "커뮤니케이션 스킬",
                "팀워크",
                "성장 가능성"
            ],
            "competency_weights": self._get_default_weights(),
            "position_type": "unknown",
            "seniority_level": "mid",
            "main_responsibilities": []
        }

    async def get_competency_definition(self, competency_name: str) -> str:
        """
        역량 정의 검색 (향후 vector DB 연동)

        Args:
            competency_name: 역량 이름 (job_expertise, analytical, ...)

        Returns:
            역량 정의 텍스트
        """
        # 현재는 하드코딩, 향후 vector DB에서 검색
        definitions = {
            "job_expertise": """
직무 전문성 (Job Expertise)
- 해당 직무에 필요한 기술 스택 보유
- 도메인 지식과 실무 경험
- 직무별 문제 해결 능력
            """,
            "analytical": """
분석적 사고 (Analytical Thinking)
- 문제를 체계적으로 분해하고 분석
- 데이터 기반 의사결정
- 논리적 추론과 패턴 인식
            """,
            "execution": """
실행력 (Execution)
- 계획을 실제 결과로 전환
- 프로젝트 완수 능력
- 주인의식과 성과 책임감
            """,
            "relationship": """
관계 형성 (Relationship Building)
- 팀워크와 협업 능력
- 커뮤니케이션 스킬
- 갈등 해결과 네트워킹
            """,
            "resilience": """
회복탄력성 (Resilience)
- 스트레스 관리 능력
- 실패로부터 학습
- 변화 적응력
            """,
            "influence": """
영향력 (Influence)
- 리더십과 설득력
- 조직 내 영향력
- 기술 전파와 멘토링
            """
        }

        return definitions.get(competency_name, "역량 정의를 찾을 수 없습니다.")

    async def get_persona_info(self, persona_id: int, db) -> Dict[str, Any]:
        """
        페르소나 정보 검색

        Args:
            persona_id: 페르소나 ID
            db: DB 세션

        Returns:
            페르소나 정보
        """
        from models.interview import PersonaDB

        persona = db.query(PersonaDB).filter(PersonaDB.id == persona_id).first()

        if not persona:
            return {}

        return {
            "persona_name": persona.persona_name,
            "archetype": persona.archetype,
            "focus_areas": persona.focus_areas,
            "focus_keywords": persona.focus_keywords,
            "style_description": persona.style_description
        }
