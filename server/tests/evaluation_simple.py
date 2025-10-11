"""
평가 서비스 (LangGraph 통합)
Phase 1-4 전체 플로우 실행
"""

import os
import json
from datetime import datetime
from typing import Dict
from pathlib import Path
from dotenv import load_dotenv
from ai.agents.graph.evaluation import create_evaluation_graph
from openai import AsyncOpenAI

# .env 파일 로드
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

# 프롬프트 생성 함수 import
from ai.prompts.competency_agents.common_competencies.problem_solving_prompt import create_problem_solving_evaluation_prompt
from ai.prompts.competency_agents.common_competencies.organizational_fit_prompt import create_organizational_fit_evaluation_prompt
from ai.prompts.competency_agents.common_competencies.growth_potential_prompt import create_growth_potential_evaluation_prompt
from ai.prompts.competency_agents.common_competencies.interpersonal_skill_prompt import create_interpersonal_skill_evaluation_prompt
from ai.prompts.competency_agents.common_competencies.achievement_motivation_prompt import create_achievement_motivation_evaluation_prompt
from ai.prompts.competency_agents.job_competencies.structured_thinking_prompt import create_structured_thinking_evaluation_prompt
from ai.prompts.competency_agents.job_competencies.business_documentation_prompt import create_business_documentation_evaluation_prompt
from ai.prompts.competency_agents.job_competencies.financing_literacy_prompt import create_financial_literacy_evaluation_prompt
from ai.prompts.competency_agents.job_competencies.industry_learning_prompt import create_industry_learning_evaluation_prompt
from ai.prompts.competency_agents.job_competencies.stakeholder_coordination_prompt import create_stakeholder_management_evaluation_prompt


PROMPT_GENERATORS = {
    "problem_solving": create_problem_solving_evaluation_prompt,
    "organizational_fit": create_organizational_fit_evaluation_prompt,
    "growth_potential": create_growth_potential_evaluation_prompt,
    "interpersonal_skills": create_interpersonal_skill_evaluation_prompt,
    "achievement_motivation": create_achievement_motivation_evaluation_prompt,
    "structured_thinking": create_structured_thinking_evaluation_prompt,
    "business_documentation": create_business_documentation_evaluation_prompt,
    "financial_literacy": create_financial_literacy_evaluation_prompt,
    "industry_learning": create_industry_learning_evaluation_prompt,
    "stakeholder_management": create_stakeholder_management_evaluation_prompt,
}


class EvaluationService:
    """평가 서비스 (Phase 1-4 전체)"""
    
    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        
        self.openai_client = AsyncOpenAI(api_key=api_key)
        self.graph = create_evaluation_graph()
    
    def _load_prompts(self, transcript: Dict) -> Dict[str, str]:
        """프롬프트 로딩"""
        transcript_str = json.dumps(transcript, ensure_ascii=False, indent=2)
        return {
            name: generator(transcript_str)
            for name, generator in PROMPT_GENERATORS.items()
        }
    
    async def evaluate_interview(
        self,
        interview_id: int,
        applicant_id: int,
        job_id: int,
        transcript: Dict,
        job_weights: Dict[str, float],
        common_weights: Dict[str, float],
        job_common_ratio: Dict[str, float] = None
    ) -> Dict:
        """
        면접 평가 실행 (Phase 1-4)
        
        Args:
            interview_id: Interview ID
            applicant_id: Applicant ID
            job_id: Job ID
            transcript: 면접 transcript
            job_weights: Job 5개 역량 가중치
            common_weights: Common 5개 역량 가중치
            job_common_ratio: Job/Common 비율 (기본 {"job": 0.6, "common": 0.4})
        
        Returns:
            최종 평가 결과
        """
        
        # 기본값 설정
        if job_common_ratio is None:
            job_common_ratio = {"job": 0.6, "common": 0.4}
        
        # 프롬프트 로딩
        prompts = self._load_prompts(transcript)
        
        # 초기 상태 구성
        initial_state = {
            # 기본 정보
            "interview_id": interview_id,
            "applicant_id": applicant_id,
            "job_id": job_id,
            "transcript": transcript,
            "openai_client": self.openai_client,
            "prompts": prompts,
            
            # 가중치
            "job_weights": job_weights,
            "common_weights": common_weights,
            "job_common_ratio": job_common_ratio,
            
            # Phase 1 결과 (초기화)
            "structured_thinking_result": None,
            "business_documentation_result": None,
            "financial_literacy_result": None,
            "industry_learning_result": None,
            "stakeholder_management_result": None,
            "problem_solving_result": None,
            "organizational_fit_result": None,
            "growth_potential_result": None,
            "interpersonal_skills_result": None,
            "achievement_motivation_result": None,
            "job_aggregation_result": None,
            "common_aggregation_result": None,
            "low_confidence_competencies": [],
            "validation_notes": None,
            "requires_revaluation": False,
            
            # Phase 2 결과 (초기화)
            "evidence_conflicts": [],
            "low_confidence_list": [],
            "requires_collaboration": False,
            
            # Phase 3 결과 (초기화)
            "mediation_results": [],
            "adversarial_results": [],
            "collaboration_count": 0,
            
            # Phase 4 결과 (초기화)
            "final_score": None,
            "avg_confidence": None,
            "final_reliability": None,
            "reliability_note": None,
            "final_result": None,
            
            # 메타 정보
            "started_at": datetime.now(),
            "completed_at": None,
            "errors": [],
            "execution_logs": [],
            "structured_transcript": None
        }
        
        # 그래프 실행
        print("\n" + "="*80)
        print(f"평가 시작: Interview ID {interview_id}")
        print("="*80)
        
        result = await self.graph.ainvoke(initial_state)
        
        print("\n" + "="*80)
        print("평가 완료")
        print("="*80)
        
        # 최종 결과 구성
        return {
            # 기본 정보
            "interview_id": interview_id,
            "applicant_id": applicant_id,
            "job_id": job_id,
            
            # Phase 1 결과
            "job_aggregation": result["job_aggregation_result"],
            "common_aggregation": result["common_aggregation_result"],
            "validation": {
                "low_confidence_competencies": result["low_confidence_competencies"],
                "validation_notes": result["validation_notes"],
                "requires_revaluation": result["requires_revaluation"]
            },
            
            # Phase 2 결과
            "issues_detected": {
                "evidence_conflicts": result.get("evidence_conflicts", []),
                "low_confidence_list": result.get("low_confidence_list", []),
                "requires_collaboration": result.get("requires_collaboration", False)
            },
            
            # Phase 3 결과 (있다면)
            "collaboration": {
                "mediation_results": result.get("mediation_results", []),
                "adversarial_results": result.get("adversarial_results", []),
                "collaboration_count": result.get("collaboration_count", 0)
            },
            
            # Phase 4 최종 결과
            "final_result": result.get("final_result", {}),
            "final_score": result.get("final_score"),
            "avg_confidence": result.get("avg_confidence"),
            "final_reliability": result.get("final_reliability"),
            "reliability_note": result.get("reliability_note"),
            
            # 메타 정보
            "started_at": result["started_at"].isoformat(),
            "completed_at": result.get("completed_at", datetime.now()).isoformat(),
            "execution_logs": result.get("execution_logs", []),
            "errors": result.get("errors", [])
        }


# ============================================
# 테스트 코드
# ============================================

if __name__ == "__main__":
    import asyncio
    
    async def test_evaluation():
        # 테스트 데이터
        test_transcript = {
            "interview_id": 123,
            "metadata": {
                "applicant_id": 456,
                "job_id": 789,
                "interview_date": "2025-01-15"
            },
            "qa_pairs": [
                {
                    "id": "q1",
                    "question_text": "본인의 강점은 무엇인가요?",
                    "answer_text": "저는 데이터 분석 능력이 강점입니다..."
                },
                # ... 더 많은 질문-답변
            ]
        }
        
        test_job_weights = {
            "structured_thinking": 0.25,
            "business_documentation": 0.20,
            "financial_literacy": 0.20,
            "industry_learning": 0.20,
            "stakeholder_management": 0.15
        }
        
        test_common_weights = {
            "problem_solving": 0.25,
            "organizational_fit": 0.20,
            "growth_potential": 0.20,
            "interpersonal_skills": 0.20,
            "achievement_motivation": 0.15
        }
        
        # 서비스 생성
        service = EvaluationService()
        
        # 평가 실행
        result = await service.evaluate_interview(
            interview_id=123,
            applicant_id=456,
            job_id=789,
            transcript=test_transcript,
            job_weights=test_job_weights,
            common_weights=test_common_weights
        )
        
        # 결과 출력
        print("\n" + "="*80)
        print("최종 결과")
        print("="*80)
        print(f"최종 점수: {result['final_score']}점")
        print(f"신뢰도: {result['final_reliability']}")
        print(f"협업 처리: {result['collaboration']['collaboration_count']}회")
        print(f"Evidence 충돌: {len(result['issues_detected']['evidence_conflicts'])}건")
        print(f"Low Confidence: {len(result['issues_detected']['low_confidence_list'])}개")
    
    # 실행하려면 아래 주석 해제
    # asyncio.run(test_evaluation())
    print("\n테스트를 실행하려면 위 주석을 해제하세요.")