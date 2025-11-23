
# server/ai/agents/aggregators/job_aggregator.py
"""
Job Aggregator
Job 5개 역량 통합
"""

from typing import Dict
from datetime import datetime
from .base_aggregator import BaseAggregator


class JobAggregator(BaseAggregator):
    """Job 역량 통합"""
    
    @staticmethod
    def aggregate(
        competency_results: Dict[str, Dict],
        weights: Dict[str, float]
    ) -> Dict:
        """
        Job 5개 역량 통합
        
        Args:
            competency_results: Job 5개 역량 평가 결과
                {
                    "structured_thinking": {...},
                    "business_documentation": {...},
                    "financial_literacy": {...},
                    "industry_learning": {...},
                    "stakeholder_management": {...}
                }
            weights: Job 역량별 가중치 (JD 파싱 결과)
                {
                    "structured_thinking": 0.25,
                    "business_documentation": 0.20,
                    ...
                }
        
        Returns:
            JobAggregationResult Dict
        """
        
        # 1. 가중 평균 계산
        overall_job_score = BaseAggregator.calculate_weighted_average(
            competency_results, 
            weights
        )
        
        # 2. 낮은 신뢰도 역량 추출
        low_confidence = BaseAggregator.extract_low_confidence_competencies(
            competency_results,
            threshold=0.7
        )
        
        # 3. 강점/약점 통합
        top_strengths = BaseAggregator.aggregate_strengths(competency_results, top_n=5)
        top_weaknesses = BaseAggregator.aggregate_weaknesses(competency_results, top_n=3)
        
        # 4. 결과 구성
        result = {
            "structured_thinking": competency_results.get("structured_thinking"),
            "business_documentation": competency_results.get("business_documentation"),
            "financial_literacy": competency_results.get("financial_literacy"),
            "industry_learning": competency_results.get("industry_learning"),
            "stakeholder_management": competency_results.get("stakeholder_management"),
            "overall_job_score": round(overall_job_score, 2),
            "weights": weights,
            "aggregated_at": datetime.now().isoformat(),
            "low_confidence_competencies": low_confidence,
            "top_strengths": top_strengths,
            "top_weaknesses": top_weaknesses
        }
        
        return result