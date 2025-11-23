"""
Common Aggregator
Common 5개 역량 통합
"""

from typing import Dict
from datetime import datetime
from .base_aggregator import BaseAggregator


class CommonAggregator(BaseAggregator):
    """Common 역량 통합"""
    
    @staticmethod
    def aggregate(
        competency_results: Dict[str, Dict],
        weights: Dict[str, float]
    ) -> Dict:
        """
        Common 5개 역량 통합
        
        Args:
            competency_results: Common 5개 역량 평가 결과
                {
                    "problem_solving": {...},
                    "organizational_fit": {...},
                    "growth_potential": {...},
                    "interpersonal_skills": {...},
                    "achievement_motivation": {...}
                }
            weights: Common 역량별 가중치 (JD 파싱 결과)
                {
                    "problem_solving": 0.25,
                    "organizational_fit": 0.20,
                    ...
                }
        
        Returns:
            CommonAggregationResult Dict
        """
        
        # 1. 가중 평균 계산
        overall_common_score = BaseAggregator.calculate_weighted_average(
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
            "problem_solving": competency_results.get("problem_solving"),
            "organizational_fit": competency_results.get("organizational_fit"),
            "growth_potential": competency_results.get("growth_potential"),
            "interpersonal_skills": competency_results.get("interpersonal_skills"),
            "achievement_motivation": competency_results.get("achievement_motivation"),
            "overall_common_score": round(overall_common_score, 2),
            "weights": weights,
            "aggregated_at": datetime.now().isoformat(),
            "low_confidence_competencies": low_confidence,
            "top_strengths": top_strengths,
            "top_weaknesses": top_weaknesses
        }
        
        return result