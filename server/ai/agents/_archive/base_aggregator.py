"""
Base Aggregator
역량 통합 기본 로직
"""

from typing import Dict, List
from datetime import datetime


class BaseAggregator:
    """역량 통합 기본 클래스"""
    
    @staticmethod
    def calculate_weighted_average(
        competency_results: Dict[str, Dict],
        weights: Dict[str, float]
    ) -> float:
        """
        가중 평균 계산
        
        Args:
            competency_results: 역량별 평가 결과
                {
                    "problem_solving": {"overall_score": 85, ...},
                    "org_fit": {"overall_score": 78, ...},
                    ...
                }
            weights: 역량별 가중치 (합이 1.0)
                {
                    "problem_solving": 0.25,
                    "org_fit": 0.20,
                    ...
                }
        
        Returns:
            가중 평균 점수 (0-100)
        """
        total_score = 0.0
        total_weight = 0.0
        
        for competency_name, result in competency_results.items():
            score = result.get("overall_score", 0)
            weight = weights.get(competency_name, 0)
            
            total_score += score * weight
            total_weight += weight
        
        # 가중치 합이 1이 아닐 경우 보정
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.0
    
    @staticmethod
    def extract_low_confidence_competencies(
        competency_results: Dict[str, Dict],
        threshold: float = 0.7
    ) -> List[str]:
        """
        낮은 신뢰도 역량 추출
        
        Args:
            competency_results: 역량별 평가 결과
            threshold: 신뢰도 임계값 (기본 0.7)
        
        Returns:
            신뢰도가 낮은 역량명 목록
        """
        low_confidence = []
        
        for competency_name, result in competency_results.items():
            confidence = result.get("confidence", {}).get("overall_confidence", 1.0)
            
            if confidence < threshold:
                low_confidence.append(competency_name)
        
        return low_confidence
    
    @staticmethod
    def aggregate_strengths(
        competency_results: Dict[str, Dict],
        top_n: int = 5
    ) -> List[str]:
        """
        전체 강점 통합
        
        Args:
            competency_results: 역량별 평가 결과
            top_n: 추출할 강점 개수
        
        Returns:
            통합 강점 목록
        """
        all_strengths = []
        
        for result in competency_results.values():
            strengths = result.get("strengths", [])
            all_strengths.extend(strengths)
        
        # 중복 제거 후 상위 N개 반환
        unique_strengths = list(dict.fromkeys(all_strengths))
        return unique_strengths[:top_n]
    
    @staticmethod
    def aggregate_weaknesses(
        competency_results: Dict[str, Dict],
        top_n: int = 3
    ) -> List[str]:
        """
        전체 약점 통합
        
        Args:
            competency_results: 역량별 평가 결과
            top_n: 추출할 약점 개수
        
        Returns:
            통합 약점 목록
        """
        all_weaknesses = []
        
        for result in competency_results.values():
            weaknesses = result.get("weaknesses", [])
            all_weaknesses.extend(weaknesses)
        
        # 중복 제거 후 상위 N개 반환
        unique_weaknesses = list(dict.fromkeys(all_weaknesses))
        return unique_weaknesses[:top_n]