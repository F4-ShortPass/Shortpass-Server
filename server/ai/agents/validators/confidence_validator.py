"""
Confidence Validator
신뢰도 검증 로직 (Phase 1에서 사용)

Phase 1에서는 간단한 검증만 수행:
- Low Confidence 역량 목록 추출 (역량명만)
- 전체 검증 노트 생성

Phase 2 ConfidenceDetector는 더 상세한 분석 수행
"""

from typing import Dict, List


class ConfidenceValidator:
    """신뢰도 검증"""
    
    @staticmethod
    def validate(
        job_aggregation: Dict,
        common_aggregation: Dict,
        threshold: float = 0.7
    ) -> Dict:
        """
        신뢰도 검증 (Phase 1)
        
        Args:
            job_aggregation: Job 통합 결과
            common_aggregation: Common 통합 결과
            threshold: Confidence 임계값 (기본 0.7)
        
        Returns:
            {
                "low_confidence_competencies": ["financial_literacy", ...],
                "validation_notes": "...",
                "requires_revaluation": False  # deprecated
            }
        """
        
        # 1. Job/Common에서 Low Confidence 역량 추출
        low_confidence_job = job_aggregation.get("low_confidence_competencies", [])
        low_confidence_common = common_aggregation.get("low_confidence_competencies", [])
        
        low_confidence_competencies = low_confidence_job + low_confidence_common
        
        # 2. 검증 노트 생성
        if not low_confidence_competencies:
            validation_notes = "모든 역량의 Confidence가 기준치 이상입니다."
        else:
            validation_notes = (
                f"{len(low_confidence_competencies)}개 역량에서 낮은 Confidence 탐지됨. "
                f"Phase 2에서 상세 분석 예정: {', '.join(low_confidence_competencies)}"
            )
        
        return {
            "low_confidence_competencies": low_confidence_competencies,
            "validation_notes": validation_notes,
            "requires_revaluation": False  # Phase 2에서 판단
        }