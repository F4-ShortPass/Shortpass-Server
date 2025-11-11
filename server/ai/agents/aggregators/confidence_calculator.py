"""
Confidence V2 Calculator
Resume 검증 결과를 반영한 Confidence 재계산 (Rule-based)
"""

from typing import Dict, List


class ConfidenceCalculator:
    """
    Confidence V2 계산기
    
    공식:
        - Resume 검증 없음(none) → interview_confidence 그대로 사용
        - Resume 검증 있음 → interview_confidence × 0.60 + resume_boost × 0.40
    
    resume_boost:
        - "high": 1.0
        - "medium": 0.7
        - "low": 0.4
        - "none": N/A (사용 안 함)
    """
    
    # Resume 검증 강도별 Boost 값
    RESUME_BOOST_MAP = {
        "high": 1.0,
        "medium": 0.7,
        "low": 0.4,
        "none": 0.0
    }
    
    
    @staticmethod
    def calculate_confidence_v2(
        interview_confidence: float,
        verification_strength: str
    ) -> float:
        """
        Confidence V2 계산
        
        전략:
            - Resume 검증 없음(none) → Interview Confidence 그대로
            - Resume 검증 있음 → Interview + Resume 조합
        
        Args:
            interview_confidence: Agent가 계산한 원본 Confidence (0.0~1.0)
            verification_strength: Resume 검증 강도 ("high"/"medium"/"low"/"none")
        
        Returns:
            overall_confidence_v2: 0.0~1.0
        """
        
        if verification_strength == "none":
            # Resume 검증 없음 → Interview Confidence 그대로 사용
            confidence_v2 = interview_confidence
        else:
            # Resume 검증 있음 → Interview + Resume 조합
            resume_boost = ConfidenceCalculator.RESUME_BOOST_MAP.get(
                verification_strength,
                0.0
            )
            
            confidence_v2 = (
                interview_confidence * 0.60 +
                resume_boost * 0.40
            )
        
        # Clamp to [0.3, 0.98]
        confidence_v2 = max(0.3, min(0.98, confidence_v2))
        
        return round(confidence_v2, 2)
    
    
    @staticmethod
    def calculate_for_segments(
        segment_evaluations: List[Dict]
    ) -> List[Dict]:
        """
        Segment 평가 목록에 대해 Confidence V2 일괄 계산
        
        Args:
            segment_evaluations: Resume 검증 결과가 추가된 Segment 평가 목록
                [
                    {
                        "competency": "achievement_motivation",
                        "segment_id": 3,
                        "interview_confidence": 0.85,
                        "resume_verification": {
                            "verified": true,
                            "strength": "high"
                        }
                    },
                    ...
                ]
        
        Returns:
            confidence_v2가 추가된 Segment 평가 목록
        """
        updated = []
        
        for seg_eval in segment_evaluations:
            interview_conf = seg_eval.get("interview_confidence", 0.5)
            
            #  Resume 검증 정보 올바르게 추출
            resume_verification = seg_eval.get("resume_verification", {})
            verification_strength = resume_verification.get("strength", "none")
            
            confidence_v2 = ConfidenceCalculator.calculate_confidence_v2(
                interview_conf,
                verification_strength
            )
            
            # 기존 dict에 추가
            updated_eval = {**seg_eval, "confidence_v2": confidence_v2}
            updated.append(updated_eval)
        
        return updated
    
    
    @staticmethod
    def calculate_competency_avg_confidence(
        competency_segments: List[Dict]
    ) -> float:
        """
        역량별 평균 Confidence V2 계산
        
        Args:
            competency_segments: 특정 역량의 Segment 평가 목록
        
        Returns:
            평균 Confidence V2
        """
        if not competency_segments:
            return 0.5
        
        confidences = [
            seg.get("confidence_v2", 0.5)
            for seg in competency_segments
        ]
        
        avg = sum(confidences) / len(confidences)
        return round(avg, 2)