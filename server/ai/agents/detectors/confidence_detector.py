"""
Low Confidence Detector (Phase 2)
Confidence가 낮은 역량 탐지 및 원인 분석
LLM 호출 없이 규칙 기반으로 빠르게 처리

탐지 기준:
- overall_confidence < 0.6
- 원인 구분:
  * evidence_strength 낮음 → 지원자 문제 (Quote 부족)
  * internal_consistency 낮음 → 평가 문제 (Agent 간 불일치)
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class LowConfidenceIssue:
    """Low Confidence 정보"""
    competency: str
    overall_confidence: float
    evidence_strength: float
    internal_consistency: float
    reason: str  # "evidence_weak", "consistency_low", "both"
    priority: int  # Confidence 낮은 순위 (1=가장 낮음)
    
    def to_dict(self) -> Dict:
        """Dict 변환"""
        return {
            "competency": self.competency,
            "overall_confidence": round(self.overall_confidence, 2),
            "evidence_strength": round(self.evidence_strength, 2),
            "internal_consistency": round(self.internal_consistency, 2),
            "reason": self.reason,
            "priority": self.priority
        }


class ConfidenceDetector:
    """Low Confidence 탐지기"""
    
    @staticmethod
    def detect_low_confidence(
        all_results: Dict[str, Dict],
        threshold: float = 0.6,
        max_issues: int = 5
    ) -> List[LowConfidenceIssue]:
        """
        모든 역량 평가 결과에서 Low Confidence 탐지
        
        Args:
            all_results: 10개 역량 평가 결과
                {
                    "problem_solving": {
                        "confidence": {
                            "overall_confidence": 0.87,
                            "evidence_strength": 0.8,
                            "internal_consistency": 0.97
                        }
                    },
                    ...
                }
            threshold: Confidence 임계값 (기본 0.6)
            max_issues: 최대 반환 개수 (Phase 3 비용 제한)
        
        Returns:
            Low Confidence 목록 (Confidence 낮은 순 정렬)
        """
        # 1. Low Confidence 역량 추출
        issues = ConfidenceDetector._find_low_confidence(all_results, threshold)
        
        # 2. Confidence 낮은 순으로 정렬
        issues_sorted = sorted(issues, key=lambda i: i.overall_confidence)
        
        # 3. Priority 부여
        for idx, issue in enumerate(issues_sorted, start=1):
            issue.priority = idx
        
        # 4. 최대 개수 제한
        return issues_sorted[:max_issues]
    
    @staticmethod
    def _find_low_confidence(
        all_results: Dict[str, Dict],
        threshold: float
    ) -> List[LowConfidenceIssue]:
        """Low Confidence 역량 찾기 및 원인 분석"""
        issues = []
        
        for comp_name, result in all_results.items():
            # 에러가 있는 결과는 스킵
            if "error" in result:
                continue
            
            # Confidence 정보 추출
            confidence = result.get("confidence", {})
            overall_confidence = confidence.get("overall_confidence", 1.0)
            evidence_strength = confidence.get("evidence_strength", 1.0)
            internal_consistency = confidence.get("internal_consistency", 1.0)
            
            # Threshold 미만인 경우만
            if overall_confidence < threshold:
                # 원인 분석
                reason = ConfidenceDetector._analyze_reason(
                    evidence_strength,
                    internal_consistency,
                    threshold
                )
                
                issues.append(LowConfidenceIssue(
                    competency=comp_name,
                    overall_confidence=overall_confidence,
                    evidence_strength=evidence_strength,
                    internal_consistency=internal_consistency,
                    reason=reason,
                    priority=0  # 아직 미할당
                ))
        
        return issues
    
    @staticmethod
    def _analyze_reason(
        evidence_strength: float,
        internal_consistency: float,
        threshold: float
    ) -> str:
        """
        Low Confidence 원인 분석
        
        Returns:
            "evidence_weak": Evidence 부족 (지원자 문제)
            "consistency_low": Agent 간 불일치 (평가 문제)
            "both": 둘 다 문제
        """
        evidence_low = evidence_strength < threshold
        consistency_low = internal_consistency < threshold
        
        if evidence_low and consistency_low:
            return "both"
        elif evidence_low:
            return "evidence_weak"
        elif consistency_low:
            return "consistency_low"
        else:
            # overall만 낮은 경우 (드문 경우)
            # 둘 중 더 낮은 쪽을 원인으로
            if evidence_strength < internal_consistency:
                return "evidence_weak"
            else:
                return "consistency_low"
    
    @staticmethod
    def format_issues_for_log(issues: List[LowConfidenceIssue]) -> str:
        """Low Confidence 정보를 로그용 문자열로 포맷"""
        if not issues:
            return "Low Confidence 없음"
        
        lines = [f"총 {len(issues)}개의 Low Confidence 역량 탐지:"]
        for issue in issues:
            reason_str = {
                "evidence_weak": "증거 부족 (지원자)",
                "consistency_low": "일관성 부족 (평가)",
                "both": "증거+일관성 모두 부족"
            }.get(issue.reason, issue.reason)
            
            lines.append(
                f"  Priority {issue.priority}. {issue.competency}: "
                f"Confidence={issue.overall_confidence:.2f} "
                f"(Evidence={issue.evidence_strength:.2f}, "
                f"Consistency={issue.internal_consistency:.2f}) "
                f"→ {reason_str}"
            )
        
        return "\n".join(lines)
    
    @staticmethod
    def get_issue_summary(issues: List[LowConfidenceIssue]) -> Dict:
        """
        Low Confidence 요약 정보
        
        Returns:
            {
                "total_issues": 2,
                "min_confidence": 0.54,
                "affected_competencies": ["financial_literacy", "industry_learning"],
                "reason_breakdown": {
                    "evidence_weak": 1,
                    "consistency_low": 1,
                    "both": 0
                }
            }
        """
        if not issues:
            return {
                "total_issues": 0,
                "min_confidence": 1.0,
                "affected_competencies": [],
                "reason_breakdown": {
                    "evidence_weak": 0,
                    "consistency_low": 0,
                    "both": 0
                }
            }
        
        reason_breakdown = {
            "evidence_weak": sum(1 for i in issues if i.reason == "evidence_weak"),
            "consistency_low": sum(1 for i in issues if i.reason == "consistency_low"),
            "both": sum(1 for i in issues if i.reason == "both")
        }
        
        return {
            "total_issues": len(issues),
            "min_confidence": round(min(i.overall_confidence for i in issues), 2),
            "affected_competencies": [i.competency for i in issues],
            "reason_breakdown": reason_breakdown
        }
    
    @staticmethod
    def categorize_by_reason(issues: List[LowConfidenceIssue]) -> Dict[str, List[str]]:
        """
        원인별로 역량 분류
        
        Returns:
            {
                "evidence_weak": ["financial_literacy"],
                "consistency_low": ["industry_learning"],
                "both": []
            }
        """
        categorized = {
            "evidence_weak": [],
            "consistency_low": [],
            "both": []
        }
        
        for issue in issues:
            categorized[issue.reason].append(issue.competency)
        
        return categorized