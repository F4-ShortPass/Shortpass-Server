"""
Final Integrator (Phase 4)
최종 점수 통합 및 신뢰도 평가

처리 내용:
1. 개별 역량 점수는 원본 유지 (중요!)
2. Phase 3 협업 결과 반영 (점수 조정)
3. 평균 Confidence 계산
4. 신뢰도 레벨 판단
5. Job 60% + Common 40% 최종 점수

핵심 원칙:
- 개별 역량 점수 = 원본 그대로 (사용자에게 표시)
- 협업 조정은 별도 필드에 기록
- Confidence는 참고용, 가중치는 기업 설정 그대로
"""

# LEGACY: early experiment version (kept for history). Not wired in evaluation.py

from typing import Dict, List, Tuple


class FinalIntegrator:
    """최종 통합자"""
    
    @staticmethod
    def integrate(
        job_aggregation: Dict,
        common_aggregation: Dict,
        mediation_results: List[Dict],
        adversarial_results: List[Dict],
        collaboration_count: int,
        job_common_ratio: Dict[str, float] = None
    ) -> Dict:
        """
        최종 점수 통합
        
        Args:
            job_aggregation: Job 5개 역량 통합 결과
            common_aggregation: Common 5개 역량 통합 결과
            mediation_results: Evidence 충돌 중재 결과
            adversarial_results: Adversarial 재평가 결과
            collaboration_count: 협업 처리 횟수
            job_common_ratio: Job/Common 비율 (기본 {"job": 0.6, "common": 0.4})
        
        Returns:
            최종 통합 결과
        """
        
        # 기본값
        if job_common_ratio is None:
            job_common_ratio = {"job": 0.6, "common": 0.4}
        
        # 1. 협업 조정 사항 정리 
        adjustments = FinalIntegrator._compile_adjustments(
            mediation_results,
            adversarial_results
        )
        
        
        # 2. 평균 Confidence 계산 (10개 역량)
        all_competencies = {
            **FinalIntegrator._extract_competencies_from_aggregation(job_aggregation),
            **FinalIntegrator._extract_competencies_from_aggregation(common_aggregation)
        }
        
        avg_confidence = FinalIntegrator._calculate_avg_confidence(all_competencies)
        
        
        # 3. 신뢰도 레벨 판단    
        reliability = FinalIntegrator._determine_reliability(
            avg_confidence,
            collaboration_count
        )
        
        
        # 4. 최종 점수 계산
        job_score = job_aggregation["overall_job_score"]
        common_score = common_aggregation["overall_common_score"]
        
        final_score = (
            job_score * job_common_ratio["job"] +
            common_score * job_common_ratio["common"]
        )
        
        
        # 5. 결과 구성
        return {
            "final_score": round(final_score, 2),
            "job_score": round(job_score, 2),
            "common_score": round(common_score, 2),
            "job_common_ratio": job_common_ratio,
            "avg_confidence": round(avg_confidence, 2),
            "reliability": reliability,
            "adjustments_summary": {
                "mediation_count": len(mediation_results),
                "adversarial_count": len(adversarial_results),
                "total_adjustments": len(mediation_results) + len(adversarial_results),
                "adjustments": adjustments
            },
            "interpretation": FinalIntegrator._generate_interpretation(
                final_score,
                reliability,
                collaboration_count
            )
        }
    
    @staticmethod
    def _compile_adjustments(
        mediation_results: List[Dict],
        adversarial_results: List[Dict]
    ) -> Dict:
        """
        협업 조정 사항 정리
        
        Returns:
            {
                "mediation": [
                    {
                        "segment_id": "5",
                        "primary_competency": "structured_thinking",
                        "adjusted_scores": {...}
                    }
                ],
                "adversarial": [
                    {
                        "competency": "financial_literacy",
                        "score_change": -4,  // 82 → 78
                        "confidence_change": +0.14  // 0.54 → 0.68
                    }
                ]
            }
        """
        
        mediation_summary = []
        for result in mediation_results:
            mediation_summary.append({
                "segment_id": result.get("segment_id"),
                "primary_competency": result.get("primary_competency"),
                "adjusted_scores": result.get("adjusted_scores", {})
            })
        
        adversarial_summary = []
        for result in adversarial_results:
            orig = result.get("original_score", 0)
            adj = result.get("adjusted_score", 0)
            adversarial_summary.append({
                "competency": result.get("competency"),
                "score_change": round(adj - orig, 1),
                "confidence_change": round(
                    result.get("confidence_adjusted", 0) - 
                    result.get("original_confidence", 0), 
                    2
                ) if "original_confidence" in result else None
            })
        
        return {
            "mediation": mediation_summary,
            "adversarial": adversarial_summary
        }
    
    @staticmethod
    def _extract_competencies_from_aggregation(aggregation: Dict) -> Dict[str, Dict]:
        """
        Aggregation에서 개별 역량 결과 추출
        
        Returns:
            {
                "problem_solving": {...},
                "structured_thinking": {...},
                ...
            }
        """
        competencies = {}
        
        # Job 또는 Common의 5개 역량 추출
        for key, value in aggregation.items():
            # overall_*, weights, aggregated_at 등 메타 필드 제외
            if isinstance(value, dict) and "overall_score" in value:
                competencies[key] = value
        
        return competencies
    
    @staticmethod
    def _calculate_avg_confidence(all_competencies: Dict[str, Dict]) -> float:
        """
        10개 역량의 평균 Confidence 계산
        
        Args:
            all_competencies: 10개 역량 결과
        
        Returns:
            평균 Confidence (0.0-1.0)
        """
        confidences = []
        
        for result in all_competencies.values():
            confidence = result.get("confidence", {}).get("overall_confidence", 0.0)
            confidences.append(confidence)
        
        if not confidences:
            return 0.0
        
        return sum(confidences) / len(confidences)
    
    @staticmethod
    def _determine_reliability(
        avg_confidence: float,
        collaboration_count: int
    ) -> Dict:
        """
        신뢰도 레벨 판단
        
        기준:
        - 평균 Confidence + 협업 횟수로 종합 판단
        - 협업이 많을수록 불확실성이 있었다는 의미
        
        Returns:
            {
                "level": "매우 높음" | "높음" | "중간" | "낮음",
                "score": 0.9,  // 0-1 점수
                "note": "..."
            }
        """
        
        # 협업 횟수 페널티 (0~1회: 0, 2~3회: -0.05, 4~5회: -0.10, 6회+: -0.15)
        if collaboration_count <= 1:
            collab_penalty = 0.0
        elif collaboration_count <= 3:
            collab_penalty = 0.05
        elif collaboration_count <= 5:
            collab_penalty = 0.10
        else:
            collab_penalty = 0.15
        
        # 조정된 신뢰도 점수
        reliability_score = max(0.0, avg_confidence - collab_penalty)
        
        # 레벨 판단
        if reliability_score >= 0.85:
            level = "매우 높음"
            note = "모든 역량에서 일관된 평가. 추가 검증 불필요."
        elif reliability_score >= 0.75:
            level = "높음"
            note = "대부분 역량에서 신뢰할 만한 평가. 일부 역량은 추가 질문 권장."
        elif reliability_score >= 0.65:
            level = "중간"
            note = "일부 역량에서 불확실성 존재. 추가 면접 또는 과제 평가 권장."
        else:
            level = "낮음"
            note = "다수 역량에서 증거 부족 또는 평가 불일치. 추가 평가 강력 권장."
        
        return {
            "level": level,
            "score": round(reliability_score, 2),
            "avg_confidence": round(avg_confidence, 2),
            "collaboration_penalty": round(collab_penalty, 2),
            "note": note
        }
    
    @staticmethod
    def _generate_interpretation(
        final_score: float,
        reliability: Dict,
        collaboration_count: int
    ) -> str:
        """최종 점수 해석"""
        
        # 점수 등급
        if final_score >= 80:
            grade = "우수"
            grade_note = "상위 10% 수준"
        elif final_score >= 70:
            grade = "양호"
            grade_note = "상위 30% 수준"
        elif final_score >= 60:
            grade = "보통"
            grade_note = "평균 수준"
        else:
            grade = "미흡"
            grade_note = "평균 이하"
        
        # 신뢰도 멘트
        reliability_note = ""
        if reliability["level"] in ["매우 높음", "높음"]:
            reliability_note = "평가 신뢰도가 높아 이 점수는 신뢰할 수 있습니다."
        elif reliability["level"] == "중간":
            reliability_note = "일부 역량에서 불확실성이 있어, 추가 검증이 권장됩니다."
        else:
            reliability_note = "평가 신뢰도가 낮아, 이 점수는 참고용으로만 사용하고 추가 평가가 필요합니다."
        
        # 협업 멘트
        collab_note = ""
        if collaboration_count == 0:
            collab_note = "모든 역량이 독립 평가만으로 충분했습니다."
        elif collaboration_count <= 3:
            collab_note = f"일부 역량({collaboration_count}개)에서 추가 검증이 수행되었습니다."
        else:
            collab_note = f"다수 역량({collaboration_count}개)에서 협업 평가가 필요했습니다."
        
        return f"""
최종 점수: {final_score:.1f}점 ({grade}, {grade_note})
신뢰도: {reliability['level']} (점수: {reliability['score']:.2f})

{reliability_note}
{collab_note}
        """.strip()