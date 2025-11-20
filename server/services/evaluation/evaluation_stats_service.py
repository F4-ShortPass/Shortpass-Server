# ai/services/evaluation_stats_service.py
"""
평가 통계 및 모니터링 서비스
"""

import statistics
from typing import Dict, List
from sqlalchemy import select
from sqlalchemy.orm import Session

from models.evaluation import CompetencyEvaluation, EvaluationLog


class EvaluationStatsService:
    """
    평가 통계 서비스
    """
    
    async def get_evaluation_statistics(
        self,
        db: Session,
        job_id: int
    ) -> dict:
        """Job별 평가 통계"""
        stmt = select(CompetencyEvaluation).where(
            CompetencyEvaluation.job_id == job_id,
            CompetencyEvaluation.evaluation_status == "completed"
        )
        
        evaluations = (await db.execute(stmt)).scalars().all()
        
        if not evaluations:
            return {"total_evaluations": 0, "message": "No evaluations"}
        
        scores = [
            e.key_insights.get("weighted_total_score", 0)
            for e in evaluations if e.key_insights
        ]
        
        return {
            "total_evaluations": len(evaluations),
            "average_score": round(statistics.mean(scores), 2) if scores else 0,
            "median_score": round(statistics.median(scores), 2) if scores else 0,
            "std_deviation": round(statistics.stdev(scores), 2) if len(scores) > 1 else 0,
            "score_distribution": self._calculate_distribution(scores),
            "competency_averages": self._calculate_competency_averages(evaluations),
            "bias_warnings": self._detect_bias(scores)
        }
    
    async def get_reasoning_log(
        self,
        db: Session,
        evaluation_id: int
    ) -> dict:
        """추론 로그 조회"""
        evaluation = await db.get(CompetencyEvaluation, evaluation_id)
        
        if not evaluation:
            raise ValueError("Evaluation not found")
        
        stmt = select(EvaluationLog).where(
            EvaluationLog.competency_evaluation_id == evaluation_id
        ).order_by(EvaluationLog.created_at)
        
        logs = (await db.execute(stmt)).scalars().all()
        
        return {
            "evaluation_id": evaluation_id,
            "execution_logs": [
                {
                    "node": log.agent_name,
                    "execution_time_ms": log.execution_time_ms,
                    "error": log.error_message if log.error_occurred else None
                }
                for log in logs
            ],
            "total_execution_time_ms": sum(log.execution_time_ms for log in logs),
            "evaluator_outputs": {
                "job_expertise": evaluation.job_expertise,
                "analytical": evaluation.analytical,
                "execution": evaluation.execution,
                "relationship": evaluation.relationship,
                "resilience": evaluation.resilience,
                "influence": evaluation.influence,
            }
        }
    
    def _calculate_distribution(self, scores: List[float]) -> dict:
        """점수 분포"""
        return {
            "0-59": sum(1 for s in scores if s < 60),
            "60-69": sum(1 for s in scores if 60 <= s < 70),
            "70-79": sum(1 for s in scores if 70 <= s < 80),
            "80-89": sum(1 for s in scores if 80 <= s < 90),
            "90-100": sum(1 for s in scores if s >= 90),
        }
    
    def _calculate_competency_averages(self, evaluations) -> dict:
        """역량별 평균"""
        comp_scores = {name: [] for name in ["job_expertise", "analytical", "execution", 
                                               "relationship", "resilience", "influence"]}
        
        for eval in evaluations:
            for comp_name in comp_scores.keys():
                comp_data = getattr(eval, comp_name)
                if comp_data and "score" in comp_data:
                    comp_scores[comp_name].append(comp_data["score"])
        
        return {
            name: round(statistics.mean(scores), 2) if scores else 0
            for name, scores in comp_scores.items()
        }
    
    def _detect_bias(self, scores: List[float]) -> list:
        """편향 경고"""
        warnings = []
        
        if not scores:
            return warnings
        
        avg = statistics.mean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0
        
        if avg > 85:
            warnings.append({
                "type": "score_inflation",
                "message": f"평균 {avg:.1f}점으로 과도하게 높음"
            })
        
        if std < 5 and len(scores) >= 5:
            warnings.append({
                "type": "low_variance",
                "message": f"표준편차 {std:.1f}로 변별력 부족"
            })
        
        return warnings