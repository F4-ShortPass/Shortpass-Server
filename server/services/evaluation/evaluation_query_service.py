# ai/services/evaluation_query_service.py
"""
평가 결과 조회 서비스 (구직자/기업용)
"""

from typing import Dict, Any, List
from sqlalchemy import select
from sqlalchemy.orm import Session

from models.evaluation import CompetencyEvaluation
from models.interview import InterviewSession
from api.evaluation import EvaluationResultResponse, CompetencyReasoningLog


class EvaluationQueryService:
    """
    평가 결과 조회 서비스
    """
    
    async def get_evaluation_for_applicant(
        self,
        db: Session,
        applicant_id: int,
        evaluation_id: int
    ) -> dict:
        """
        구직자용 평가 결과 (성장 중심)
        """
        evaluation = await self._get_and_verify_evaluation(
            db, evaluation_id, applicant_id=applicant_id
        )
        
        if evaluation.evaluation_status != "completed":
            return EvaluationResultResponse(
                evaluation_id=evaluation.id,
                applicant_id=evaluation.applicant_id,
                job_id=evaluation.job_id,
                match_score=0.0, # 또는 적절한 기본값
                normalized_score=None,
                overall_feedback=f"Evaluation is still {evaluation.evaluation_status}",
                hiring_recommendation=None,
                reasoning_log={},
            )
        
        return EvaluationResultResponse(
            evaluation_id=evaluation.id,
            applicant_id=evaluation.applicant_id,
            job_id=evaluation.job_id,
            match_score=evaluation.match_score,
            normalized_score=evaluation.normalized_score,
            overall_feedback=evaluation.overall_feedback,
            hiring_recommendation=evaluation.hiring_recommendation,
            reasoning_log=evaluation.reasoning_log or {}
        )
    
    async def get_evaluation_for_company(
        self,
        db: Session,
        job_id: int,
        applicant_id: int
    ) -> dict:
        """
        기업용 평가 결과 (채용 판단 중심)
        """
        evaluation = await self._find_evaluation_by_job_and_applicant(
            db, job_id, applicant_id
        )
        
        if evaluation.evaluation_status != "completed":
            return {
                "status": evaluation.evaluation_status,
                "message": "Evaluation is still in progress"
            }
        
        if evaluation.evaluation_status != "completed":
            return EvaluationResultResponse(
                evaluation_id=evaluation.id,
                applicant_id=applicant_id,
                job_id=evaluation.job_id,
                match_score=0.0, # 또는 적절한 기본값
                normalized_score=None,
                overall_feedback=f"Evaluation is still {evaluation.evaluation_status}",
                hiring_recommendation=None,
                reasoning_log={},
            )
        
        return EvaluationResultResponse(
            evaluation_id=evaluation.id,
            applicant_id=applicant_id,
            job_id=evaluation.job_id,
            match_score=evaluation.match_score,
            normalized_score=evaluation.normalized_score,
            overall_feedback=evaluation.overall_feedback,
            hiring_recommendation=evaluation.hiring_recommendation,
            reasoning_log=evaluation.reasoning_log or {}
        )
    
    async def get_job_applicants_summary(
        self,
        db: Session,
        job_id: int,
        min_score: float = 0,
        sort_by: str = "score"
    ) -> dict:
        """
        기업용 - 지원자 목록
        """
        stmt = select(CompetencyEvaluation).join(
            InterviewSession,
            CompetencyEvaluation.interview_id == InterviewSession.id
        ).where(
            CompetencyEvaluation.job_id == job_id,
            CompetencyEvaluation.evaluation_status == "completed"
        )
        
        evaluations = (await db.execute(stmt)).scalars().all()
        
        applicants = []
        for eval in evaluations:
            weighted_score = eval.key_insights.get("weighted_total_score", 0) if eval.key_insights else 0
            
            if weighted_score < min_score:
                continue
            
            interview = await db.get(InterviewSession, eval.interview_id)
            
            applicants.append({
                "applicant_id": interview.applicant_id,
                "evaluation_id": eval.id,
                "weighted_total_score": weighted_score,
                "job_requirement_fit_score": eval.job_requirement_fit_score,
                "hiring_recommendation": eval.hiring_recommendation,
                "top_3_strengths": eval.key_insights.get("top_3_strengths", []) if eval.key_insights else [],
                "top_3_concerns": eval.key_insights.get("top_3_concerns", []) if eval.key_insights else [],
                "created_at": eval.created_at
            })
        
        if sort_by == "score":
            applicants.sort(key=lambda x: x["weighted_total_score"], reverse=True)
        elif sort_by == "fit_score":
            applicants.sort(key=lambda x: x["job_requirement_fit_score"], reverse=True)
        
        return {
            "total_count": len(evaluations),
            "filtered_count": len(applicants),
            "applicants": applicants
        }
    
    # ==================== Private Methods ====================
    
    async def _get_and_verify_evaluation(
        self, 
        db: Session, 
        evaluation_id: int,
        applicant_id: int = None
    ) -> CompetencyEvaluation:
        """평가 조회 및 권한 확인"""
        evaluation = await db.get(CompetencyEvaluation, evaluation_id)
        
        if not evaluation:
            raise ValueError("Evaluation not found")
        
        if applicant_id:
            interview = await db.get(InterviewSession, evaluation.interview_id)
            if interview.applicant_id != applicant_id:
                raise ValueError("Unauthorized access")
        
        return evaluation
    
    async def _find_evaluation_by_job_and_applicant(
        self,
        db: Session,
        job_id: int,
        applicant_id: int
    ) -> CompetencyEvaluation:
        """Job + Applicant로 평가 찾기"""
        stmt = select(CompetencyEvaluation).join(
            InterviewSession,
            CompetencyEvaluation.interview_id == InterviewSession.id
        ).where(
            CompetencyEvaluation.job_id == job_id,
            InterviewSession.applicant_id == applicant_id
        )
        
        evaluation = (await db.execute(stmt)).scalar_one_or_none()
        
        if not evaluation:
            raise ValueError("Evaluation not found")
        
        return evaluation
    
    def _extract_competencies_simple(self, evaluation) -> dict:
        """구직자용 - 간단 버전"""
        competencies = {}
        
        for comp_name in ["job_expertise", "analytical", "execution", 
                          "relationship", "resilience", "influence"]:
            comp_data = getattr(evaluation, comp_name)
            
            if not comp_data:
                continue
            
            competencies[comp_name] = {
                "score": comp_data.get("score"),
                "reasoning": comp_data.get("reasoning"),
                "evidence": comp_data.get("evidence", []),
                "strengths": comp_data.get("strengths", []),
                "areas_for_improvement": comp_data.get("areas_for_improvement", []),
            }
        
        return competencies
    
    def _extract_competencies_detailed(self, evaluation) -> dict:
        """기업용 - 상세 버전"""
        competencies = {}
        
        for comp_name in ["job_expertise", "analytical", "execution", 
                          "relationship", "resilience", "influence"]:
            comp_data = getattr(evaluation, comp_name)
            
            if not comp_data:
                continue
            
            # 모든 필드 포함
            competencies[comp_name] = comp_data
        
        return competencies
    
    def _add_evidence_to_insights(self, evaluation) -> dict:
        """key_insights에 근거 추가"""
        key_insights = evaluation.key_insights or {}
        
        if "top_3_strengths" in key_insights:
            key_insights["top_3_strengths_with_evidence"] = [
                {
                    "strength": s,
                    "evidence": self._find_supporting_evidence(s, evaluation)
                }
                for s in key_insights["top_3_strengths"]
            ]
        
        if "top_3_concerns" in key_insights:
            key_insights["top_3_concerns_with_evidence"] = [
                {
                    "concern": c,
                    "evidence": self._find_supporting_evidence(c, evaluation)
                }
                for c in key_insights["top_3_concerns"]
            ]
        
        return key_insights
    
    def _find_supporting_evidence(self, claim: str, evaluation) -> list:
        """주장에 대한 근거 찾기"""
        evidence = []
        
        for comp_name in ["job_expertise", "analytical", "execution", 
                          "relationship", "resilience", "influence"]:
            comp_data = getattr(evaluation, comp_name)
            if not comp_data:
                continue
            
            for ev in comp_data.get("evidence", []):
                # 간단한 키워드 매칭
                if any(word in ev.get("quote", "").lower() for word in claim.lower().split()):
                    evidence.append({
                        "competency": comp_name,
                        "turn": ev.get("turn"),
                        "quote": ev.get("quote")
                    })
        
        return evidence[:3]  # 상위 3개만
    
    def _extract_all_strengths(self, evaluation, limit: int) -> list:
        """전체 강점 추출"""
        strengths = []
        
        for comp_name in ["job_expertise", "analytical", "execution", 
                          "relationship", "resilience", "influence"]:
            comp_data = getattr(evaluation, comp_name)
            if comp_data and "strengths" in comp_data:
                strengths.extend(comp_data["strengths"])
        
        return strengths[:limit]
    
    def _extract_all_improvements(self, evaluation, limit: int) -> list:
        """전체 개선사항 추출"""
        improvements = []
        
        for comp_name in ["job_expertise", "analytical", "execution", 
                          "relationship", "resilience", "influence"]:
            comp_data = getattr(evaluation, comp_name)
            if comp_data and "areas_for_improvement" in comp_data:
                improvements.extend(comp_data["areas_for_improvement"])
        
        return improvements[:limit]