"""
평가 서비스 (LangGraph 통합)
"""

import os
import json
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path
from dotenv import load_dotenv
import asyncio
from ai.agents.graph.evaluation import create_evaluation_graph
from services.storage.s3_service import S3Service
from sqlalchemy.orm import Session
from db.database import SessionLocal
from models.evaluation import Evaluation
from models.interview import Applicant, Company, InterviewSession, InterviewStatus
from models.job import Job

env_path = Path(__file__).parent.parent.parent / '.env'
# Force override so .env values (e.g., OPENAI_API_KEY) are used even if the shell has others.
load_dotenv(env_path, override=True)

from ai.prompts.competency_agents.common_competencies.problem_solving_prompt import create_problem_solving_evaluation_prompt
from ai.prompts.competency_agents.common_competencies.organizational_fit_prompt import create_organizational_fit_evaluation_prompt
from ai.prompts.competency_agents.common_competencies.growth_potential_prompt import create_growth_potential_evaluation_prompt
from ai.prompts.competency_agents.common_competencies.interpersonal_skill_prompt import create_interpersonal_skill_evaluation_prompt
from ai.prompts.competency_agents.common_competencies.achievement_motivation_prompt import create_achievement_motivation_evaluation_prompt

from ai.prompts.competency_agents.job_competencies.customer_journey_marketing_prompt import create_customer_journey_marketing_evaluation_prompt
from ai.prompts.competency_agents.job_competencies.data_analysis_prompt import create_md_data_analysis_evaluation_prompt
from ai.prompts.competency_agents.job_competencies.seasonal_strategy_kpi_prompt import create_seasonal_strategy_kpi_evaluation_prompt
from ai.prompts.competency_agents.job_competencies.stakeholder_collaboration_prompt import create_stakeholder_collaboration_evaluation_prompt
from ai.prompts.competency_agents.job_competencies.value_chain_optimization_prompt import create_value_chain_optimization_evaluation_prompt

from openai import AsyncOpenAI

PROMPT_GENERATORS = {
    # Common Competencies (5개)
    "problem_solving": create_problem_solving_evaluation_prompt,
    "organizational_fit": create_organizational_fit_evaluation_prompt,
    "growth_potential": create_growth_potential_evaluation_prompt,
    "interpersonal_skill": create_interpersonal_skill_evaluation_prompt, 
    "achievement_motivation": create_achievement_motivation_evaluation_prompt,
    
    # Job Competencies (5개)
    "customer_journey_marketing": create_customer_journey_marketing_evaluation_prompt,
    "md_data_analysis": create_md_data_analysis_evaluation_prompt,
    "seasonal_strategy_kpi": create_seasonal_strategy_kpi_evaluation_prompt,
    "stakeholder_collaboration": create_stakeholder_collaboration_evaluation_prompt,
    "value_chain_optimization": create_value_chain_optimization_evaluation_prompt,
}


class EvaluationService:
    """평가 서비스"""
    
    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        
        from core.config import settings
        self.openai_client = AsyncOpenAI(api_key=api_key)
        self.graph = create_evaluation_graph()
        self.s3_service = S3Service(bucket_name=settings.S3_BUCKET_NAME, region_name=settings.AWS_REGION)
    
    def _load_prompts(self, transcript: Dict) -> Dict[str, str]:
        """프롬프트 로딩"""
        transcript_str = json.dumps(transcript, ensure_ascii=False, indent=2)
        return {
            name: generator(transcript_str)
            for name, generator in PROMPT_GENERATORS.items()
        }
    
    def _ensure_required_fields(self, result: Dict) -> Dict:
        """
        필수 필드 강제 보장 (Post-processing)
        
        누락된 필드를 자동 생성:
            1. key_observations (역량별 핵심 관찰)
            2. resume_verification_summary (Resume 검증 근거)
            3. overall_evaluation_summary (종합 심사평)
        """
        print("\n" + "="*60)
        print("[필수 필드 검증] 시작...")
        print("="*60)
        
        # ========================================
        # 1. Stage 2: aggregated_competencies 검증
        # ========================================
        aggregated_competencies = result.get("aggregated_competencies", {})
        
        for comp_name, comp_data in aggregated_competencies.items():
            print(f"\n[{comp_name}] 검증 중...")
            
            # 1-1. key_observations 보장
            if "key_observations" not in comp_data or not comp_data.get("key_observations"):
                key_obs = []
                # Strengths에서 추출 (최대 2개)
                strengths = comp_data.get("strengths", [])
                key_obs.extend(strengths[:2])
                
                # Weaknesses에서 추출 (최대 1개)
                weaknesses = comp_data.get("weaknesses", [])
                if weaknesses:
                    key_obs.append(weaknesses[0])
                
                # 최소 3개 보장
                if len(key_obs) < 3:
                    score = comp_data.get("overall_score", 0)
                    conf_v2 = comp_data.get("confidence_v2", 0.5)
                    
                    if score >= 75:
                        key_obs.append(f"신입 기준 우수한 수준 (점수: {score}, 신뢰도: {conf_v2:.2f})")
                    elif score >= 60:
                        key_obs.append(f"신입 기준 양호한 수준 (점수: {score}, 신뢰도: {conf_v2:.2f})")
                    elif score >= 50:
                        key_obs.append(f"신입 기준 평균 수준 (점수: {score}, 신뢰도: {conf_v2:.2f})")
                    else:
                        key_obs.append(f"신입 기준 미흡한 수준 (점수: {score}, 신뢰도: {conf_v2:.2f})")
                
                # 중복 제거 및 최대 5개
                key_obs = list(dict.fromkeys(key_obs))[:5]
                comp_data["key_observations"] = key_obs
            
            
            # 1-2. resume_verification_summary 보장
            if "resume_verification_summary" not in comp_data:
                print(f"    resume_verification_summary 누락 → 생성")
                
                # segment_evaluations_with_resume에서 추출
                segment_evals = result.get("segment_evaluations_with_resume", [])
                comp_segments = [s for s in segment_evals if s.get("competency") == comp_name]
                
                verified_segments = [
                    s for s in comp_segments 
                    if s.get("resume_verified", False)
                ]
                
                high_strength_segments = [
                    s for s in verified_segments
                    if s.get("verification_strength") == "high"
                ]
                
                # 검증 강도 순 정렬
                strength_order = {"high": 3, "medium": 2, "low": 1, "none": 0}
                verified_segments_sorted = sorted(
                    verified_segments,
                    key=lambda s: strength_order.get(s.get("verification_strength", "none"), 0),
                    reverse=True
                )
                
                # 상위 3개만 추출
                key_evidence = []
                for seg in verified_segments_sorted[:3]:
                    key_evidence.append({
                        "segment_id": seg.get("segment_id"),
                        "quote": seg.get("quote_text", ""),
                        "resume_section": "unknown",
                        "matched_content": ", ".join(seg.get("resume_evidence", [])),
                        "verification_strength": seg.get("verification_strength", "none"),
                        "reasoning": f"Resume 증거: {', '.join(seg.get('resume_evidence', []))}"
                    })
                
                comp_data["resume_verification_summary"] = {
                    "verified_count": len(verified_segments),
                    "high_strength_count": len(high_strength_segments),
                    "key_evidence": key_evidence
                }
                
                print(f"    → resume_verification_summary 생성 (verified: {len(verified_segments)}개, high: {len(high_strength_segments)}개)")
        
        result["aggregated_competencies"] = aggregated_competencies

        # 요약 로그: Resume 검증 총합
        total_verified = sum(
            comp.get("resume_verification_summary", {}).get("verified_count", 0)
            for comp in aggregated_competencies.values()
        )
        total_high = sum(
            comp.get("resume_verification_summary", {}).get("high_strength_count", 0)
            for comp in aggregated_competencies.values()
        )
        print("\n[요약] Resume 검증 총합")
        print(f"  - verified_count 합계: {total_verified}")
        print(f"  - high_strength_count 합계: {total_high}")
        
        
        # ========================================
        # 2. Stage 3: final_result 검증
        # ========================================
        final_result = result.get("final_result", {})
        
        # 2-1. overall_evaluation_summary 보장
        if "overall_evaluation_summary" not in final_result or not final_result.get("overall_evaluation_summary"):
            print(f"\n  overall_evaluation_summary 누락 → Fallback 생성")
            
            final_score = result.get("final_score", 0)
            avg_confidence = result.get("avg_confidence", 0.5)
            
            # 간단한 템플릿 심사평
            fallback = (
                f"지원자는 패션 MD 직무에 필요한 역량을 전반적으로 갖추고 있습니다 "
                f"(종합 점수: {final_score:.1f}점, 평균 신뢰도: {avg_confidence:.2f}). "
                f"신입 기준으로 평가되며, 입사 후 체계적인 교육과 멘토링을 통해 성장이 기대됩니다."
            )
            final_result["overall_evaluation_summary"] = fallback
            
            print(f"  → 생성됨 ({len(fallback)} chars)")
        else:
            print(f"\n overall_evaluation_summary 존재 ({len(final_result['overall_evaluation_summary'])} chars)")
        
        
        # 2-2. competency_details에 key_observations 및 resume_verification_summary 포함
        competency_details = final_result.get("competency_details", {})
        
        for comp_name, comp_detail in competency_details.items():
            # aggregated_competencies에서 복사
            if comp_name in aggregated_competencies:
                agg_comp = aggregated_competencies[comp_name]
                
                if "key_observations" not in comp_detail:
                    comp_detail["key_observations"] = agg_comp.get("key_observations", [])
                
                if "resume_verification_summary" not in comp_detail:
                    comp_detail["resume_verification_summary"] = agg_comp.get("resume_verification_summary", {
                        "verified_count": 0,
                        "high_strength_count": 0,
                        "key_evidence": []
                    })
        
        final_result["competency_details"] = competency_details
        result["final_result"] = final_result
        
        
        print("\n" + "="*60)
        print("[필수 필드 검증] 완료")
        print("="*60 + "\n")
        
        return result
    
    async def evaluate_interview(
        self,
        interview_id: int,
        applicant_id: int,
        job_id: int,
        transcript: Dict,
        competency_weights: Dict[str, float],
        resume_data: Optional[Dict] = None,
        session_id: Optional[int] = None
    ) -> Dict:
        """
        면접 평가 실행

        Args:
            interview_id: 면접 ID
            applicant_id: 지원자 ID
            job_id: JD ID
            transcript: 면접 Transcript JSON
            competency_weights: 10개 역량 가중치
            resume_data: 파싱된 Resume JSON (선택적)
            session_id: 세션 ID (RAG 사용 시, 선택적)

        Returns:
            평가 결과
        """

        # Job에서 역량 정보 로드
        print(f"\n[평가 초기화] Job {job_id}에서 역량 정보 로딩...")
        job_competencies = None
        try:
            from db.database import SessionLocal
            db = SessionLocal()
            try:
                job = db.query(Job).filter(Job.id == job_id).first()
                if job and job.dynamic_evaluation_criteria:
                    job_competencies = job.dynamic_evaluation_criteria
                    print(f"  ✓ {len(job_competencies)}개 직무 역량 로드 완료")
                    for i, comp in enumerate(job_competencies[:5], 1):
                        print(f"    {i}. {comp.get('name')}")
                else:
                    print(f"  ⚠ Job {job_id}에 역량 정보 없음 - 기본 역량 사용")
            finally:
                db.close()
        except Exception as e:
            print(f"  ✗ 역량 로드 실패: {e}")
            print(f"  → 기본 역량으로 진행")

        transcript_content = transcript
        transcript_s3_url = f"s3://{self.s3_service.bucket_name}/transcripts/{interview_id}.json"
        prompts = self._load_prompts(transcript_content)

        # Initial State 구성
        initial_state = {
            # 기본 정보
            "interview_id": interview_id,
            "applicant_id": applicant_id,
            "job_id": job_id,
            "transcript_s3_url": transcript_s3_url,
            "transcript_content": transcript_content,
            "transcript": transcript_content,
            "resume_data": resume_data,
            "openai_client": self.openai_client,
            "prompts": prompts,

            # 가중치
            "competency_weights": competency_weights,

            # JD에서 추출한 직무 역량 (동적)
            "job_competencies": job_competencies,
            "session_id": session_id,
            
            # Stage 1 결과 (초기화)
            "achievement_motivation_result": None,
            "growth_potential_result": None,
            "interpersonal_skill_result": None, 
            "organizational_fit_result": None,
            "problem_solving_result": None,
            "customer_journey_marketing_result": None, 
            "md_data_analysis_result": None, 
            "seasonal_strategy_kpi_result": None, 
            "stakeholder_collaboration_result": None, 
            "value_chain_optimization_result": None, 
            
            # Stage 2 결과 (초기화)
            "segment_evaluations_with_resume": [],
            "confidence_v2_calculated": False,
            "segment_overlap_adjustments": [],
            "cross_competency_flags": [],
            "aggregated_competencies": {},
            "low_confidence_list": [],
            "requires_collaboration": False,
            
            # Stage 3 결과 (초기화)
            "collaboration_results": [],
            "collaboration_count": 0,
            
            # Final 결과 (초기화)
            "final_score": None,
            "avg_confidence": None,
            "final_reliability": None,
            "reliability_note": None,
            "final_result": None,
            
            # Presentation 결과 (초기화)
            "presentation_result": None,
            
            # 메타 정보
            "started_at": datetime.now(),
            "completed_at": None,
            "errors": [],
            "execution_logs": []
        }
        
        # 그래프 실행
        print("\n" + "="*80)
        print(f"평가 시작: Interview ID {interview_id}")
        print("="*80)
        
        result = await self.graph.ainvoke(initial_state)
        
        #  필수 필드 강제 보장
        result = self._ensure_required_fields(result)
        
        # 공통 런 타임스탬프
        run_timestamp = result.get("started_at")
        if isinstance(run_timestamp, datetime):
            run_ts_str = run_timestamp.strftime("%Y%m%dT%H%M%S")
        else:
            run_ts_str = datetime.now().strftime("%Y%m%dT%H%M%S")
        evaluation_base_prefix = f"evaluations/{interview_id}/{run_ts_str}"

        # S3 업로드
        execution_logs_s3_key = f"logs/evaluations/{applicant_id}/{interview_id}/{run_ts_str}_execution_logs.json"
        agent_logs_s3_url = self.s3_service.upload_json(
            execution_logs_s3_key, result["execution_logs"]
        )

        # Stage 1 결과만 별도로 저장 (Resume 검증 전 raw agent 결과)
        stage1_payload = {
            "achievement_motivation": result.get("achievement_motivation_result"),
            "growth_potential": result.get("growth_potential_result"),
            "interpersonal_skill": result.get("interpersonal_skill_result"),
            "organizational_fit": result.get("organizational_fit_result"),
            "problem_solving": result.get("problem_solving_result"),
            "customer_journey_marketing": result.get("customer_journey_marketing_result"),
            "md_data_analysis": result.get("md_data_analysis_result"),
            "seasonal_strategy_kpi": result.get("seasonal_strategy_kpi_result"),
            "stakeholder_collaboration": result.get("stakeholder_collaboration_result"),
            "value_chain_optimization": result.get("value_chain_optimization_result"),
        }
        stage1_key = f"{evaluation_base_prefix}/stage1_evidence.json"
        stage1_evidence_url = self.s3_service.upload_json(stage1_key, stage1_payload)

        stage2_payload = {
            "segment_evaluations_with_resume": result.get("segment_evaluations_with_resume", []),
            "confidence_v2_calculated": result.get("confidence_v2_calculated", False),
            "segment_overlap_adjustments": result.get("segment_overlap_adjustments", []),
            "cross_competency_flags": result.get("cross_competency_flags", []),
            "aggregated_competencies": result.get("aggregated_competencies", {}),
            "low_confidence_list": result.get("low_confidence_list", []),
            "requires_collaboration": result.get("requires_collaboration", False),
            "competency_weights": competency_weights,
        }
        stage2_key = f"{evaluation_base_prefix}/stage2_aggregator.json"
        stage2_aggregator_url = self.s3_service.upload_json(stage2_key, stage2_payload)

        stage3_payload = {
            "final_score": result.get("final_score"),
            "avg_confidence": result.get("avg_confidence"),
            "final_reliability": result.get("final_reliability"),
            "reliability_note": result.get("reliability_note"),
            "final_result": result.get("final_result"),
            "aggregated_competencies": result.get("aggregated_competencies", {}),
            "collaboration_results": result.get("collaboration_results", []),
            "analysis_summary": result.get("analysis_summary"),
            "post_processing": result.get("post_processing"),
        }
        stage3_key = f"{evaluation_base_prefix}/stage3_final_integration.json"
        stage3_final_url = self.s3_service.upload_json(stage3_key, stage3_payload)

        #  Stage 4: Presentation 결과 S3 저장
        presentation_key = f"{evaluation_base_prefix}/stage4_presentation_frontend.json"
        presentation_s3_url = self.s3_service.upload_json(
            presentation_key,
            result.get("presentation_result", {})
        )

        # DB 저장
        db = SessionLocal()
        try:
            evaluation_record = await asyncio.to_thread(
                self._save_evaluation_to_db, 
                db, 
                result, 
                transcript_s3_url, 
                agent_logs_s3_url,
                stage1_evidence_url,
                stage2_aggregator_url,
                stage3_final_url,
                presentation_s3_url, 
                run_ts_str
            )
            evaluation_id = evaluation_record.id
        finally:
            await asyncio.to_thread(db.close)
        
        print("\n" + "="*80)
        print("평가 완료")
        print("="*80)
        
        # 최종 결과 구성
        return {
            "evaluation_id": evaluation_id,
            "interview_id": interview_id,
            "applicant_id": applicant_id,
            "job_id": job_id,
            "transcript_s3_url": transcript_s3_url,
            "agent_logs_s3_url": agent_logs_s3_url,
            "stage1_evidence_s3_url": stage1_evidence_url,
            "stage2_aggregator_s3_url": stage2_aggregator_url,
            "stage3_final_integration_s3_url": stage3_final_url,
            "stage4_presentation_s3_url": presentation_s3_url, 
            "evaluation_run_ts": run_ts_str,
            
            "execution_logs": result.get("execution_logs", []),
            "segment_evaluations_with_resume": result.get("segment_evaluations_with_resume", []),
            "segment_overlap_adjustments": result.get("segment_overlap_adjustments", []),
            "cross_competency_flags": result.get("cross_competency_flags", []),
            "aggregated_competencies": result.get("aggregated_competencies", {}),
            
            "final_score": result.get("final_score"),
            "avg_confidence": result.get("avg_confidence"),
            "final_reliability": result.get("final_reliability"),
            "reliability_note": result.get("reliability_note"),
            "analysis_summary": result.get("analysis_summary"),
            "post_processing": result.get("post_processing"),
            
            "competency_details": result.get("final_result", {}).get("competency_details", {}),
            "low_confidence_summary": result.get("final_result", {}).get("low_confidence_summary", {}),
            "collaboration_summary": result.get("final_result", {}).get("collaboration_summary", {}),
            
            # Presentation 결과 추가
            "presentation_result": result.get("presentation_result", {}),
            
            "started_at": result["started_at"].isoformat(),
            "completed_at": datetime.now().isoformat()
        }

    def _save_evaluation_to_db(
        self, 
        db: Session, 
        state: Dict, 
        transcript_s3_url: str, 
        agent_logs_s3_url: str,
        evidence_s3_url: str,
        stage2_aggregator_s3_url: str,
        stage3_final_s3_url: str,
        presentation_s3_url: str, 
        evaluation_run_ts: str
    ):
        """평가 결과를 DB에 저장"""
        
        applicant_id = state["applicant_id"]
        job_id = state["job_id"]

        applicant = db.query(Applicant).filter(Applicant.id == applicant_id).first()
        if not applicant:
            applicant = Applicant(
                id=applicant_id,
                name=f"Applicant {applicant_id}",
                email=f"applicant{applicant_id}@example.com"
            )
            db.add(applicant)

        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            company = db.query(Company).filter(Company.id == 1).first()
            if not company:
                company = Company(id=1, name="Default Company")
                db.add(company)
            job = Job(
                id=job_id,
                company_id=company.id,
                title=f"Job {job_id}",
                description="Placeholder job for testing"
            )
            db.add(job)

        interview_id = state.get("interview_id")
        if interview_id:
            interview = db.query(InterviewSession).filter(InterviewSession.id == interview_id).first()
            if not interview:
                interview = InterviewSession(
                    id=interview_id,
                    applicant_id=applicant_id,
                    company_id=job.company_id,
                    status=InterviewStatus.COMPLETED
                )
                db.add(interview)

        db.flush()

        aggregated_competencies = state.get("aggregated_competencies", {})
        final_result = state.get("final_result", {})
        analysis_summary = state.get("analysis_summary")
        post_processing = state.get("post_processing")
        aggregated_eval_payload = {
            "final_result": final_result,
            "analysis_summary": analysis_summary,
            "post_processing": post_processing
        }
        
        evaluation_metadata = {
            "resume_verified_count": sum(
                comp.get("resume_verified_count", 0)
                for comp in aggregated_competencies.values()
            ),
            "segment_overlap_adjustments": len(state.get("segment_overlap_adjustments", [])),
            "collaboration_count": state.get("collaboration_count", 0),
            "s3_paths": {
                "stage1_evidence": evidence_s3_url,
                "stage2_aggregator": stage2_aggregator_s3_url,
                "stage3_final_integration": stage3_final_s3_url,
                "stage4_presentation_frontend": presentation_s3_url,  
                "execution_logs": agent_logs_s3_url,
            },
            "evaluation_run_ts": evaluation_run_ts,
            "evaluation_prefix": f"evaluations/{state.get('interview_id')}/{evaluation_run_ts}"
        }
        
        evaluation_record = Evaluation(
            applicant_id=state["applicant_id"],
            job_id=state["job_id"],
            interview_id=state["interview_id"],
            evaluation_status="completed",
            
            match_score=state.get("final_score", 0.0),
            normalized_score=state.get("final_score", 0.0),
            weighted_score=state.get("final_score", 0.0),
            confidence_score=state.get("avg_confidence", 0.0),

            transcript_s3_url=transcript_s3_url,
            agent_logs_s3_url=agent_logs_s3_url,
            evidence_s3_url=evidence_s3_url,

            problem_solving=aggregated_competencies.get("problem_solving", {}).get("overall_score", 0),
            organizational_fit=aggregated_competencies.get("organizational_fit", {}).get("overall_score", 0),
            growth_potential=aggregated_competencies.get("growth_potential", {}).get("overall_score", 0),
            interpersonal_skills=aggregated_competencies.get("interpersonal_skill", {}).get("overall_score", 0),
            achievement_motivation=aggregated_competencies.get("achievement_motivation", {}).get("overall_score", 0),
            
            structured_thinking=aggregated_competencies.get("customer_journey_marketing", {}).get("overall_score", 0),
            business_documentation=aggregated_competencies.get("md_data_analysis", {}).get("overall_score", 0),
            financial_literacy=aggregated_competencies.get("seasonal_strategy_kpi", {}).get("overall_score", 0),
            industry_learning=aggregated_competencies.get("stakeholder_collaboration", {}).get("overall_score", 0),
            stakeholder_management=aggregated_competencies.get("value_chain_optimization", {}).get("overall_score", 0),

            job_aggregation=None,
            common_aggregation=None,
            
            validation_result={
                "low_confidence_list": state.get("low_confidence_list", []),
                "cross_competency_flags": state.get("cross_competency_flags", []),
            },

            competency_scores={
                comp_name: comp_data.get("overall_score", 0)
                for comp_name, comp_data in aggregated_competencies.items()
            },
            individual_evaluations=aggregated_competencies,
            aggregated_evaluation=aggregated_eval_payload,
            match_result={
                "final_score": state.get("final_score"),
                "reliability": state.get("final_reliability"),
                "reliability_note": state.get("reliability_note")
            },
            reasoning_log={},
            rubric_scores={},
            normalization_metadata={},
            evaluation_metadata=evaluation_metadata,
            
            created_at=state["started_at"],
            updated_at=datetime.now()
        )
        db.add(evaluation_record)
        db.commit()
        db.refresh(evaluation_record)
        return evaluation_record
