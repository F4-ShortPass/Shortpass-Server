"""
MAS (Multi-Agent System) 아키텍처 검증 테스트

검증 시나리오:
1. 일관성 테스트 (Consistency Test)
2. Stage별 검증 (Stage Validation)
3. Confidence 검증 (Confidence Validation)
4. RAG 효율성 (RAG Efficiency)
5. Cross-competency 일관성 (Cross-competency Consistency)
"""

import asyncio
import json
import statistics
import sys
from datetime import datetime
from typing import Dict, List
from pathlib import Path

# 상위 디렉토리를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import AsyncOpenAI
from sqlalchemy.orm import Session

from ai.agents.graph.evaluation import create_evaluation_graph
from ai.agents.graph.state import EvaluationState


class MASValidator:
    """MAS 아키텍처 검증 도구"""

    def __init__(self, openai_client: AsyncOpenAI, db_session: Session = None):
        self.client = openai_client
        self.db_session = db_session
        self.test_results = {
            "consistency_tests": [],
            "stage_validations": [],
            "confidence_tests": [],
            "rag_efficiency_tests": [],
            "cross_competency_tests": []
        }

    async def run_consistency_test(
        self,
        transcript: Dict,
        resume: Dict,
        weights: Dict,
        iterations: int = 3
    ) -> Dict:
        """
        1. 일관성 테스트

        같은 입력으로 여러 번 평가하여 결과 일관성 확인

        Args:
            transcript: 면접 transcript
            resume: 지원자 resume
            weights: 역량 가중치
            iterations: 반복 횟수 (기본 3회)

        Returns:
            일관성 테스트 결과
        """
        print("\n" + "="*80)
        print("[일관성 테스트] 같은 입력으로 여러 번 평가 실행")
        print("="*80)

        results = []

        for i in range(iterations):
            print(f"\n[반복 {i+1}/{iterations}] 평가 실행 중...")

            # 평가 그래프 실행
            result = await self._run_evaluation_graph(
                transcript=transcript,
                resume=resume,
                weights=weights,
                use_rag=False  # 일관성 테스트는 RAG 없이
            )

            results.append(result)

            print(f"[반복 {i+1}] 완료 - 최종 점수: {result['final_result']['final_score']}")

        # 일관성 분석
        consistency_analysis = self._analyze_consistency(results)

        print("\n" + "="*80)
        print("[일관성 분석 결과]")
        print("="*80)
        print(f"최종 점수 평균: {consistency_analysis['final_score_mean']:.2f}")
        print(f"최종 점수 표준편차: {consistency_analysis['final_score_std']:.2f}")
        print(f"최종 점수 범위: {consistency_analysis['final_score_min']:.2f} ~ {consistency_analysis['final_score_max']:.2f}")
        print("\n[역량별 점수 일관성]")
        for comp, stats in consistency_analysis['competency_scores'].items():
            print(f"  {comp}: {stats['mean']:.2f} (±{stats['std']:.2f})")

        # 허용 범위 체크 (±5점)
        consistency_pass = consistency_analysis['final_score_std'] <= 5.0
        print(f"\n일관성 검증: {'✅ PASS' if consistency_pass else '❌ FAIL'} (허용 범위: ±5점)")

        # JSON serializable한 결과만 저장
        serializable_results = []
        for r in results:
            # OpenAI client 제외하고 복사
            serializable_r = {k: v for k, v in r.items() if k != "openai_client"}
            serializable_results.append(serializable_r)

        test_result = {
            "test_type": "consistency",
            "iterations": iterations,
            "results": serializable_results,
            "analysis": consistency_analysis,
            "passed": consistency_pass,
            "timestamp": datetime.now().isoformat()
        }

        self.test_results["consistency_tests"].append(test_result)

        return test_result

    async def run_stage_validation(
        self,
        transcript: Dict,
        resume: Dict,
        weights: Dict
    ) -> Dict:
        """
        2. Stage별 검증

        각 Stage의 출력을 검증하여 정상 작동 확인

        Returns:
            Stage별 검증 결과
        """
        print("\n" + "="*80)
        print("[Stage별 검증] 각 Stage 출력 검증")
        print("="*80)

        result = await self._run_evaluation_graph(
            transcript=transcript,
            resume=resume,
            weights=weights,
            use_rag=True
        )

        validation_results = {}

        # Stage 1 검증: 10개 역량 모두 평가되었는지
        stage1_competencies = [
            "achievement_motivation", "growth_potential", "interpersonal_skill",
            "organizational_fit", "problem_solving",
            "customer_journey_marketing", "md_data_analysis", "seasonal_strategy_kpi",
            "stakeholder_collaboration", "value_chain_optimization"
        ]

        evaluated_competencies = list(result.get("aggregated_competencies", {}).keys())
        stage1_pass = len(evaluated_competencies) == len(stage1_competencies)

        validation_results["stage_1"] = {
            "expected_competencies": len(stage1_competencies),
            "evaluated_competencies": len(evaluated_competencies),
            "missing": list(set(stage1_competencies) - set(evaluated_competencies)),
            "passed": stage1_pass
        }

        print(f"\n[Stage 1 검증]")
        print(f"  예상 역량 수: {len(stage1_competencies)}개")
        print(f"  평가 완료 역량: {len(evaluated_competencies)}개")
        print(f"  결과: {'✅ PASS' if stage1_pass else '❌ FAIL'}")
        if not stage1_pass:
            print(f"  누락 역량: {validation_results['stage_1']['missing']}")

        # Stage 2 검증: Resume 검증, Confidence V2, Overlap 조정
        stage2_checks = {
            "segment_evaluations_with_resume": "segment_evaluations_with_resume" in result,
            "confidence_v2_calculated": result.get("confidence_v2_calculated", False),
            "aggregated_competencies": "aggregated_competencies" in result,
            "low_confidence_detected": "low_confidence_list" in result
        }

        stage2_pass = all(stage2_checks.values())

        validation_results["stage_2"] = {
            "checks": stage2_checks,
            "passed": stage2_pass
        }

        print(f"\n[Stage 2 검증]")
        for check, passed in stage2_checks.items():
            print(f"  {check}: {'✅' if passed else '❌'}")
        print(f"  결과: {'✅ PASS' if stage2_pass else '❌ FAIL'}")

        # Stage 3 검증: Final Integration
        stage3_checks = {
            "final_result": "final_result" in result,
            "final_score": result.get("final_result", {}).get("final_score") is not None,
            "avg_confidence": result.get("avg_confidence") is not None,
            "reliability": "reliability" in result.get("final_result", {})
        }

        stage3_pass = all(stage3_checks.values())

        validation_results["stage_3"] = {
            "checks": stage3_checks,
            "passed": stage3_pass
        }

        print(f"\n[Stage 3 검증]")
        for check, passed in stage3_checks.items():
            print(f"  {check}: {'✅' if passed else '❌'}")
        print(f"  결과: {'✅ PASS' if stage3_pass else '❌ FAIL'}")

        # Stage 4 검증: Presentation Format
        stage4_checks = {
            "presentation_format": "presentation_result" in result or "presentation_frontend" in result
        }

        stage4_pass = all(stage4_checks.values())

        validation_results["stage_4"] = {
            "checks": stage4_checks,
            "passed": stage4_pass
        }

        print(f"\n[Stage 4 검증]")
        for check, passed in stage4_checks.items():
            print(f"  {check}: {'✅' if passed else '❌'}")
        print(f"  결과: {'✅ PASS' if stage4_pass else '❌ FAIL'}")

        # 전체 검증 결과
        all_passed = all([
            validation_results["stage_1"]["passed"],
            validation_results["stage_2"]["passed"],
            validation_results["stage_3"]["passed"],
            validation_results["stage_4"]["passed"]
        ])

        print(f"\n전체 Stage 검증: {'✅ PASS' if all_passed else '❌ FAIL'}")

        test_result = {
            "test_type": "stage_validation",
            "validations": validation_results,
            "passed": all_passed,
            "timestamp": datetime.now().isoformat()
        }

        self.test_results["stage_validations"].append(test_result)

        return test_result

    async def run_confidence_validation(
        self,
        transcript: Dict,
        resume: Dict,
        weights: Dict
    ) -> Dict:
        """
        3. Confidence 검증

        Low confidence 케이스 처리 확인

        Returns:
            Confidence 검증 결과
        """
        print("\n" + "="*80)
        print("[Confidence 검증] Low confidence 케이스 처리")
        print("="*80)

        result = await self._run_evaluation_graph(
            transcript=transcript,
            resume=resume,
            weights=weights,
            use_rag=True
        )

        aggregated_competencies = result.get("aggregated_competencies", {})
        low_confidence_list = result.get("low_confidence_list", [])

        # Confidence 분포 분석
        confidence_v2_scores = [
            comp_data.get("confidence_v2", 0)
            for comp_data in aggregated_competencies.values()
        ]

        confidence_stats = {
            "mean": statistics.mean(confidence_v2_scores) if confidence_v2_scores else 0,
            "median": statistics.median(confidence_v2_scores) if confidence_v2_scores else 0,
            "min": min(confidence_v2_scores) if confidence_v2_scores else 0,
            "max": max(confidence_v2_scores) if confidence_v2_scores else 0,
            "std": statistics.stdev(confidence_v2_scores) if len(confidence_v2_scores) > 1 else 0
        }

        print(f"\n[Confidence V2 분포]")
        print(f"  평균: {confidence_stats['mean']:.3f}")
        print(f"  중앙값: {confidence_stats['median']:.3f}")
        print(f"  범위: {confidence_stats['min']:.3f} ~ {confidence_stats['max']:.3f}")
        print(f"  표준편차: {confidence_stats['std']:.3f}")

        print(f"\n[Low Confidence 역량]")
        print(f"  탐지 개수: {len(low_confidence_list)}개")

        if low_confidence_list:
            for item in low_confidence_list:
                print(f"    - {item['competency']}: {item['confidence_v2']:.3f} (점수: {item['score']})")
        else:
            print("  없음")

        # Collaboration 트리거 확인
        requires_collaboration = result.get("requires_collaboration", False)

        print(f"\n[Collaboration 트리거]")
        print(f"  requires_collaboration: {requires_collaboration}")

        # 검증: Low confidence가 있으면 requires_collaboration이 True여야 함
        validation_pass = (len(low_confidence_list) > 0) == requires_collaboration

        print(f"\n검증 결과: {'✅ PASS' if validation_pass else '❌ FAIL'}")

        test_result = {
            "test_type": "confidence_validation",
            "confidence_stats": confidence_stats,
            "low_confidence_count": len(low_confidence_list),
            "low_confidence_list": low_confidence_list,
            "requires_collaboration": requires_collaboration,
            "passed": validation_pass,
            "timestamp": datetime.now().isoformat()
        }

        self.test_results["confidence_tests"].append(test_result)

        return test_result

    async def run_rag_efficiency_test(
        self,
        transcript: Dict,
        resume: Dict,
        weights: Dict
    ) -> Dict:
        """
        4. RAG 효율성 테스트

        RAG 사용 시 segment 필터링 효과 측정

        Returns:
            RAG 효율성 테스트 결과
        """
        print("\n" + "="*80)
        print("[RAG 효율성 테스트] RAG 사용 시 토큰 절감 효과")
        print("="*80)

        # RAG 없이 평가
        print("\n[1] RAG 없이 평가 실행...")
        result_without_rag = await self._run_evaluation_graph(
            transcript=transcript,
            resume=resume,
            weights=weights,
            use_rag=False
        )

        # RAG 사용하여 평가
        print("\n[2] RAG 사용하여 평가 실행...")
        result_with_rag = await self._run_evaluation_graph(
            transcript=transcript,
            resume=resume,
            weights=weights,
            use_rag=True
        )

        # 토큰 절감 효과 분석 (추정)
        # segment 필터링률로 토큰 절감률 추정
        total_segments = len(transcript.get("segments", []))

        # RAG 메타데이터가 있는 역량에서 필터링률 추출
        rag_metadata_list = []

        for comp_name, comp_data in result_with_rag.get("aggregated_competencies", {}).items():
            perspectives = comp_data.get("perspectives", {})
            evidence_details = perspectives.get("evidence_details", [])

            for ev in evidence_details:
                if "rag_metadata" in ev:
                    rag_metadata_list.append(ev["rag_metadata"])

        if rag_metadata_list:
            avg_token_reduction = statistics.mean([
                meta.get("token_reduction_rate", 0)
                for meta in rag_metadata_list
            ])

            avg_filtered_segment_count = statistics.mean([
                meta.get("filtered_segment_count", 0)
                for meta in rag_metadata_list
            ])
        else:
            avg_token_reduction = 0
            avg_filtered_segment_count = 0

        print(f"\n[RAG 효율성 분석]")
        print(f"  전체 segment 수: {total_segments}개")
        print(f"  RAG 필터링 후 평균 segment 수: {avg_filtered_segment_count:.1f}개")
        print(f"  평균 토큰 절감률: {avg_token_reduction*100:.1f}%")

        # 점수 비교
        score_without_rag = result_without_rag["final_result"]["final_score"]
        score_with_rag = result_with_rag["final_result"]["final_score"]
        score_diff = abs(score_without_rag - score_with_rag)

        print(f"\n[점수 비교]")
        print(f"  RAG 없음: {score_without_rag:.2f}")
        print(f"  RAG 사용: {score_with_rag:.2f}")
        print(f"  점수 차이: {score_diff:.2f}")

        # 검증: 점수 차이가 10점 이내여야 함 (RAG가 평가 품질을 크게 떨어뜨리지 않아야 함)
        efficiency_pass = score_diff <= 10.0 and avg_token_reduction > 0

        print(f"\n검증 결과: {'✅ PASS' if efficiency_pass else '❌ FAIL'} (점수 차이 ≤ 10점, 토큰 절감 > 0)")

        test_result = {
            "test_type": "rag_efficiency",
            "total_segments": total_segments,
            "avg_filtered_segments": avg_filtered_segment_count,
            "avg_token_reduction_rate": avg_token_reduction,
            "score_without_rag": score_without_rag,
            "score_with_rag": score_with_rag,
            "score_diff": score_diff,
            "passed": efficiency_pass,
            "timestamp": datetime.now().isoformat()
        }

        self.test_results["rag_efficiency_tests"].append(test_result)

        return test_result

    async def run_cross_competency_test(
        self,
        transcript: Dict,
        resume: Dict,
        weights: Dict
    ) -> Dict:
        """
        5. Cross-competency 일관성 테스트

        관련 역량 간 점수 상관관계 확인

        Returns:
            Cross-competency 테스트 결과
        """
        print("\n" + "="*80)
        print("[Cross-competency 일관성 테스트] 역량 간 상관관계")
        print("="*80)

        result = await self._run_evaluation_graph(
            transcript=transcript,
            resume=resume,
            weights=weights,
            use_rag=True
        )

        aggregated_competencies = result.get("aggregated_competencies", {})

        # 역량 간 예상 상관관계 정의
        expected_correlations = [
            ("problem_solving", "md_data_analysis", "데이터 기반 문제해결"),
            ("interpersonal_skill", "stakeholder_collaboration", "협업 및 대인관계"),
            ("growth_potential", "achievement_motivation", "성장과 성취 동기"),
        ]

        correlation_results = []

        print(f"\n[역량 간 상관관계 분석]")

        for comp1, comp2, description in expected_correlations:
            if comp1 in aggregated_competencies and comp2 in aggregated_competencies:
                score1 = aggregated_competencies[comp1]["overall_score"]
                score2 = aggregated_competencies[comp2]["overall_score"]

                diff = abs(score1 - score2)

                # 관련 역량은 점수 차이가 20점 이내여야 함
                correlated = diff <= 20

                print(f"\n  {description}")
                print(f"    - {comp1}: {score1}")
                print(f"    - {comp2}: {score2}")
                print(f"    - 점수 차이: {diff:.2f} ({'✅ 일관성 있음' if correlated else '⚠️ 차이가 큼'})")

                correlation_results.append({
                    "competency_1": comp1,
                    "competency_2": comp2,
                    "description": description,
                    "score_1": score1,
                    "score_2": score2,
                    "diff": diff,
                    "correlated": correlated
                })

        # 검증: 모든 관련 역량 쌍이 일관성을 가져야 함
        consistency_pass = all(r["correlated"] for r in correlation_results)

        print(f"\n검증 결과: {'✅ PASS' if consistency_pass else '⚠️ PARTIAL'}")

        test_result = {
            "test_type": "cross_competency",
            "correlation_results": correlation_results,
            "passed": consistency_pass,
            "timestamp": datetime.now().isoformat()
        }

        self.test_results["cross_competency_tests"].append(test_result)

        return test_result

    async def run_all_tests(
        self,
        transcript: Dict,
        resume: Dict,
        weights: Dict
    ) -> Dict:
        """
        모든 검증 테스트 실행

        Returns:
            전체 테스트 결과
        """
        print("\n" + "="*80)
        print("MAS 아키텍처 검증 테스트 시작")
        print("="*80)

        start_time = datetime.now()

        # 1. 일관성 테스트 (3회 반복)
        consistency_result = await self.run_consistency_test(
            transcript=transcript,
            resume=resume,
            weights=weights,
            iterations=3
        )

        # 2. Stage별 검증
        stage_validation_result = await self.run_stage_validation(
            transcript=transcript,
            resume=resume,
            weights=weights
        )

        # 3. Confidence 검증
        confidence_result = await self.run_confidence_validation(
            transcript=transcript,
            resume=resume,
            weights=weights
        )

        # 4. RAG 효율성 테스트
        rag_efficiency_result = await self.run_rag_efficiency_test(
            transcript=transcript,
            resume=resume,
            weights=weights
        )

        # 5. Cross-competency 테스트
        cross_competency_result = await self.run_cross_competency_test(
            transcript=transcript,
            resume=resume,
            weights=weights
        )

        duration = (datetime.now() - start_time).total_seconds()

        # 전체 결과 요약
        all_passed = all([
            consistency_result["passed"],
            stage_validation_result["passed"],
            confidence_result["passed"],
            rag_efficiency_result["passed"],
            cross_competency_result["passed"]
        ])

        summary = {
            "total_tests": 5,
            "passed_tests": sum([
                consistency_result["passed"],
                stage_validation_result["passed"],
                confidence_result["passed"],
                rag_efficiency_result["passed"],
                cross_competency_result["passed"]
            ]),
            "all_passed": all_passed,
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat()
        }

        print("\n" + "="*80)
        print("MAS 아키텍처 검증 테스트 완료")
        print("="*80)
        print(f"\n전체 테스트: {summary['passed_tests']}/{summary['total_tests']} PASS")
        print(f"소요 시간: {duration:.2f}초")
        print(f"\n최종 결과: {'✅ 모든 테스트 통과' if all_passed else '❌ 일부 테스트 실패'}")

        return {
            "summary": summary,
            "test_results": self.test_results,
            "consistency": consistency_result,
            "stage_validation": stage_validation_result,
            "confidence": confidence_result,
            "rag_efficiency": rag_efficiency_result,
            "cross_competency": cross_competency_result
        }

    async def _run_evaluation_graph(
        self,
        transcript: Dict,
        resume: Dict,
        weights: Dict,
        use_rag: bool = False
    ) -> Dict:
        """
        평가 그래프 실행 (내부 헬퍼)

        Returns:
            평가 결과 State
        """
        # 프롬프트 생성
        prompts = self._load_prompts(transcript)

        # 평가 그래프 생성
        graph = create_evaluation_graph()

        # 초기 State 구성
        initial_state = EvaluationState(
            interview_id=101,
            applicant_id=101,
            job_id=1,
            transcript_s3_url="",
            transcript=transcript,
            transcript_content=transcript,  # 추가
            resume_data=resume,
            competency_weights=weights,
            openai_client=self.client,
            prompts=prompts,  # 추가
            use_rag=use_rag,
            rag_top_k=8 if use_rag else None,
            session_id=101 if use_rag and self.db_session else None,  # RAG 사용 시 session_id 필요
            job_competencies=None,  # 기본 직무 역량 사용
            started_at=datetime.now(),
            execution_logs=[]
        )

        # 그래프 실행
        final_state = await graph.ainvoke(initial_state)

        return final_state

    def _load_prompts(self, transcript: Dict) -> Dict[str, str]:
        """프롬프트 로딩"""
        import json
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

        transcript_str = json.dumps(transcript, ensure_ascii=False, indent=2)
        return {
            name: generator(transcript_str)
            for name, generator in PROMPT_GENERATORS.items()
        }

    def _analyze_consistency(self, results: List[Dict]) -> Dict:
        """
        일관성 분석 (내부 헬퍼)

        Args:
            results: 여러 번 평가한 결과 목록

        Returns:
            일관성 분석 결과
        """
        # 최종 점수 통계
        final_scores = [r["final_result"]["final_score"] for r in results]

        final_score_stats = {
            "mean": statistics.mean(final_scores),
            "std": statistics.stdev(final_scores) if len(final_scores) > 1 else 0,
            "min": min(final_scores),
            "max": max(final_scores)
        }

        # 역량별 점수 통계
        competency_scores = {}

        # 첫 번째 결과에서 역량 목록 추출
        first_result = results[0]
        competencies = list(first_result.get("aggregated_competencies", {}).keys())

        for comp in competencies:
            scores = [
                r.get("aggregated_competencies", {}).get(comp, {}).get("overall_score", 0)
                for r in results
            ]

            if scores:
                competency_scores[comp] = {
                    "mean": statistics.mean(scores),
                    "std": statistics.stdev(scores) if len(scores) > 1 else 0,
                    "min": min(scores),
                    "max": max(scores)
                }

        return {
            "final_score_mean": final_score_stats["mean"],
            "final_score_std": final_score_stats["std"],
            "final_score_min": final_score_stats["min"],
            "final_score_max": final_score_stats["max"],
            "competency_scores": competency_scores
        }

    def save_test_results(self, output_path: str):
        """
        테스트 결과를 JSON 파일로 저장

        Args:
            output_path: 저장 경로
        """
        # JSON serializable하게 변환
        def make_serializable(obj):
            """객체를 JSON serializable하게 변환"""
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items() if k != "openai_client"}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            else:
                return obj

        serializable_results = make_serializable(self.test_results)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2, default=str)

        print(f"\n테스트 결과 저장: {output_path}")


async def main():
    """
    메인 실행 함수
    """
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # OpenAI 클라이언트 생성
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 테스트 데이터 로드
    test_data_dir = Path(__file__).parent.parent / "test_data"

    with open(test_data_dir / "transcript_jiwon_101.json", "r", encoding="utf-8") as f:
        transcript = json.load(f)

    with open(test_data_dir / "resume_jiwon.json", "r", encoding="utf-8") as f:
        resume = json.load(f)

    # weights는 transcript에서 추출
    weights = transcript.get("weights", {})

    # Validator 생성
    validator = MASValidator(openai_client=openai_client)

    # 모든 테스트 실행
    test_results = await validator.run_all_tests(
        transcript=transcript,
        resume=resume,
        weights=weights
    )

    # 결과 저장
    output_path = test_data_dir / f"mas_validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    validator.save_test_results(str(output_path))

    print(f"\n전체 테스트 완료!")
    print(f"결과 파일: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
