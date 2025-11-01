"""
전체 평가 시스템 테스트
Phase 1-4 전체 플로우 + 상세 출력
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
from services.evaluation.evaluation_service import EvaluationService


async def test_full_evaluation():
    """전체 평가 시스템 테스트 (Phase 1-4)"""
    
    # ============================================
    # 테스트 데이터
    # ============================================
    
    transcript = {
        "metadata": {
            "interview_id": 1,
            "applicant_id": 100,
            "job_id": 200,
            "duration_sec": 600
        },
        "segments": [
            {
                "segment_id": 1,
                "segment_order": 1,
                "question_text": "자기소개를 부탁드립니다.",
                "answer_text": "안녕하세요. 저는 데이터 분석에 관심이 많은 지원자입니다. 대학에서 통계학을 전공하면서 파이썬과 R을 활용한 데이터 분석 프로젝트를 다수 진행했습니다. 특히 비즈니스 문제를 구조화하여 접근하는 것을 좋아하는데, 예를 들어 시장 진입 전략을 수립할 때 문제를 수요, 공급, 가격 세 축으로 나눠서 분석하는 MECE 방식을 선호합니다. 또한 팀 프로젝트에서 리더 역할을 맡아 이해관계자들 간의 의견을 조율하고 합의점을 도출한 경험이 있습니다.",
                "answer_duration_sec": 60,
                "char_index_start": 0,
                "char_index_end": 280
            },
            {
                "segment_id": 2,
                "segment_order": 2,
                "question_text": "가장 도전적이었던 프로젝트는 무엇인가요?",
                "answer_text": "작년 여름 인턴십에서 6개월치 디지털 마케팅 데이터를 분석한 프로젝트가 가장 도전적이었습니다. 데이터가 Excel 파일 15개에 흩어져 있었고, 각 파일의 형식도 달라서 먼저 데이터 정합성을 확인하고 피벗 테이블과 VLOOKUP으로 통합했습니다. 그 다음 각 캠페인별로 ROI를 계산했는데, 투입 비용 대비 매출 증가를 추적한 결과 평균 ROI가 약 150%였습니다. 이 분석을 바탕으로 고ROI 캠페인에 예산을 집중 배분하자는 제안서를 작성했고, 실제로 다음 분기에 마케팅 효율이 20% 개선되었습니다. 이 경험을 통해 데이터 전처리의 중요성과 재무 지표를 활용한 의사결정 방법을 배웠습니다.",
                "answer_duration_sec": 90,
                "char_index_start": 0,
                "char_index_end": 420
            },
            {
                "segment_id": 3,
                "segment_order": 3,
                "question_text": "팀에서 갈등을 해결한 경험이 있나요?",
                "answer_text": "네, 졸업 프로젝트에서 팀원 간 의견이 크게 갈렸던 적이 있습니다. 한 팀원은 기술적 완성도를 중시했고, 다른 팀원은 시장성을 우선시했습니다. 저는 먼저 각자의 의견을 표로 정리해서 기술 완성도와 시장성의 장단점을 객관적으로 비교 분석했습니다. 그리고 교수님과 업계 전문가 2분께 조언을 구해서 제3자의 관점도 추가했습니다. 결과적으로 MVP 방식으로 핵심 기능만 먼저 완성하고 시장 반응을 보며 개선하는 절충안을 제시했고, 양측 모두 동의했습니다. 이 과정에서 이해관계자 간 소통과 데이터 기반 의사결정의 중요성을 깨달았습니다.",
                "answer_duration_sec": 80,
                "char_index_start": 0,
                "char_index_end": 380
            },
            {
                "segment_id": 4,
                "segment_order": 4,
                "question_text": "실패한 경험과 그로부터 배운 점을 말씀해주세요.",
                "answer_text": "작년 데이터 분석 공모전에서 본선 탈락한 경험이 있습니다. 당시 정말 힘들었지만, 심사위원 피드백을 받아보니 데이터 수집과 전처리는 잘했지만 비즈니스 인사이트 도출이 부족했다는 평가였습니다. 저는 이 피드백을 진지하게 받아들이고, 제 강점은 데이터 처리 능력이지만 약점은 전략적 사고라는 것을 인정했습니다. 이후 3개월간 하버드 비즈니스 리뷰 케이스 스터디를 매주 2개씩 분석하고, 컨설팅 회사의 보고서를 읽으며 프레임워크를 학습했습니다. 결과적으로 다음 공모전에서는 장려상을 받았고, 무엇보다 실패를 성장의 기회로 만드는 회복탄력성을 얻었습니다.",
                "answer_duration_sec": 90,
                "char_index_start": 0,
                "char_index_end": 410
            },
            {
                "segment_id": 5,
                "segment_order": 5,
                "question_text": "시장 분석을 어떻게 접근하시나요?",
                "answer_text": "시장 분석을 할 때는 항상 구조화된 접근을 선호합니다. 먼저 시장을 수요, 공급, 가격이라는 세 축으로 나눠서 분석합니다. 수요 측면에서는 고객 세그먼트별 니즈와 구매 패턴을 파악하고, 설문조사나 인터뷰 데이터를 활용합니다. 공급 측면에서는 주요 경쟁사 3-5개를 선정해서 시장 점유율, 가격 전략, 제품 포트폴리오를 비교 분석합니다. 가격 측면에서는 가격 탄력성을 고려하여 최적 가격대를 도출하고, 수익성 시뮬레이션을 엑셀로 작성합니다. 예를 들어 인턴십에서 신제품 출시 전략을 수립할 때 이 프레임워크를 활용해서 시장 진입 타이밍과 가격대를 제안했고, 실제로 채택되었습니다.",
                "answer_duration_sec": 90,
                "char_index_start": 0,
                "char_index_end": 420
            },
            {
                "segment_id": 6,
                "segment_order": 6,
                "question_text": "문서 작성 능력은 어느 정도인가요?",
                "answer_text": "저는 명확하고 논리적인 문서 작성을 중요하게 생각합니다. 인턴십에서 월간 마케팅 성과 보고서를 작성했는데, 경영진이 3분 안에 핵심을 파악할 수 있도록 Executive Summary를 먼저 작성하고, 그 다음 상세 분석을 첨부하는 구조를 사용했습니다. 또한 숫자만 나열하지 않고 차트와 그래프를 활용해서 시각적으로 이해하기 쉽게 만들었습니다. 예를 들어 캠페인별 ROI를 막대 그래프로 표현하고, 트렌드는 선 그래프로 보여줬습니다. 상사로부터 '엑셀 데이터를 스토리로 만드는 능력이 뛰어나다'는 피드백을 받았고, 제가 작성한 보고서 양식이 팀 표준 템플릿으로 채택되기도 했습니다.",
                "answer_duration_sec": 85,
                "char_index_start": 0,
                "char_index_end": 400
            },
            {
                "segment_id": 7,
                "segment_order": 7,
                "question_text": "새로운 산업 지식을 빠르게 학습한 경험이 있나요?",
                "answer_text": "네, 제가 전공한 통계학과는 전혀 다른 헬스케어 산업에 대해 빠르게 학습한 경험이 있습니다. 작년 겨울방학에 헬스케어 스타트업 인턴 기회를 얻었는데, 의료 용어나 규제에 대한 지식이 전무했습니다. 첫 2주간 저는 하루 2시간씩 식약처 가이드라인과 업계 리포트를 읽었고, 현업 직원들에게 점심시간마다 질문하며 핵심 개념을 정리했습니다. 또한 경쟁사 분석을 위해 주요 헬스케어 기업 10곳의 IR 자료를 읽고 비즈니스 모델을 비교했습니다. 3주 차부터는 실제 데이터 분석 프로젝트에 투입될 수 있었고, 인턴 종료 시점에는 헬스케어 시장 동향 보고서를 독자적으로 작성할 수 있었습니다. 이 경험을 통해 새로운 도메인을 빠르게 습득하는 나만의 학습 방법론을 확립했습니다.",
                "answer_duration_sec": 95,
                "char_index_start": 0,
                "char_index_end": 450
            },
            {
                "segment_id": 8,
                "segment_order": 8,
                "question_text": "우리 회사에 지원한 이유는 무엇인가요?",
                "answer_text": "귀사가 전략 컨설팅 업계에서 디지털 트랜스포메이션 프로젝트에 강점이 있다는 점에 매력을 느꼈습니다. 저는 데이터 분석 역량과 비즈니스 문제 해결에 대한 열정을 갖추고 있어서, 귀사의 프로젝트에서 크게 기여할 수 있다고 생각합니다. 특히 최근 귀사가 발표한 제조업 스마트 팩토리 구축 사례를 읽고 깊은 인상을 받았습니다. 저도 데이터 기반으로 클라이언트의 의사결정을 지원하고, 다양한 산업을 경험하며 성장하고 싶습니다. 또한 귀사의 인재 육성 프로그램과 글로벌 프로젝트 기회가 제 장기 커리어 목표와 잘 맞는다고 판단했습니다. 입사 후에는 우선 데이터 분석 역량을 활용해 팀에 기여하고, 장기적으로는 전략 기획과 프로젝트 매니지먼트 역량까지 갖춘 시니어 컨설턴트로 성장하고 싶습니다.",
                "answer_duration_sec": 90,
                "char_index_start": 0,
                "char_index_end": 450
            }
        ]
    }
    
    job_weights = {
        "structured_thinking": 0.25,
        "business_documentation": 0.20,
        "financial_literacy": 0.20,
        "industry_learning": 0.20,
        "stakeholder_management": 0.15
    }
    
    common_weights = {
        "problem_solving": 0.25,
        "organizational_fit": 0.20,
        "growth_potential": 0.20,
        "interpersonal_skills": 0.20,
        "achievement_motivation": 0.15
    }
    
     # ============================================
    # 평가 실행
    # ============================================
    
    print("\n" + "="*80)
    print(" AI 면접 평가 시스템 - 전체 테스트 (Phase 1-4)")
    print("="*80 + "\n")
    
    # 서비스 초기화
    service = EvaluationService()
    
    print(" 평가 시작...\n")
    
    try:
        # 평가 실행
        result = await service.evaluate_interview(
            interview_id=1,
            applicant_id=100,
            job_id=200,
            transcript=transcript,
            job_weights=job_weights,
            common_weights=common_weights
        )
        
        print("\n 평가 완료!\n")
        
        # ============================================
        # Phase 1 결과 출력
        # ============================================
        
        print("="*80)
        print(" Phase 1: 개별 역량 평가 결과")
        print("="*80)
        
        # Job 역량
        print("\n" + "-"*80)
        print(" Job 역량 (5개)")
        print("-"*80)
        
        job_agg = result["job_aggregation"]
        print(f"\n 종합 점수: {job_agg['overall_job_score']:.1f}점\n")
        
        for comp_name in ["structured_thinking", "business_documentation",
                          "financial_literacy", "industry_learning", "stakeholder_management"]:
            comp_data = job_agg.get(comp_name)
            if comp_data:
                display_name = comp_data.get('competency_display_name', comp_name)
                score = comp_data.get('overall_score', 0)
                confidence = comp_data.get('confidence', {})
                conf_overall = confidence.get('overall_confidence', 0)
                
                print(f"  ├─ [{display_name}]: {score}점 (신뢰도: {conf_overall:.2f})")
        
        # Common 역량
        print("\n" + "-"*80)
        print(" Common 역량 (5개)")
        print("-"*80)
        
        common_agg = result["common_aggregation"]
        print(f"\n 종합 점수: {common_agg['overall_common_score']:.1f}점\n")
        
        for comp_name in ["problem_solving", "organizational_fit",
                          "growth_potential", "interpersonal_skills", "achievement_motivation"]:
            comp_data = common_agg.get(comp_name)
            if comp_data:
                display_name = comp_data.get('competency_display_name', comp_name)
                score = comp_data.get('overall_score', 0)
                confidence = comp_data.get('confidence', {})
                conf_overall = confidence.get('overall_confidence', 0)
                
                print(f"  ├─ [{display_name}]: {score}점 (신뢰도: {conf_overall:.2f})")
        
        # ============================================
        # Phase 2 결과 출력
        # ============================================
        
        print("\n" + "="*80)
        print(" Phase 2: 문제 탐지 결과")
        print("="*80)
        
        issues = result.get("issues_detected", {})
        
        conflicts = issues.get("evidence_conflicts", [])
        low_conf_list = issues.get("low_confidence_list", [])
        requires_collab = issues.get("requires_collaboration", False)
        
        print(f"\n  Evidence 충돌: {len(conflicts)}건")
        if conflicts:
            for conf in conflicts:
                seg_id = conf.get("segment_id")
                comps = conf.get("competencies", [])
                gap = conf.get("gap", 0)
                print(f"  ├─ Segment {seg_id}: {', '.join(comps)} (gap: {gap:.2f})")
        
        print(f"\n  Low Confidence: {len(low_conf_list)}개")
        if low_conf_list:
            for issue in low_conf_list:
                comp = issue.get("competency")
                conf = issue.get("overall_confidence", 0)
                reason = issue.get("reason", "")
                print(f"  ├─ {comp}: {conf:.2f} (원인: {reason})")
        
        print(f"\n 협업 필요 여부: {'YES (Phase 3 실행)' if requires_collab else 'NO (Phase 4로 바로 진행)'}")
        
        # ============================================
        # Phase 3 결과 출력
        # ============================================
        
        collaboration = result.get("collaboration", {})
        mediation_results = collaboration.get("mediation_results", [])
        adversarial_results = collaboration.get("adversarial_results", [])
        collab_count = collaboration.get("collaboration_count", 0)
        
        if requires_collab and collab_count > 0:
            print("\n" + "="*80)
            print(" Phase 3: 협업 처리 결과")
            print("="*80)
            
            print(f"\n Evidence 중재: {len(mediation_results)}건")
            if mediation_results:
                for med in mediation_results:
                    seg_id = med.get("segment_id")
                    primary = med.get("primary_competency")
                    print(f"  ├─ Segment {seg_id}: Primary={primary}")
            
            print(f"\n Adversarial 재평가: {len(adversarial_results)}개")
            if adversarial_results:
                for adv in adversarial_results:
                    comp = adv.get("competency")
                    orig = adv.get("original_score", 0)
                    adj = adv.get("adjusted_score", 0)
                    print(f"  ├─ {comp}: {orig}점 → {adj}점")
        
        # ============================================
        # Phase 4 결과 출력
        # ============================================
        
        print("\n" + "="*80)
        print(" Phase 4: 최종 통합 결과")
        print("="*80)
        
        final_score = result.get("final_score", 0)
        avg_confidence = result.get("avg_confidence", 0)
        final_reliability = result.get("final_reliability", "")
        reliability_note = result.get("reliability_note", "")
        
        print(f"\n 최종 점수: {final_score:.1f}점")
        print(f" 평균 Confidence: {avg_confidence:.2f}")
        print(f" 신뢰도 레벨: {final_reliability}")
        print(f" 신뢰도 설명: {reliability_note}")
        
        final_result = result.get("final_result", {})
        if final_result:
            job_score = final_result.get("job_score", 0)
            common_score = final_result.get("common_score", 0)
            ratio = final_result.get("job_common_ratio", {"job": 0.6, "common": 0.4})
            
            print(f"\n 세부 점수:")
            print(f"  ├─ Job 점수: {job_score:.1f}점 (가중치 {ratio['job']*100:.0f}%)")
            print(f"  └─ Common 점수: {common_score:.1f}점 (가중치 {ratio['common']*100:.0f}%)")
        
        # ============================================
        # 상세 역량별 출력 (선택)
        # ============================================
        
        print("\n" + "="*80)
        print(" 상세 역량별 평가 (샘플: problem_solving - Common)")
        print("="*80)
        
        st_data = common_agg.get("problem_solving")
        if st_data:
            print(f"\n역량명: {st_data.get('competency_display_name', 'N/A')}")
            print(f"카테고리: {st_data.get('competency_category', 'N/A')}")
            print(f"평가 시각: {st_data.get('evaluated_at', 'N/A')}")
            print(f"최종 점수: {st_data.get('overall_score', 0)}점")

            # Perspectives 상세 출력
            perspectives = st_data.get('perspectives', {})
            if perspectives:
                print(f"\n" + "-"*60)
                print(" 3-Perspective 평가 상세")
                print("-"*60)

                # Evidence Perspective
                print(f"\n[1] Evidence Perspective:")
                print(f"  ├─ Evidence Score: {perspectives.get('evidence_score', 0)}점")
                print(f"  ├─ Evidence Weight: {perspectives.get('evidence_weight', 0)}")
                print(f"  └─ Reasoning: {perspectives.get('evidence_reasoning', 'N/A')[:200]}...")

                evidence_details = perspectives.get('evidence_details', [])
                if evidence_details:
                    print(f"\n  증거 Quote ({len(evidence_details)}개):")
                    for i, ev in enumerate(evidence_details[:3], 1):
                        seg_id = ev.get('segment_id', 'N/A')
                        char_idx = ev.get('char_index', 'N/A')
                        text = ev.get('text', '')[:60]
                        relevance = ev.get('relevance_note', 'N/A')
                        quality = ev.get('quality_score', 0)
                        print(f"    {i}. [Seg {seg_id}, Idx {char_idx}] Quality: {quality:.2f}")
                        print(f"       \"{text}...\"")
                        print(f"       관련성: {relevance}")

                # Behavioral Perspective
                print(f"\n[2] Behavioral Perspective:")
                print(f"  ├─ Behavioral Score: {perspectives.get('behavioral_score', 0)}점")
                print(f"  └─ Reasoning: {perspectives.get('behavioral_reasoning', 'N/A')[:200]}...")

                behavioral_pattern = perspectives.get('behavioral_pattern', {})
                if behavioral_pattern:
                    print(f"\n  패턴 분석:")
                    print(f"    ├─ 설명: {behavioral_pattern.get('pattern_description', 'N/A')}")

                    examples = behavioral_pattern.get('specific_examples', [])
                    if examples:
                        print(f"    ├─ 구체적 예시 ({len(examples)}개):")
                        for i, ex in enumerate(examples[:2], 1):
                            print(f"    │   {i}. {ex}")

                    consistency = behavioral_pattern.get('consistency_note', '')
                    if consistency:
                        print(f"    └─ 일관성: {consistency}")

                # Critical Perspective
                print(f"\n[3] Critical Perspective:")
                print(f"  ├─ Critical Penalties: {perspectives.get('critical_penalties', 0)}점")
                print(f"  └─ Reasoning: {perspectives.get('critical_reasoning', 'N/A')[:200]}...")

                red_flags = perspectives.get('red_flags', [])
                if red_flags:
                    print(f"\n  Red Flags ({len(red_flags)}개):")
                    for i, flag in enumerate(red_flags, 1):
                        flag_type = flag.get('flag_type', 'N/A')
                        description = flag.get('description', 'N/A')
                        severity = flag.get('severity', 'N/A')
                        penalty = flag.get('penalty', 0)
                        evidence_ref = flag.get('evidence_reference', 'N/A')
                        print(f"    {i}. Type: {flag_type} | Severity: {severity} | Penalty: {penalty}점")
                        print(f"       설명: {description}")
                        print(f"       증거: {evidence_ref}")

            # Calculation 상세
            calculation = st_data.get('calculation', {})
            if calculation:
                print(f"\n" + "-"*60)
                print(" 점수 계산 상세")
                print("-"*60)
                print(f"  Base Score: {calculation.get('base_score', 0)}점")
                print(f"  Evidence Weight: {calculation.get('evidence_weight', 0)}")
                print(f"  Behavioral Adjustment: {calculation.get('behavioral_adjustment', 0)}")
                print(f"  Adjusted Base: {calculation.get('adjusted_base', 0)}")
                print(f"  Critical Penalties: {calculation.get('critical_penalties', 0)}")
                print(f"  Final Score: {calculation.get('final_score', 0)}점")
                print(f"  Formula: {calculation.get('formula', 'N/A')}")

            # 신뢰도
            confidence = st_data.get('confidence', {})
            if confidence:
                print(f"\n" + "-"*60)
                print(" 신뢰도 분석")
                print("-"*60)
                overall_conf = confidence.get('overall_confidence', 0)
                evidence_str = confidence.get('evidence_strength', 0)
                internal_cons = confidence.get('internal_consistency', 0)
                confidence_note = confidence.get('confidence_note', 'N/A')

                print(f"  ├─ Overall Confidence: {overall_conf:.3f} {'  (낮음)' if overall_conf < 0.7 else '✅ (높음)' if overall_conf >= 0.8 else ''}")
                print(f"  ├─ Evidence Strength: {evidence_str:.3f} {'  (부족)' if evidence_str < 0.6 else '✅' if evidence_str >= 0.8 else ''}")
                print(f"  ├─ Internal Consistency: {internal_cons:.3f}")
                print(f"  └─ Note: {confidence_note}")

                # 해석 추가
                if overall_conf >= 0.8 and evidence_str >= 0.8:
                    print(f"\n   해석: 충분한 증거와 일관된 평가 (신뢰도 높음)")
                elif overall_conf >= 0.7 and evidence_str < 0.6:
                    print(f"\n   해석: 증거는 적지만 평가는 일관적 (추가 질문 권장)")
                elif overall_conf < 0.7:
                    print(f"\n   해석: 평가 신뢰도 낮음 (재평가 또는 협업 필요)")

            # 강점/약점/관찰/후속질문
            print(f"\n" + "-"*60)
            print(" 평가 요약")
            print("-"*60)

            strengths = st_data.get('strengths', [])
            if strengths:
                print(f"\n강점 ({len(strengths)}개):")
                for i, s in enumerate(strengths, 1):
                    print(f"  {i}. {s}")

            weaknesses = st_data.get('weaknesses', [])
            if weaknesses:
                print(f"\n약점 ({len(weaknesses)}개):")
                for i, w in enumerate(weaknesses, 1):
                    print(f"  {i}. {w}")

            key_observations = st_data.get('key_observations', [])
            if key_observations:
                print(f"\n핵심 관찰 ({len(key_observations)}개):")
                for i, obs in enumerate(key_observations, 1):
                    print(f"  {i}. {obs}")

            followup_questions = st_data.get('suggested_followup_questions', [])
            if followup_questions:
                print(f"\n권장 후속 질문 ({len(followup_questions)}개):")
                for i, q in enumerate(followup_questions, 1):
                    print(f"  {i}. {q}")
        
        # ============================================
        # 실행 로그 출력
        # ============================================
        
        execution_logs = result.get("execution_logs", [])
        if execution_logs:
            print("\n" + "="*80)
            print("  실행 로그 (Performance)")
            print("="*80 + "\n")
            
            total_duration = 0
            total_cost = 0
            
            for log in execution_logs:
                phase = log.get("phase", "")
                node = log.get("node", "")
                duration = log.get("duration_seconds", 0)
                cost = log.get("cost_usd", 0)
                
                total_duration += duration
                total_cost += cost
                
                print(f"[{phase}] {node}: {duration:.2f}초")
            
            print(f"\n총 소요 시간: {total_duration:.2f}초")
        
        # ============================================
        # 최종 요약
        # ============================================
        
        print("\n" + "="*80)
        print(" 테스트 완료")
        print("="*80)
        
        print(f"\n 요약:")
        print(f"  ├─ 최종 점수: {final_score:.1f}점")
        print(f"  ├─ 신뢰도: {final_reliability}")
        print(f"  ├─ 협업 처리: {collab_count}회")
        print(f"  ├─ 소요 시간: {total_duration:.2f}초" if execution_logs else "  ├─ 소요 시간: N/A")
        
        print("\n" + "="*80 + "\n")
        
    except Exception as e:
        print(f"\n 오류 발생: {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_full_evaluation())
