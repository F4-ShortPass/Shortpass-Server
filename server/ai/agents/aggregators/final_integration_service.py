"""
Final Integrator
Stage 3: 최종 점수 계산 및 신뢰도 평가

처리 내용:
    1. 10개 역량 가중 평균으로 최종 점수 계산
    2. 평균 Confidence V2 계산
    3. 신뢰도 레벨 판단
    4. 종합 심사평 생성 (AI 호출)
    5. 최종 리포트 구성
"""

import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from openai import AsyncOpenAI

from services.evaluation.post_processing_service import PostProcessingService


class FinalIntegrator:
    """
    최종 통합기
    
    역할:
        - Job/Common 구분 없이 10개 역량 직접 처리
        - Confidence V2 기반 신뢰도 평가
        - Collaboration 결과 반영
        - 종합 심사평 생성 (AI)
    """
    
    # 신뢰도 레벨 Threshold
    RELIABILITY_THRESHOLDS = {
        "very_high": 0.85,  # 매우 높음
        "high": 0.70,       # 높음
        "medium": 0.55,     # 중간
        "low": 0.0          # 낮음
    }
    
    
    @staticmethod
    async def integrate(
        openai_client: AsyncOpenAI,
        aggregated_competencies: Dict[str, Dict],
        competency_weights: Dict[str, float],
        collaboration_results: Optional[List[Dict]] = None,
        low_confidence_list: Optional[List[Dict]] = None
    ) -> Dict:
        """
        최종 통합
        
        Args:
            openai_client: OpenAI AsyncClient (종합 심사평 생성용)
            aggregated_competencies: Aggregator에서 집계된 10개 역량
            competency_weights: 10개 역량 가중치
            collaboration_results: Collaboration Node 결과 (선택적)
            low_confidence_list: Low Confidence 목록
        
        Returns:
            final_result: 최종 리포트
        """
        
        print("\n[Final Integrator] 최종 통합 시작")
        
    
        # 1. Collaboration 결과 반영       
        if collaboration_results:
            print(f"  Collaboration 결과 반영: {len(collaboration_results)}건")
            aggregated_competencies = FinalIntegrator._apply_collaboration_results(
                aggregated_competencies,
                collaboration_results
            )
        else:
            print("  Collaboration 결과 없음 (스킵)")
        
        
    
        # 2. 최종 점수 계산 (가중 평균) 
        final_score, competency_scores = FinalIntegrator._calculate_final_score(
            aggregated_competencies,
            competency_weights
        )
        
        print(f"\n  최종 점수: {final_score:.1f}점")
        if competency_scores:
            print("  상위 기여 역량:")
            for cs in competency_scores[:3]:
                print(
                    f"    • {cs['competency']}: score={cs['score']} "
                    f"weight={cs['weight']:.2f} contrib={cs['contribution']}"
                )
            if len(competency_scores) > 3:
                print("    • ...")
        

    
        # 3. 평균 Confidence V2 계산
        avg_confidence = FinalIntegrator._calculate_avg_confidence(
            aggregated_competencies,
            competency_weights
        )
        
        print(f"  평균 Confidence V2: {avg_confidence:.2f}")
        if competency_scores:
            best_conf = max(competency_scores, key=lambda c: c["confidence_v2"])
            worst_conf = min(competency_scores, key=lambda c: c["confidence_v2"])
            print(
                f"  Confidence 범위: {best_conf['competency']}={best_conf['confidence_v2']:.2f} "
                f"/ {worst_conf['competency']}={worst_conf['confidence_v2']:.2f}"
            )
        

    
        # 4. 신뢰도 레벨 판단
        reliability = FinalIntegrator._determine_reliability(
            avg_confidence,
            low_confidence_list or []
        )
        
        print(f"  신뢰도 레벨: {reliability['level']}")
        print(f"  신뢰도 근거: {reliability['note']}")
        if low_confidence_list:
            print("  Low Confidence 상세:")
            for item in low_confidence_list[:5]:
                print(f"    • {item.get('competency')}: conf={item.get('confidence_v2')}")
            if len(low_confidence_list) > 5:
                print("    • ...")
        

    
        # 5. 종합 심사평 생성 (AI 호출)
        print("\n   종합 심사평 생성 중 (AI 호출)...")
        overall_evaluation_summary = await FinalIntegrator._generate_overall_evaluation_summary(
            openai_client,
            aggregated_competencies,
            final_score,
            avg_confidence,
            reliability
        )
        print(f"   종합 심사평 생성 완료 ({len(overall_evaluation_summary)} chars)")
        
        
    
        # 6. Collaboration 요약
        collaboration_summary = FinalIntegrator._summarize_collaboration(
            collaboration_results or []
        )
        
        
    
        # 7. Low Confidence 요약
        low_confidence_summary = FinalIntegrator._summarize_low_confidence(
            low_confidence_list or []
        )
        
        
    
        # 8. 최종 리포트 구성
        final_result = {
            "final_score": final_score,
            "avg_confidence": avg_confidence,
            
            "reliability": reliability,
            
            #  종합 심사평 (AI 생성)
            "overall_evaluation_summary": overall_evaluation_summary,
            
            "competency_scores": competency_scores,
            
            "collaboration_summary": collaboration_summary,
            
            "low_confidence_summary": low_confidence_summary,
            
            # 역량별 상세 (Resume 검증 근거 포함)
            "competency_details": {
                comp_name: {
                    "overall_score": comp_data["overall_score"],
                    "confidence_v2": comp_data["confidence_v2"],
                    "weight": competency_weights.get(comp_name, 0.0),
                    "weighted_contribution": comp_data["overall_score"] * competency_weights.get(comp_name, 0.0),
                    "resume_verified_count": comp_data.get("resume_verified_count", 0),
                    "segment_count": comp_data.get("segment_count", 0),
                    "perspectives": comp_data.get("perspectives", {}),
                    "strengths": comp_data.get("strengths", []),
                    "weaknesses": comp_data.get("weaknesses", []),
                    "key_observations": comp_data.get("key_observations", []),
                    "resume_verification_summary": comp_data.get("resume_verification_summary", {})
                }
                for comp_name, comp_data in aggregated_competencies.items()
            }
        }
        
        print("\n[Final Integrator] 최종 통합 완료")
        
        return final_result
    
    
    @staticmethod
    async def _generate_overall_evaluation_summary(
        openai_client: AsyncOpenAI,
        aggregated_competencies: Dict[str, Dict],
        final_score: float,
        avg_confidence: float,
        reliability: Dict
    ) -> str:
        """
        종합 심사평 생성 (AI 호출)
        
        전략: Option 1 - 역량별 핵심 관찰 (key_observations)을 종합하여 심사평 작성
        
        Returns:
            5-7문장으로 구성된 종합 심사평
        """
        
        # 역량별 핵심 관찰 및 점수 정리
        competency_summary = {}
        total_resume_verified = 0
        
        for comp_name, comp_data in aggregated_competencies.items():
            competency_summary[comp_name] = {
                "score": comp_data["overall_score"],
                "confidence_v2": comp_data["confidence_v2"],
                "key_observations": comp_data.get("key_observations", []),
                "strengths": comp_data.get("strengths", []),
                "weaknesses": comp_data.get("weaknesses", []),
                "resume_verified_count": comp_data.get("resume_verified_count", 0)
            }
            total_resume_verified += comp_data.get("resume_verified_count", 0)
        
        # AI 프롬프트 생성
        prompt = f"""당신은 패션 MD 직무 채용을 위한 HR 전문가입니다.

10개 역량 평가 결과를 종합하여 지원자에 대한 **최종 심사평**을 작성해야 합니다.

## 역량별 평가 요약:
```json
{json.dumps(competency_summary, ensure_ascii=False, indent=2)}
```

## 최종 점수: {final_score:.1f}점 / 100점
## 평균 신뢰도 (Confidence V2): {avg_confidence:.2f}
## 신뢰도 레벨: {reliability['level_display']}
## Resume 검증: 총 {total_resume_verified}개 증거 확인됨

---

## 종합 심사평 작성 가이드라인:

### 1. 평가의 초점
- **종합적 평가**: 강점/약점 나열이 아닌, 지원자의 전체적인 모습을 입체적으로 평가
- **직무 적합성**: 패션 MD 직무에 얼마나 적합한지
- **역량 간 균형**: 10개 역량이 고르게 발달했는지, 편중되지 않았는지
- **Resume 신뢰도**: 면접 답변이 Resume 경력과 얼마나 일치하는지
- **성장 가능성**: 신입 기준으로 향후 성장 잠재력이 있는지

### 2. 심사평 구조 (5-7문장)
1문장: 전체적인 인상 및 종합 평가
2-3문장: 주요 강점 (높은 점수 역량 중심, Resume 검증 언급)
1-2문장: 보완 필요 영역 (낮은 점수 역량, 균형 문제)
1문장: 최종 판단 (채용 추천 여부, 성장 기대)

### 3. 주의사항
- ❌ "성취동기가 높습니다. 성장잠재력이 양호합니다." 같은 단순 나열 금지
- ✅ "지원자는 높은 성취동기를 바탕으로 자기주도적 학습을 해왔으며..." 같은 연결된 서술
- ❌ 점수를 그대로 언급 ("85점", "70점") 금지
- ✅ "우수함", "양호함", "보완 필요" 같은 표현 사용
- ✅ Resume 검증 결과를 자연스럽게 녹여서 언급
  - 예: "Resume에서도 리테일 분석 공모전 입상 경력이 확인되어 신뢰도가 높습니다"

### 4. 신입 기준 평가
- 신입 지원자임을 염두에 두고 평가
- "경험 부족"보다는 "성장 가능성", "학습 태도" 중심
- 과도한 기대치 적용 금지

---

## 출력 형식 (JSON):
{{
  "overall_evaluation_summary": "지원자는 패션 MD 직무에 필요한 대부분의 역량을 고르게 갖추고 있습니다. 특히 고객 여정 설계 및 마케팅 통합 전략 역량이 우수하며, Resume에서도 관련 프로젝트 경험이 다수 확인되어 신뢰도가 높습니다. 데이터 분석 및 상품 기획 역량도 양호하여 데이터 기반 의사결정이 가능할 것으로 보입니다. 다만 조직 적합성과 유관부서 협업 역량이 다소 낮아, 팀 문화 적응 및 이해관계자 조정에 시간이 필요할 수 있습니다. 전반적으로 신입 기준 상위권 수준이며, 입사 후 체계적인 교육과 멘토링을 통해 빠른 성장이 기대됩니다."
}}

---

**중요**: 
- 반드시 JSON만 출력하세요. 다른 텍스트 금지.
- 마크다운 코드 블록 (```) 사용 금지.
- overall_evaluation_summary는 한 문단으로 작성 (줄바꿈 없음).
"""
        
        try:
            response = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert HR evaluator specializing in fashion MD hiring. Create comprehensive, insightful evaluation summaries."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # 마크다운 제거
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            result_text = result_text.strip()
            
            result = json.loads(result_text)
            return result.get("overall_evaluation_summary", "")
        
        except Exception as e:
            print(f"    종합 심사평 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return f"지원자는 패션 MD 직무에 필요한 역량을 전반적으로 갖추고 있습니다 (종합 점수: {final_score:.1f}점). 신입 기준으로 적합하며, 입사 후 성장이 기대됩니다."
    
    
    @staticmethod
    def _apply_collaboration_results(
        aggregated_competencies: Dict[str, Dict],
        collaboration_results: List[Dict]
    ) -> Dict[str, Dict]:
        """
        Collaboration 결과를 Aggregated Competencies에 반영
        """
        
        for result in collaboration_results:
            competency = result.get("competency")
            if competency not in aggregated_competencies:
                continue
            
            # Confidence 보정 적용
            aggregated_competencies[competency]["confidence_v2"] = min(
                result.get("confidence_adjusted", aggregated_competencies[competency].get("confidence_v2", 0)),
                1.0
            )
            
            # Collaboration 근거 추가
            aggregated_competencies[competency]["collaboration_adjustment"] = {
                "original_score": result.get("original_score"),
                "adjusted_score": result.get("adjusted_score"),
                "reason": result.get("reason", "Collaboration applied")
            }
        
        return aggregated_competencies
    
    
    @staticmethod
    def _calculate_final_score(
        aggregated_competencies: Dict[str, Dict],
        competency_weights: Dict[str, float]
    ) -> Tuple[float, List[Dict]]:
        """
        가중 평균으로 최종 점수 계산
        """
        
        total_score = 0.0
        total_weight = 0.0
        competency_scores = []
        
        for competency, result in aggregated_competencies.items():
            score = result.get("overall_score", 0)
            weight = competency_weights.get(competency, 0.1)  # 기본 0.1 가중치
            
            total_score += score * weight
            total_weight += weight
            
            competency_scores.append({
                "competency": competency,
                "score": round(score, 1),
                "weight": round(weight, 2),
                "contribution": round(score * weight, 2),
                "confidence_v2": result.get("confidence_v2", 0.0)
            })
        
        # Normalize
        if total_weight == 0:
            return 0.0, competency_scores
        
        final_score = total_score / total_weight
        
        # 기여도 내림차순 정렬
        competency_scores = sorted(
            competency_scores,
            key=lambda x: x["contribution"],
            reverse=True
        )
        
        return final_score, competency_scores
    
    
    @staticmethod
    def _calculate_avg_confidence(
        aggregated_competencies: Dict[str, Dict],
        competency_weights: Dict[str, float]
    ) -> float:
        """
        가중치 기반 평균 Confidence V2 계산
        """
        
        total_confidence = 0.0
        total_weight = 0.0
        
        for comp_name, comp_data in aggregated_competencies.items():
            confidence = comp_data.get("confidence_v2", 0.0)
            weight = competency_weights.get(comp_name, 0.1)
            
            total_confidence += confidence * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return round(total_confidence / total_weight, 3)
    
    
    @staticmethod
    def _determine_reliability(
        avg_confidence: float,
        low_confidence_list: List[Dict]
    ) -> Dict:
        """
        신뢰도 레벨 판단
        
        Returns:
            {
                "level": "medium",
                "level_display": "중간 신뢰도",
                "note": "Low Confidence 역량 2개 감지"
            }
        """
        
        if avg_confidence >= FinalIntegrator.RELIABILITY_THRESHOLDS["very_high"]:
            level = "very_high"
            level_display = "매우 높음"
            note = "전반적으로 매우 높은 신뢰도의 인터뷰 응답"
        elif avg_confidence >= FinalIntegrator.RELIABILITY_THRESHOLDS["high"]:
            level = "high"
            level_display = "높음"
            note = "대체로 신뢰할 수 있는 응답, 일부 세부 확인 필요"
        elif avg_confidence >= FinalIntegrator.RELIABILITY_THRESHOLDS["medium"]:
            level = "medium"
            level_display = "중간"
            note = "일부 응답에서 불확실성이 감지됨, 추가 확인 권장"
        else:
            level = "low"
            level_display = "낮음"
            note = "여러 응답에서 신뢰도 낮음, 면접 결과 활용 시 주의 필요"
        
        if low_confidence_list:
            note += f" (Low Confidence 역량 {len(low_confidence_list)}개 감지)"
        
        return {
            "level": level,
            "level_display": level_display,
            "note": note
        }
    
    
    @staticmethod
    def _summarize_collaboration(
        collaboration_results: List[Dict]
    ) -> Dict:
        """
        Collaboration 결과 요약
        """
        
        if not collaboration_results:
            return {
                "collaboration_applied": False,
                "collaboration_count": 0,
                "details": []
            }
        
        return {
            "collaboration_applied": True,
            "collaboration_count": len(collaboration_results),
            "details": collaboration_results
        }
    
    
    @staticmethod
    def _summarize_low_confidence(
        low_confidence_list: List[Dict]
    ) -> Dict:
        """
        Low Confidence 요약
        """
        
        if not low_confidence_list:
            return {
                "total_low_confidence": 0,
                "competencies": [],
                "details": []
            }
        
        competencies = [item["competency"] for item in low_confidence_list]
        
        return {
            "total_low_confidence": len(low_confidence_list),
            "competencies": competencies,
            "details": low_confidence_list
        }
