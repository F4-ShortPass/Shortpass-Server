"""
Segment Overlap Checker
같은 Segment를 여러 Agent가 평가한 경우 점수 차이 조정
"""

import json
from typing import Dict, List, Tuple, Optional
from openai import AsyncOpenAI
from collections import defaultdict


class SegmentOverlapChecker:
    """
    Segment Overlap 체크 및 조정
    
    로직:
        1. Segment ID로 그룹핑
        2. 점수 격차 > 1.5 (5점 척도 기준 30점) → 조정 필요
        3. Confidence 차이 > 0.2 → Rule-based 조정 (높은 Confidence 기준)
        4. Confidence 차이 < 0.2 → AI 호출 (판단 필요)
    """
    
    # Threshold
    SCORE_GAP_THRESHOLD = 1.5  # 5점 척도 기준 (100점 환산 시 30점)
    CONFIDENCE_GAP_THRESHOLD = 0.2
    
    def __init__(self, openai_client: AsyncOpenAI):
        self.client = openai_client
    
    
    async def check_and_adjust(
        self,
        segment_evaluations: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Segment Overlap 체크 및 조정
        
        Args:
            segment_evaluations: Confidence V2가 계산된 Segment 평가 목록
                [
                    {
                        "competency": "achievement_motivation",
                        "segment_id": 3,
                        "score": 85,
                        "confidence_v2": 0.85,
                        ...
                    },
                    ...
                ]
        
        Returns:
            (adjusted_segments, adjustment_logs)
            
            adjusted_segments: 조정된 Segment 평가 목록
            adjustment_logs: 조정 내역
                [
                    {
                        "segment_id": 3,
                        "competencies": ["achievement_motivation", "interpersonal_skills"],
                        "score_gap": 2.5,
                        "adjustment_type": "rule_based",  # or "ai_mediated"
                        "adjustments": [
                            {
                                "competency": "achievement_motivation",
                                "original_score": 2.0,
                                "adjusted_score": 2.75,
                                "original_confidence": 0.5,
                                "adjusted_confidence": 0.45,
                                "reason": "Adjusted towards higher confidence (interpersonal_skills)"
                            }
                        ]
                    },
                    ...
                ]
        """
        
        print("\n[Segment Overlap Checker] 시작")
        
        # 1. Segment별 그룹핑
        segment_groups = self._group_by_segment(segment_evaluations)
        
        print(f"  총 Segment 수: {len(segment_groups)}개")
        print(f"  중복 평가된 Segment: {sum(1 for segs in segment_groups.values() if len(segs) > 1)}개")
        
        
        # 2. 조정이 필요한 Segment 탐지
        segments_need_adjustment = []
        
        for segment_id, evaluations in segment_groups.items():
            if len(evaluations) < 2:
                continue  # 중복 없음
            
            # 점수 격차 계산 (5점 척도로 정규화)
            scores = [e["score"] for e in evaluations]
            score_gap = max(scores) - min(scores)
            
            if score_gap > 30:  # 수정 (기존 1.5 → 30)
                segments_need_adjustment.append({
                    "segment_id": segment_id,
                    "evaluations": evaluations,
                    "score_gap": score_gap
            })
        
        print(f"  조정 필요 Segment: {len(segments_need_adjustment)}개")
        
        if not segments_need_adjustment:
            print("   조정 불필요 - 모든 Segment 일관적")
            return segment_evaluations, []
        
        
        # 3. Segment별 조정 수행
        adjusted_segments = segment_evaluations.copy()
        adjustment_logs = []
        
        for segment_info in segments_need_adjustment:
            segment_id = segment_info["segment_id"]
            evaluations = segment_info["evaluations"]
            score_gap = segment_info["score_gap"]
            
            print(f"\n  [Segment {segment_id}] 조정 시작 (gap: {score_gap:.2f})")
            
            # Confidence 차이 체크
            confidences = [e["confidence_v2"] for e in evaluations]
            confidence_gap = max(confidences) - min(confidences)
            
            if confidence_gap > self.CONFIDENCE_GAP_THRESHOLD:
                # Rule-based 조정
                adjustment_result = self._adjust_by_rule(
                    segment_id,
                    evaluations,
                    score_gap
                )
                print(f"    → Rule-based 조정 완료")
            else:
                # AI 호출
                adjustment_result = await self._adjust_by_ai(
                    segment_id,
                    evaluations,
                    score_gap
                )
                print(f"    → AI 판단 완료")
            
            # 조정 결과 반영
            adjusted_segments = self._apply_adjustments(
                adjusted_segments,
                adjustment_result["adjustments"]
            )
            
            adjustment_logs.append(adjustment_result)
        
        print(f"\n[Segment Overlap Checker] 완료 - {len(adjustment_logs)}건 조정")
        
        return adjusted_segments, adjustment_logs
    
    
    def _group_by_segment(
        self,
        segment_evaluations: List[Dict]
    ) -> Dict[int, List[Dict]]:
        """
        Segment ID로 그룹핑
        
        Returns:
            {
                3: [
                    {"competency": "achievement", "score": 85, ...},
                    {"competency": "interpersonal", "score": 82, ...}
                ],
                7: [...],
                ...
            }
        """
        groups = defaultdict(list)
        
        for eval_item in segment_evaluations:
            segment_id = eval_item.get("segment_id")
            if segment_id is not None:
                groups[segment_id].append(eval_item)
        
        return dict(groups)
    
    
    def _adjust_by_rule(
        self,
        segment_id: int,
        evaluations: List[Dict],
        score_gap: float
    ) -> Dict:
        """
        Rule-based 조정 (높은 Confidence 기준으로 조정)
        
        조정 로직:
            1. 가장 높은 Confidence를 가진 평가를 Trust Agent로 선정
            2. 낮은 Confidence 평가들을 Trust Agent 방향으로 조정 (30%)
            3. 조정된 평가의 Confidence도 페널티 (-5%)
        
        Returns:
            {
                "segment_id": 3,
                "competencies": ["achievement_motivation", "interpersonal_skills"],
                "score_gap": 2.5,
                "adjustment_type": "rule_based",
                "adjustments": [
                    {
                        "competency": "achievement_motivation",
                        "original_score": 2.0,
                        "adjusted_score": 2.75,
                        "original_confidence": 0.5,
                        "adjusted_confidence": 0.45,
                        "reason": "Adjusted towards Trust Agent (interpersonal_skills, conf=0.85)"
                    }
                ]
            }
        """
        
        # Trust Agent 선정 (가장 높은 Confidence)
        trust_agent = max(evaluations, key=lambda e: e["confidence_v2"])
        trust_score_5scale = trust_agent["score"] / 20  # 100점 → 5점
        
        adjustments = []
        competencies = []
        
        for eval_item in evaluations:
            competencies.append(eval_item["competency"])
            
            if eval_item["competency"] == trust_agent["competency"]:
                continue  # Trust Agent는 조정 안 함
            
            original_score_5scale = eval_item["score"] / 20
            original_confidence = eval_item["confidence_v2"]
            
            # 30% 조정
            adjusted_score_5scale = original_score_5scale + (trust_score_5scale - original_score_5scale) * 0.3
            adjusted_score_100 = round(adjusted_score_5scale * 20, 1)  # 5점 → 100점
            
            # Confidence 페널티 -5%
            adjusted_confidence = max(0.3, original_confidence - 0.05)
            
            adjustments.append({
                "competency": eval_item["competency"],
                "original_score": eval_item["score"],
                "adjusted_score": adjusted_score_100,
                "original_confidence": original_confidence,
                "adjusted_confidence": round(adjusted_confidence, 2),
                "reason": f"Adjusted towards Trust Agent ({trust_agent['competency']}, conf={trust_agent['confidence_v2']:.2f})"
            })
        
        return {
            "segment_id": segment_id,
            "competencies": competencies,
            "score_gap": round(score_gap, 2),
            "adjustment_type": "rule_based",
            "adjustments": adjustments
        }
    
    
    async def _adjust_by_ai(
        self,
        segment_id: int,
        evaluations: List[Dict],
        score_gap: float
    ) -> Dict:
        """
        AI 기반 조정 (Confidence 차이가 작을 때)
        
        AI에게 질문:
            "둘 다 확신하는데 점수가 다름 - 어느 쪽이 맞나?"
        
        Returns:
            동일한 구조 (adjustment_type만 "ai_mediated")
        """
        
        # AI 프롬프트 생성
        prompt = self._build_ai_mediation_prompt(segment_id, evaluations)
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at mediating conflicting competency evaluations. When multiple agents evaluate the same segment with similar confidence but different scores, determine which evaluation is more accurate."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            ai_result = json.loads(result_text)
            
            # AI 결과를 표준 형식으로 변환
            return self._parse_ai_mediation_result(
                segment_id,
                evaluations,
                score_gap,
                ai_result
            )
        
        except Exception as e:
            print(f"      AI 호출 실패 - Rule-based로 폴백: {e}")
            return self._adjust_by_rule(segment_id, evaluations, score_gap)
    
    
    def _build_ai_mediation_prompt(
        self,
        segment_id: int,
        evaluations: List[Dict]
    ) -> str:
        """
        AI 조정 프롬프트 생성
        """
        
        eval_summary = []
        for e in evaluations:
            eval_summary.append({
                "competency": e["competency"],
                "score": e["score"],
                "confidence_v2": e["confidence_v2"],
                "quotes": e.get("quotes", [])
            })
        
        prompt = f"""# Task: Mediate Conflicting Evaluations

Multiple competency agents evaluated the same segment ({segment_id}) with similar confidence but different scores.

## Evaluations:
```json
{json.dumps(eval_summary, ensure_ascii=False, indent=2)}
```

## Instructions:
1. Analyze which evaluation is more accurate based on:
   - Quality of quotes
   - Relevance to competency
   - Consistency with evidence

2. Determine adjustments needed for lower-scored evaluations.

3. Output JSON format:
{{
  "trust_agent": "interpersonal_skills",  // Most accurate evaluation
  "adjustments": [
    {{
      "competency": "achievement_motivation",
      "adjustment_direction": "increase",  // "increase" or "decrease"
      "adjustment_amount": 0.5,  // 5점 척도 기준
      "confidence_penalty": 0.05,
      "reason": "Evidence better supports interpersonal skills evaluation"
    }}
  ]
}}

CRITICAL: Output ONLY valid JSON. NO markdown.
"""
        return prompt
    
    
    def _parse_ai_mediation_result(
        self,
        segment_id: int,
        evaluations: List[Dict],
        score_gap: float,
        ai_result: Dict
    ) -> Dict:
        """
        AI 결과를 표준 형식으로 변환
        """
        
        adjustments = []
        competencies = [e["competency"] for e in evaluations]
        
        for ai_adj in ai_result.get("adjustments", []):
            competency = ai_adj.get("competency")
            
            # 원본 평가 찾기
            original_eval = next((e for e in evaluations if e["competency"] == competency), None)
            if not original_eval:
                continue
            
            original_score_5scale = original_eval["score"] / 20
            adjustment_amount = ai_adj.get("adjustment_amount", 0)
            direction = ai_adj.get("adjustment_direction", "increase")
            
            if direction == "increase":
                adjusted_score_5scale = original_score_5scale + adjustment_amount
            else:
                adjusted_score_5scale = original_score_5scale - adjustment_amount
            
            adjusted_score_100 = round(adjusted_score_5scale * 20, 1)
            
            original_confidence = original_eval["confidence_v2"]
            confidence_penalty = ai_adj.get("confidence_penalty", 0.05)
            adjusted_confidence = max(0.3, original_confidence - confidence_penalty)
            
            adjustments.append({
                "competency": competency,
                "original_score": original_eval["score"],
                "adjusted_score": adjusted_score_100,
                "original_confidence": original_confidence,
                "adjusted_confidence": round(adjusted_confidence, 2),
                "reason": ai_adj.get("reason", "AI mediation")
            })
        
        return {
            "segment_id": segment_id,
            "competencies": competencies,
            "score_gap": round(score_gap, 2),
            "adjustment_type": "ai_mediated",
            "trust_agent": ai_result.get("trust_agent"),
            "adjustments": adjustments
        }
    
    
    def _apply_adjustments(
        self,
        segment_evaluations: List[Dict],
        adjustments: List[Dict]
    ) -> List[Dict]:
        """
        조정 결과를 Segment 평가 목록에 반영
        """
        
        updated = []
        
        for seg_eval in segment_evaluations:
            # 조정 대상인지 확인
            adjustment = next(
                (adj for adj in adjustments if adj["competency"] == seg_eval["competency"]),
                None
            )
            
            if adjustment:
                # 조정 적용
                updated_eval = {
                    **seg_eval,
                    "score": adjustment["adjusted_score"],
                    "confidence_v2": adjustment["adjusted_confidence"],
                    "adjusted": True,
                    "adjustment_reason": adjustment["reason"]
                }
                updated.append(updated_eval)
            else:
                # 조정 없음
                updated.append(seg_eval)
        
        return updated