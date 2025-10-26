"""
Adversarial Validator (Phase 3 Collaborator)
Low Confidence 역량을 비판적 관점에서 재평가

처리 방식:
1. 원본 평가 결과 수집 (Evidence, Behavioral, Critical)
2. Adversarial 질문 제시 ("과대평가 아닌가?")
3. LLM에게 재평가 요청
4. 조정된 점수 + Confidence 반환

비용 제한: 최대 5개 역량만 처리
"""

import json
from typing import Dict, List
from openai import AsyncOpenAI


class AdversarialValidator:
    """Adversarial 재평가자"""
    
    def __init__(self, openai_client: AsyncOpenAI):
        """
        Args:
            openai_client: OpenAI AsyncClient
        """
        self.client = openai_client
        self.model = "gpt-4o"
    
    async def validate_low_confidence(
        self,
        low_confidence_issues: List[Dict],
        all_results: Dict[str, Dict],
        max_validations: int = 5
    ) -> List[Dict]:
        """
        Low Confidence 역량 재평가 (배치 처리)
        
        Args:
            low_confidence_issues: Phase 2에서 탐지된 Low Confidence 목록
                [
                    {
                        "competency": "financial_literacy",
                        "overall_confidence": 0.54,
                        "evidence_strength": 0.6,
                        "internal_consistency": 0.48,
                        "reason": "consistency_low",
                        "priority": 1
                    },
                    ...
                ]
            all_results: 10개 역량 평가 결과
            max_validations: 최대 처리 개수 (기본 5)
        
        Returns:
            재평가 결과 목록
                [
                    {
                        "competency": "financial_literacy",
                        "original_score": 82,
                        "adversarial_score": 75,
                        "adjusted_score": 78,
                        "confidence_adjusted": 0.68,
                        "reasoning": "..."
                    },
                    ...
                ]
        """
        # Priority 높은 순서로 최대 개수만 처리
        issues_to_process = low_confidence_issues[:max_validations]
        
        results = []
        for issue in issues_to_process:
            try:
                result = await self._validate_single_competency(
                    issue,
                    all_results
                )
                results.append(result)
            except Exception as e:
                competency = issue["competency"]
                original_score = all_results[competency].get("overall_score", 0)
                print(f"⚠️  {competency} 재평가 실패: {e}")
                # 실패 시 원본 점수 유지
                results.append({
                    "competency": competency,
                    "original_score": original_score,
                    "adversarial_score": original_score,
                    "adjusted_score": original_score,
                    "confidence_adjusted": issue["overall_confidence"],
                    "reasoning": f"재평가 실패 (원본 점수 유지): {str(e)}"
                })
        
        return results
    
    async def _validate_single_competency(
        self,
        issue: Dict,
        all_results: Dict[str, Dict]
    ) -> Dict:
        """단일 역량 재평가"""
        
        competency = issue["competency"]
        result = all_results[competency]
        
        # 1. 원본 평가 정보 추출
        original_evaluation = self._extract_original_evaluation(result)
        
        # 2. 재평가 프롬프트 생성
        prompt = self._build_adversarial_prompt(
            competency,
            original_evaluation,
            issue
        )
        
        # 3. LLM 호출
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a critical HR evaluator. Re-evaluate competencies with skepticism. Respond with ONLY valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )
        
        # 4. 응답 파싱
        content = response.choices[0].message.content.strip()
        result_parsed = json.loads(content)
        
        return result_parsed
    
    def _extract_original_evaluation(self, result: Dict) -> Dict:
        """원본 평가 정보 추출"""
        
        perspectives = result.get("perspectives", {})
        confidence = result.get("confidence", {})
        
        return {
            "overall_score": result.get("overall_score", 0),
            "competency_display_name": result.get("competency_display_name", ""),
            "evidence_score": perspectives.get("evidence_score", 0),
            "evidence_weight": perspectives.get("evidence_weight", 0),
            "evidence_details": perspectives.get("evidence_details", [])[:5],  # 최대 5개
            "evidence_reasoning": perspectives.get("evidence_reasoning", ""),
            "behavioral_score": perspectives.get("behavioral_score", 0),
            "behavioral_reasoning": perspectives.get("behavioral_reasoning", ""),
            "critical_penalties": perspectives.get("critical_penalties", 0),
            "critical_reasoning": perspectives.get("critical_reasoning", ""),
            "confidence": confidence
        }
    
    def _build_adversarial_prompt(
        self,
        competency: str,
        original: Dict,
        issue: Dict
    ) -> str:
        """Adversarial 재평가 프롬프트 생성"""
        
        # Evidence 포맷팅 (최대 5개)
        evidence_text = ""
        for i, ev in enumerate(original["evidence_details"][:5], 1):
            evidence_text += f"""
  {i}. "{ev.get('text', '')[:150]}..."
     (Segment {ev.get('segment_id')}, Score: {ev.get('quality_score', 0):.2f})
     해석: {ev.get('relevance_note', '')}
"""
        
        # Low Confidence 원인 설명
        reason_explanation = {
            "evidence_weak": "증거가 부족합니다 (Quote 개수 적음, evidence_strength 낮음). 지원자가 이 역량을 충분히 보여주지 못했을 가능성.",
            "consistency_low": "평가 일관성이 낮습니다 (Evidence vs Behavioral 점수 차이 큼). Agent 간 해석 차이가 있을 가능성.",
            "both": "증거 부족 + 평가 일관성 모두 낮습니다."
        }.get(issue["reason"], "원인 불명")
        
        prompt = f"""다음 역량 평가를 **비판적 관점**에서 재검토하세요.

[원본 평가: {original['competency_display_name']}]
- Overall Score: {original['overall_score']}점
- Confidence: {issue['overall_confidence']:.2f} (⚠️ Low!)
  * Evidence Strength: {issue['evidence_strength']:.2f}
  * Internal Consistency: {issue['internal_consistency']:.2f}
  * 원인: {reason_explanation}

[Evidence Perspective]
- Evidence Score: {original['evidence_score']}점
- Evidence Weight: {original['evidence_weight']:.2f}
- Evidence Details:
{evidence_text}
- Evidence Reasoning: "{original['evidence_reasoning'][:300]}..."

[Behavioral Perspective]
- Behavioral Score: {original['behavioral_score']}점
- Behavioral Reasoning: "{original['behavioral_reasoning'][:300]}..."

[Critical Perspective]
- Critical Penalties: {original['critical_penalties']}점
- Critical Reasoning: "{original['critical_reasoning'][:200]}..."

**Adversarial 질문 (비판적 검토):**
1. ⚠️ 이 점수가 과대평가 아닌가?
   - Evidence가 실제로 이 역량을 증명하는가?
   - Quote 해석이 과도하게 긍정적이지 않은가?

2. ⚠️ Evidence가 충분한가?
   - Quote 개수가 적지 않은가?
   - 구체성이 부족하지 않은가?

3. ⚠️ 다른 역량과 혼동하지 않았나?
   - 이 Quote가 정말 이 역량의 증거인가?
   - 다른 역량(예: Problem Solving vs Structured Thinking)으로 봐야 하지 않나?

4. ⚠️ Evidence-Behavioral 불일치는 왜 발생했나?
   - 어느 쪽이 더 정확한가?
   - 점수 차이를 설명할 수 있는가?

**재평가 지침:**
1. 비판적 관점에서 원본 평가의 약점을 찾으세요.
2. 과대평가된 부분은 점수를 낮추세요.
3. 단, 명백한 증거가 있다면 인정하세요 (과도하게 감점하지 마세요).
4. 조정 후 Confidence를 0.6 이상으로 올릴 수 있는지 판단하세요.

**출력 형식 (JSON ONLY):**
{{
  "competency": "{competency}",
  "original_score": {original['overall_score']},
  "adversarial_score": 75,  // 비판적 관점에서 재산정한 점수
  "adjusted_score": 78,      // 원본과 Adversarial의 중재 점수
  "confidence_adjusted": 0.68,  // 조정 후 Confidence (0.6 이상 목표)
  "reasoning": "일부 Quote(Segment 3, 5)가 재무 감각보다는 일반적 비즈니스 직관에 가까움. 단, Segment 7의 'ROI 계산' 언급은 명백한 재무 개념 사용. 과대평가 요소를 제거하여 82→75점(Adversarial). 기본 재무 감각은 확인되므로 최종 78점으로 중재. Confidence를 0.68로 상향 (재평가로 불확실성 감소)."
}}

**중요:** 
- 반드시 JSON만 출력하세요. 다른 텍스트 금지.
- adjusted_score는 original_score와 adversarial_score의 **중간값** 정도로 설정하세요.
- confidence_adjusted는 재평가를 통해 불확실성이 감소했으므로 원본보다 높게 설정하세요 (최소 0.6 이상 목표).
"""
        
        return prompt