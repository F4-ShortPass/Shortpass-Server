"""
Evidence Mediator (Phase 3 Collaborator)
같은 segment를 여러 역량이 다르게 해석한 경우 중재

처리 방식:
1. Segment 원문 추출 (transcript에서)
2. 충돌된 역량들의 해석 제시
3. LLM에게 중재 요청
4. 조정된 점수 반환

비용 제한: 최대 3개 충돌만 처리
"""

import json
from typing import Dict, List
from openai import AsyncOpenAI


class EvidenceMediator:
    """Evidence 충돌 중재자"""
    
    def __init__(self, openai_client: AsyncOpenAI):
        """
        Args:
            openai_client: OpenAI AsyncClient
        """
        self.client = openai_client
        self.model = "gpt-4o"
    
    async def mediate_conflicts(
        self,
        conflicts: List[Dict],
        all_results: Dict[str, Dict],
        transcript: Dict,
        max_mediations: int = 3
    ) -> List[Dict]:
        """
        Evidence 충돌 중재 (배치 처리)
        
        Args:
            conflicts: Phase 2에서 탐지된 충돌 목록
                [
                    {
                        "segment_id": "5",
                        "competencies": ["structured_thinking", "problem_solving"],
                        "scores": [0.9, 0.45],
                        "gap": 0.45,
                        "priority": 1
                    },
                    ...
                ]
            all_results: 10개 역량 평가 결과
            transcript: 전체 면접 transcript
            max_mediations: 최대 처리 개수 (기본 3)
        
        Returns:
            중재 결과 목록
                [
                    {
                        "segment_id": "5",
                        "primary_competency": "structured_thinking",
                        "adjusted_scores": {
                            "structured_thinking": 0.90,
                            "problem_solving": 0.60
                        },
                        "reasoning": "..."
                    },
                    ...
                ]
        """
        # Priority 높은 순서로 최대 개수만 처리
        conflicts_to_process = conflicts[:max_mediations]
        
        results = []
        for conflict in conflicts_to_process:
            try:
                result = await self._mediate_single_conflict(
                    conflict,
                    all_results,
                    transcript
                )
                results.append(result)
            except Exception as e:
                print(f"⚠️  Segment {conflict['segment_id']} 중재 실패: {e}")
                # 실패 시 원본 점수 유지
                results.append({
                    "segment_id": conflict["segment_id"],
                    "primary_competency": conflict["competencies"][0],
                    "adjusted_scores": {
                        comp: score 
                        for comp, score in zip(conflict["competencies"], conflict["scores"])
                    },
                    "reasoning": f"중재 실패 (원본 점수 유지): {str(e)}"
                })
        
        return results
    
    async def _mediate_single_conflict(
        self,
        conflict: Dict,
        all_results: Dict[str, Dict],
        transcript: Dict
    ) -> Dict:
        """단일 충돌 중재"""
        
        segment_id = conflict["segment_id"]
        competencies = conflict["competencies"]
        
        # 1. Segment 원문 추출
        segment_text = self._extract_segment_text(transcript, segment_id)
        
        # 2. 각 역량의 해석 추출
        interpretations = self._extract_interpretations(
            all_results,
            competencies,
            segment_id
        )
        
        # 3. 중재 프롬프트 생성
        prompt = self._build_mediation_prompt(
            segment_id,
            segment_text,
            interpretations
        )
        
        # 4. LLM 호출
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert HR evaluator. Mediate conflicts between competency evaluations. Respond with ONLY valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        # 5. 응답 파싱
        content = response.choices[0].message.content.strip()
        result = json.loads(content)
        
        return result
    
    def _extract_segment_text(self, transcript: Dict, segment_id: str) -> str:
        """
        Transcript에서 segment 원문 추출
        
        Args:
            transcript: 전체 면접 transcript
            segment_id: segment ID
        
        Returns:
            segment 텍스트 (질문 + 답변)
        """
        # Transcript 구조 확인 (qa_pairs 또는 segments)
        qa_pairs = transcript.get("qa_pairs", [])
        segments = transcript.get("segments", [])
        
        # qa_pairs에서 찾기
        for qa in qa_pairs:
            if str(qa.get("id", "")).replace("q", "") == str(segment_id):
                question = qa.get("question_text", "")
                answer = qa.get("answer_text", "")
                return f"질문: {question}\n\n답변: {answer}"
        
        # segments에서 찾기
        for seg in segments:
            if str(seg.get("segment_id", "")) == str(segment_id):
                question = seg.get("question_text", "")
                answer = seg.get("answer_text", "")
                return f"질문: {question}\n\n답변: {answer}"
        
        return f"[Segment {segment_id} 원문을 찾을 수 없음]"
    
    def _extract_interpretations(
        self,
        all_results: Dict[str, Dict],
        competencies: List[str],
        segment_id: str
    ) -> List[Dict]:
        """
        각 역량의 해석 추출
        
        Returns:
            [
                {
                    "competency": "structured_thinking",
                    "score": 0.9,
                    "quote": "...",
                    "relevance_note": "..."
                },
                ...
            ]
        """
        interpretations = []
        
        for comp in competencies:
            result = all_results.get(comp, {})
            perspectives = result.get("perspectives", {})
            evidence_details = perspectives.get("evidence_details", [])
            
            # 해당 segment를 참조한 evidence 찾기
            for evidence in evidence_details:
                if str(evidence.get("segment_id", "")) == str(segment_id):
                    interpretations.append({
                        "competency": comp,
                        "competency_display_name": result.get("competency_display_name", comp),
                        "score": evidence.get("quality_score", 0.0),
                        "quote": evidence.get("text", ""),
                        "relevance_note": evidence.get("relevance_note", "")
                    })
                    break
        
        return interpretations
    
    def _build_mediation_prompt(
        self,
        segment_id: str,
        segment_text: str,
        interpretations: List[Dict]
    ) -> str:
        """중재 프롬프트 생성"""
        
        # 해석 정보 포맷팅
        interp_text = ""
        for i, interp in enumerate(interpretations, 1):
            interp_text += f"""
[{i}. {interp['competency_display_name']} 평가]
- Quality Score: {interp['score']:.2f}
- 해석: "{interp['relevance_note']}"
- 추출한 Quote: "{interp['quote'][:200]}..."
"""
        
        prompt = f"""다음 segment에 대해 {len(interpretations)}개 역량이 다르게 평가했습니다.

[Segment {segment_id} 원문]
{segment_text}

{interp_text}

**중재 지침:**
1. 이 segment가 어느 역량의 증거로 더 적합한지 판단하세요.
2. 각 역량의 quality_score를 0.0-1.0 범위에서 조정하세요.
3. Primary competency (가장 적합한 역량)를 선택하세요.
4. 조정 이유를 구체적으로 설명하세요.

**출력 형식 (JSON ONLY):**
{{
  "segment_id": "{segment_id}",
  "primary_competency": "structured_thinking",  // 가장 적합한 역량
  "adjusted_scores": {{
    "structured_thinking": 0.90,  // 조정된 점수 (유지 또는 변경)
    "problem_solving": 0.60       // 조정된 점수
  }},
  "reasoning": "이 segment는 문제를 3가지 축으로 나눈 구조화 사고를 보여줌. Problem Solving도 일부 관련되나 원인 분석보다는 구조화에 초점. Problem Solving 점수를 0.45→0.60으로 상향 조정 (완전히 무관하지는 않음)."
}}

**중요:** 반드시 JSON만 출력하세요. 다른 텍스트 금지.
"""
        
        return prompt