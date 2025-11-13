"""
Presentation Formatter
프론트엔드용 데이터 재구성 + LLM 배치 근거 재생성
"""

import json
import os
from typing import Dict, List
from openai import AsyncOpenAI
from pathlib import Path
from dotenv import load_dotenv


class PresentationFormatter:
    """
    프론트엔드용 데이터 변환기
    
    핵심 기능:
        1. LLM 배치 호출로 모든 역량 근거 한 번에 재생성
        2. Strengths/Weaknesses/Key_observations 평서형 변환
        3. Resume 검증 결과 포함
        4. 역량별 근거 그룹핑 (segment 여러 개)
        5. Transcript 매핑 정보 포함
        6. 직무/공통 점수 계산
    """
    
    # 역량명 한글 매핑
    COMPETENCY_DISPLAY_NAMES = {
        "achievement_motivation": "성취/동기 역량",
        "growth_potential": "성장 잠재력",
        "interpersonal_skill": "대인관계 역량",
        "organizational_fit": "조직 적합성",
        "problem_solving": "문제 해결",
        "customer_journey_marketing": "고객 여정 마케팅",
        "md_data_analysis": "MD 데이터 분석",
        "seasonal_strategy_kpi": "시즌 전략 KPI",
        "stakeholder_collaboration": "이해관계자 협업",
        "value_chain_optimization": "가치사슬 최적화",
    }
    
    # 역량 그룹 정의
    JOB_COMPETENCIES = [
        "customer_journey_marketing",
        "md_data_analysis", 
        "seasonal_strategy_kpi",
        "stakeholder_collaboration",
        "value_chain_optimization"
    ]
    
    COMMON_COMPETENCIES = [
        "achievement_motivation",
        "growth_potential",
        "interpersonal_skill",
        "organizational_fit",
        "problem_solving"
    ]
    
    
    def __init__(self, openai_client: AsyncOpenAI):
        self.client = openai_client
        
        env_path = Path(__file__).resolve().parents[3] / ".env"
        load_dotenv(env_path, override=False)
        # 기본값은 gpt-4o, 필요 시 .env에서 OPENAI_MODEL로 오버라이드
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.summary_model = os.getenv("OPENAI_SUMMARY_MODEL", self.model)
        self._transcript_data = None
    
    
    async def format(
        self,
        final_result: Dict,
        aggregated_competencies: Dict,
        competency_weights: Dict[str, float],
        transcript: Dict 
    ) -> Dict:
        """
        프론트엔드용 응답 생성
        """
        
        print("\n[Presentation Formatter] 근거 배치 재생성 시작...")
        
        # 1. 전체 요약
        overall_summary = self._extract_overall_summary(final_result)
        
        # 2. 점수 분해 (전체/직무/공통)
        score_breakdown = self._calculate_score_breakdown(
            aggregated_competencies,
            competency_weights,
            final_result
        )
        
        # 3. 역량별 점수
        competency_scores = final_result.get("competency_scores", [])
        
        # 4. 배치로 모든 역량 재생성 (LLM 1회 호출)
        print("  10개 역량의 근거/강점/약점/관찰을 1번의 LLM 호출로 배치 생성 중...")
        
        batch_result = await self._regenerate_all_batch(
            aggregated_competencies,
            transcript
        )
        
        # 5. 역량별 상세 구성
        competency_details = {}
        
        for comp_name, comp_data in aggregated_competencies.items():
            comp_batch = batch_result.get(comp_name, {})
            
            competency_details[comp_name] = {
                "competency_display_name": self.COMPETENCY_DISPLAY_NAMES.get(comp_name, comp_name),
                "overall_score": comp_data.get("overall_score"),
                "confidence_v2": comp_data.get("confidence_v2"),
                
                # 평서형으로 재생성
                "strengths": comp_batch.get("strengths", comp_data.get("strengths", [])),
                "weaknesses": comp_batch.get("weaknesses", comp_data.get("weaknesses", [])),
                "key_observations": comp_batch.get("key_observations", comp_data.get("key_observations", [])),
                
                # 재생성된 근거
                "evidences": comp_batch.get("evidences", []),
                
                # Resume 검증 결과
                "resume_verification_summary": comp_data.get("resume_verification_summary", {
                    "verified_count": 0,
                    "high_strength_count": 0,
                    "key_evidence": []
                })
            }
        
        
        total_evidences = sum(len(cd.get('evidences', [])) for cd in competency_details.values())
        print(f"  배치 생성 완료: 총 {total_evidences}개 근거")
        
        # Connected Summary 추가
        print("  Connected Summary 생성 중...")
        competency_details = await self._add_connected_summaries(competency_details)
        
        total_evidences = sum(
            len(cd.get('evidences', [])) 
            for cd in competency_details.values()
        )
        
        print(f"\n  총 근거 생성: {total_evidences}개")
        print("  강점/약점/관찰 평서형 변환: 10개 역량")
        print("  Resume 검증 결과 포함: 완료")
        
        return {
            "overall_summary": overall_summary,
            "score_breakdown": score_breakdown,
            "competency_scores": competency_scores,
            "competency_details": competency_details
        }
    
    def _extract_overall_summary(self, final_result: Dict) -> Dict:
        """전체 요약 추출"""
        return {
            "final_score": final_result.get("final_score"),
            "avg_confidence": final_result.get("avg_confidence"),
            "reliability": final_result.get("reliability"),
            "overall_evaluation_summary": final_result.get("overall_evaluation_summary")
        }
    
    
    def _calculate_score_breakdown(
        self,
        aggregated_competencies: Dict,
        competency_weights: Dict[str, float],
        final_result: Dict
    ) -> Dict:
        """
        점수 분해 계산 (전체/직무/공통)
        """
        
        # 직무 역량 점수
        job_total = 0.0
        job_weight_sum = 0.0
        
        for comp_name in self.JOB_COMPETENCIES:
            if comp_name in aggregated_competencies:
                score = aggregated_competencies[comp_name].get("overall_score", 0)
                weight = competency_weights.get(comp_name, 0)
                job_total += score * weight
                job_weight_sum += weight
        
        job_score = round(job_total / job_weight_sum, 1) if job_weight_sum > 0 else 0.0
        
        # 공통 역량 점수
        common_total = 0.0
        common_weight_sum = 0.0
        
        for comp_name in self.COMMON_COMPETENCIES:
            if comp_name in aggregated_competencies:
                score = aggregated_competencies[comp_name].get("overall_score", 0)
                weight = competency_weights.get(comp_name, 0)
                common_total += score * weight
                common_weight_sum += weight
        
        common_score = round(common_total / common_weight_sum, 1) if common_weight_sum > 0 else 0.0
        
        return {
            "final_score": final_result.get("final_score"),
            "job_score": job_score,
            "common_score": common_score,
            "job_competencies": self.JOB_COMPETENCIES,
            "common_competencies": self.COMMON_COMPETENCIES
        }
    
    
    async def _regenerate_all_batch(
        self,
        aggregated_competencies: Dict,
        transcript: Dict
    ) -> Dict[str, Dict]:

        self.transcript = transcript
        self._transcript_data = transcript
    
        """
        배치: 모든 역량의 근거/강점/약점/관찰을 1번의 LLM 호출로 재생성
        """
        
        # 1. 모든 역량 데이터 수집
        all_competencies_data = []
        
        for comp_name, comp_data in aggregated_competencies.items():
            perspectives = comp_data.get("perspectives", {})
            evidence_details = perspectives.get("evidence_details", [])

            for ev in evidence_details:
                segment_id = ev.get("segment_id")
                char_index = ev.get("char_index")
                
                # char_length 확인 및 계산
                if "char_length" not in ev or ev["char_length"] is None:
                    ev["char_length"] = len(ev.get("text", ""))
                
                # Transcript에서 실제 텍스트 추가
                if segment_id and char_index is not None:
                    actual_text = self._get_transcript_text(segment_id, char_index)
                    ev["actual_transcript_text"] = actual_text

            all_competencies_data.append({
                "competency_name": comp_name,
                "competency_display_name": self.COMPETENCY_DISPLAY_NAMES.get(comp_name, comp_name),
                "overall_score": comp_data.get("overall_score", 0),
                
                # Evidence 데이터
                "evidence_details": perspectives.get("evidence_details", []),
                "evidence_reasoning": perspectives.get("evidence_reasoning", ""),
                
                # Behavioral 데이터
                "behavioral_pattern": perspectives.get("behavioral_pattern", {}),
                "behavioral_reasoning": perspectives.get("behavioral_reasoning", ""),
                
                # Critical 데이터
                "red_flags": perspectives.get("red_flags", []),
                "critical_reasoning": perspectives.get("critical_reasoning", ""),
                
                # 원본 Strengths/Weaknesses/Key_observations
                "original_strengths": comp_data.get("strengths", []),
                "original_weaknesses": comp_data.get("weaknesses", []),
                "original_key_observations": comp_data.get("key_observations", [])
            })
        
        # 1.5. transcript에서 실제 텍스트 추가
        for comp_data in all_competencies_data:
            evidence_details = comp_data["evidence_details"]
            
            for ev in evidence_details:
                segment_id = ev.get("segment_id")
                char_index = ev.get("char_index")
                
                # transcript에서 실제 텍스트 가져오기
                if segment_id and char_index is not None:
                    actual_text = self._get_transcript_text(segment_id, char_index)
                    ev["actual_transcript_text"] = actual_text
        
        # 2. 배치 프롬프트 생성
        prompt = self._build_comprehensive_batch_prompt(all_competencies_data)
        
        # 3. LLM 1회 호출
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at synthesizing competency evaluation data into clear, professional summaries for HR reports."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=12000,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content.strip()
            result = json.loads(result_text)
            
            # 4. 역량별로 파싱
            batch_by_competency = {}
            
            for comp_result in result.get("competencies", []):
                comp_name = comp_result.get("competency_name")
                batch_by_competency[comp_name] = {
                    "evidences": comp_result.get("evidences", []),
                    "strengths": comp_result.get("strengths", []),
                    "weaknesses": comp_result.get("weaknesses", []),
                    "key_observations": comp_result.get("key_observations", [])
                }
            
            return batch_by_competency
        
        except Exception as e:
            print(f"        배치 LLM 호출 실패: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: 원본 데이터 그대로 반환
            return self._fallback_all_batch(aggregated_competencies)
    
    
    def _build_comprehensive_batch_prompt(
        self,
        all_competencies_data: List[Dict]
    ) -> str:
        """
        종합 배치 재생성 프롬프트 (근거 + 강점 + 약점 + 관찰)
        """
        
        template = """# Task: 10개 역량 평가 데이터 종합 배치 재생성

입력 데이터(JSON):
__COMPETENCY_DATA__

역할: HR 평가 보고서 작성 전문가. 근거/강점/약점/관찰을 한 번에 평서문으로 재작성.

핵심 규칙:
- JSON만 반환, markdown 금지.
- evidences는 segment_id/char_index/char_length를 **입력 값 그대로** 사용 (임의 생성/수정 금지). 없으면 null/0 그대로.
- evidences 개수: 최소 2, 최대 8. evidence_details → positive, red_flags → negative.
- strengths/weaknesses/key_observations: 각 항목 1문장. strengths 3~6, weaknesses 2~4, observations 3~5.
- 점수 숫자 노출 금지, 서술형으로 표현.

Segment 정보 우선순위:
1) evidence_details 값 그대로
2) red_flags.evidence_reference에서 segment_id/char_index 파싱
3) behavioral/critical에서 추론 시 char_index=0, char_length는 문장 길이 추정(30~120자)

char_length 가이드:
- 원본 값 있으면 그대로 사용
- 추론 시 핵심 문장 길이로 30~120자 범위

Evidences 작성:
- summary 2-3문장, 존댓말, 중복 서술 피하기
- impact: positive/negative/neutral
- evidence_type: 간단한 키워드(예: "주도성", "도전 회피")

강점/약점/관찰 작성:
- 각 1문장, 점수 언급 금지, 자연스러운 평서문

출력 형식(JSON):
{
  "competencies": [
    {
      "competency_name": "achievement_motivation",
      "evidences": [
        {
          "summary": "지원자는 교수님께 직접 연구 프로젝트를 제안하며 시작했습니다. 이는 주도성과 내적 동기를 보여줍니다.",
          "segment_id": 3,
          "char_index": 1200,
          "char_length": 45,
          "impact": "positive",
          "evidence_type": "주도성"
        },
        {
          "summary": "지원자는 일부 상황에서 쉬운 선택을 선호했다고 말했습니다. 도전 회피 성향이 엿보입니다.",
          "segment_id": 9,
          "char_index": 3450,
          "char_length": 30,
          "impact": "negative",
          "evidence_type": "도전 회피"
        }
      ],
      "strengths": ["자발적으로 과제를 제안하고 주도합니다.", "도전적인 목표를 끝까지 수행합니다."],
      "weaknesses": ["어려운 선택을 회피하는 경향이 일부 관찰됩니다."],
      "key_observations": ["신입 기준 상위권의 주도성을 보입니다."]
    }
  ]
}
"""
        
        prompt = template.replace(
            "__COMPETENCY_DATA__",
            json.dumps(all_competencies_data, ensure_ascii=False, indent=2)
        )

        return prompt

    def _get_transcript_text(self, segment_id: int, char_index: int) -> str:
        """
        transcript에서 segment 답변 텍스트 일부를 반환.
        """
        
        segments = self._transcript_data.get("segments", []) if isinstance(self._transcript_data, dict) else []
        segment = next((s for s in segments if s.get("segment_id") == segment_id), None)
        if not segment:
            return ""
        
        answer_text = segment.get("answer_text", "")
        if not isinstance(answer_text, str):
            return ""
        
        if isinstance(char_index, int) and 0 <= char_index < len(answer_text):
            return answer_text[char_index: char_index + 120]
        
        return answer_text

    async def _add_connected_summaries(
        self,
        competency_details: Dict
    ) -> Dict:
        """
        각 역량의 evidences.summary를 자연스럽게 연결한 문단 생성
        """
        
        for comp_name, comp_data in competency_details.items():
            evidences = comp_data.get("evidences", [])
            if not evidences:
                comp_data["connected_summary"] = ""
                continue
            
            individual_summaries = [ev.get("summary", "") for ev in evidences]
            connected = await self._connect_summaries_naturally(
                individual_summaries,
                comp_data.get("competency_display_name", comp_name)
            )
            comp_data["connected_summary"] = connected
        
        return competency_details


    async def _connect_summaries_naturally(
        self,
        summaries: List[str],
        competency_name: str
    ) -> str:
        """
        여러 summary를 하나의 자연스러운 문단으로 연결
        """
        
        if not summaries:
            return ""
        bullet_block = "\n".join(f"{i+1}. {s}" for i, s in enumerate(summaries))
        prompt = f"""다음은 "{competency_name}" 역량에 대한 여러 개의 근거 문장입니다.
이것들을 하나의 자연스러운 문단으로 연결해주세요.

규칙:
1. "지원자는"을 반복하지 말고, 한 번만 사용하거나 생략
2. 접속사(또한, 그리고, 아울러)로 자연스럽게 연결
3. 존댓말 유지 (했습니다 체)
4. 핵심 내용은 모두 포함
5. 3-5문장으로 구성
6. Negative 근거는 "다만," 또는 "그러나"로 구분

개별 근거들:
{bullet_block}

연결된 문단:"""

        try:
            response = await self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at writing natural, flowing Korean prose."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.summary_model,
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"        Summary 연결 실패: {e}")
            return " ".join(summaries)

    def _fallback_all_batch(
        self,
        aggregated_competencies: Dict
    ) -> Dict[str, Dict]:
        """
        배치 LLM 실패 시 Fallback (원본 데이터 그대로)
        """
        
        fallback = {}
        
        for comp_name, comp_data in aggregated_competencies.items():
            perspectives = comp_data.get("perspectives", {})
            evidence_details = perspectives.get("evidence_details", [])
            red_flags = perspectives.get("red_flags", [])
            comp_evidences = []
            
            # Evidence details
            for ev in evidence_details:
                comp_evidences.append({
                    "summary": f"지원자는 다음과 같은 행동을 보였습니다: {ev.get('text', '')}",
                    "segment_id": ev.get("segment_id"),
                    "char_index": ev.get("char_index"),
                    "char_length": 50,
                    "impact": "positive",
                    "evidence_type": ev.get("relevance_note", "")
                })
            
            # Red flags
            for flag in red_flags:
                seg_id = self._extract_segment_id(flag.get("evidence_reference", ""))
                char_idx = self._extract_char_index(flag.get("evidence_reference", ""))
                
                comp_evidences.append({
                    "summary": flag.get("description", ""),
                    "segment_id": seg_id,
                    "char_index": char_idx,
                    "char_length": 30,
                    "impact": "negative",
                    "evidence_type": flag.get("flag_type", "")
                })
            
            fallback[comp_name] = {
                "evidences": comp_evidences,
                "strengths": comp_data.get("strengths", []),
                "weaknesses": comp_data.get("weaknesses", []),
                "key_observations": comp_data.get("key_observations", [])
            }
        
        return fallback
    
    
    @staticmethod
    def _extract_segment_id(evidence_reference: str) -> int:
        """evidence_reference에서 segment_id 추출"""
        try:
            if "segment_id:" in evidence_reference:
                segment_part = evidence_reference.split("segment_id:")[1].split(",")[0].strip()
                return int(segment_part)
        except Exception:
            pass
        return None
    
    
    @staticmethod
    def _extract_char_index(evidence_reference: str) -> int:
        """evidence_reference에서 char_index 추출"""
        try:
            if "char_index:" in evidence_reference:
                char_part = evidence_reference.split("char_index:")[1].split("-")[0].strip()
                return int(char_part)
        except Exception:
            pass
        return None
