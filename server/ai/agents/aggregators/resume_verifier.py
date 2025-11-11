"""
Resume Verifier
10개 Agent의 모든 Segment 평가를 Resume와 비교 (AI 1회 Batch)
"""

import json
from typing import Dict, List, Any
from openai import AsyncOpenAI


class ResumeVerifier:
    """
    Resume 검증기
    
    역할:
        - 10개 Agent의 모든 Segment 평가를 Resume와 비교
        - AI 1회 호출로 Batch 처리
        - 각 Segment 평가에 Resume 검증 결과 추가:
            * resume_verified: bool
            * verification_strength: "high"/"medium"/"low"/"none"
            * resume_evidence: List[str]
    """
    
    def __init__(self, openai_client: AsyncOpenAI):
        self.client = openai_client
    
    
    async def verify_batch(
        self,
        all_competency_results: Dict[str, Dict],
        resume_data: Dict
    ) -> List[Dict]:
        """
        Batch Resume 검증
        
        Args:
            all_competency_results: 10개 역량 평가 결과
                {
                    "achievement_motivation": {...},
                    "growth_potential": {...},
                    ...
                }
            resume_data: 파싱된 Resume JSON
                {
                    "education": [...],
                    "experience": [...],
                    "projects": [...],
                    "skills": [...]
                }
        
        Returns:
            segment_evaluations_with_resume: Resume 검증 추가된 Segment 평가 목록
                [
                    {
                        "competency": "achievement_motivation",
                        "segment_id": 3,
                        "score": 85,
                        "quotes": [...],
                        "interview_confidence": 0.85,
                        "resume_verified": true,
                        "verification_strength": "high",
                        "resume_evidence": ["학부생 창업 경험", ...]
                    },
                    ...
                ]
        """
        
        # 1. Segment 평가 목록 추출
        segment_evaluations = self._extract_segment_evaluations(all_competency_results)
        
        print(f"\n[Resume Verifier] Segment 평가 추출: {len(segment_evaluations)}개")
        
        # Resume 데이터 없으면 검증 스킵
        if not resume_data:
            print("  Resume 데이터 없음 - 검증 스킵")
            return self._add_empty_verification(segment_evaluations)
        
        
        # 2. AI 프롬프트 생성
        prompt = self._build_verification_prompt(segment_evaluations, resume_data)
        
        
        # 3. AI 호출 (1회)
        print("[Resume Verifier] AI 호출 시작 (Batch)...")
        
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a resume verification expert. Verify if interview segment evaluations are supported by the candidate's resume."
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
        print(f"[Resume Verifier] AI 응답 길이: {len(result_text)} chars")
        
        
        # 4. 결과 파싱
        try:
            verification_results = json.loads(result_text)
            verified_list = verification_results.get("verifications", [])
            
            print(f"[Resume Verifier] 검증 완료: {len(verified_list)}개")
            
            # 원본 Segment 평가와 병합
            merged = self._merge_verification_results(
                segment_evaluations,
                verified_list
            )
            
            return merged
        
        except json.JSONDecodeError as e:
            print(f"  JSON 파싱 실패: {e}")
            return self._add_empty_verification(segment_evaluations)
    
    
    def _extract_segment_evaluations(
        self,
        all_competency_results: Dict[str, Dict]
    ) -> List[Dict]:
        """
        10개 역량 결과에서 Segment 평가 추출
        
        Returns:
            [
                {
                    "competency": "achievement_motivation",
                    "segment_id": 3,
                    "score": 85,
                    "quotes": [...],
                    "interview_confidence": 0.85
                },
                ...
            ]
        """
        segment_evals = []
        
        for comp_name, comp_result in all_competency_results.items():
            if not comp_result:
                continue
            
            # Evidence details에서 Segment 평가 추출
            perspectives = comp_result.get("perspectives", {})
            evidence_details = perspectives.get("evidence_details", [])
            overall_score = comp_result.get("overall_score", 0)
            overall_confidence = comp_result.get("confidence", {}).get("overall_confidence", 0.5)
            
            for detail in evidence_details:
                segment_eval = {
                    "competency": comp_name,
                    "segment_id": detail.get("segment_id"),
                    "quote_text": detail.get("text", ""),
                    "relevance_note": detail.get("relevance_note", ""),
                    "quality_score": detail.get("quality_score", 0.0),
                    "overall_score": overall_score,
                    "score": overall_score,
                    "interview_confidence": overall_confidence 
                }
                segment_evals.append(segment_eval)
        
        return segment_evals
    
    
    def _build_verification_prompt(
        self,
        segment_evaluations: List[Dict],
        resume_data: Dict
    ) -> str:
        """
        AI 검증 프롬프트 생성
        """
        prompt = f"""# Task: Resume Verification (Batch) with Detailed Reasoning

    당신은 면접 답변이 Resume 경력과 일치하는지 검증하는 전문가입니다.

    ## 역할:
    1. 면접에서 지원자가 언급한 경험/스킬이 Resume에 실제로 있는지 확인
    2. Resume의 어느 부분(education, experience, projects 등)과 매칭되는지 **구체적으로** 명시
    3. 검증 신뢰도(high/medium/low/none)를 판단하고 **그 근거를 상세히** 제시

    ## Resume Data:
    ```json
    {json.dumps(resume_data, ensure_ascii=False, indent=2)}
    ```

    ## Segment Evaluations to Verify:
    ```json
    {json.dumps(segment_evaluations, ensure_ascii=False, indent=2)}
    ```

    ## 검증 기준:

    **high (강한 검증):**
    - Resume에 구체적으로 명시됨 (프로젝트명, 역할, 기술 등)
    - 여러 섹션에서 일관되게 확인됨
    - 시간적으로 모순 없음

    **medium (중간 검증):**
    - Resume에 간접적으로 언급됨 (관련 경험은 있으나 구체적 내용 다름)
    - 일부 섹션에서만 확인됨
    - 시간적으로 약간의 불일치

    **low (약한 검증):**
    - Resume에 매우 간접적으로만 언급됨
    - 추론 가능하지만 직접 언급 없음

    **none (검증 실패):**
    - Resume에 전혀 관련 내용 없음
    - 시간적으로 명백한 모순

    ## 출력 형식 (JSON):
    {{
    "verifications": [
        {{
        "competency": "achievement_motivation",
        "segment_id": 3,
        "quote_text": "학부생 때 창업 프로젝트를 이끌었습니다",
        
        "resume_verified": true,
        "verification_strength": "high",
        
        "reasoning": "지원자는 Resume의 'projects' 섹션에 '학부생 창업 프로젝트 (2020.03-2021.02, 팀장)'가 명시되어 있음. 기간, 역할, 프로젝트명이 모두 일치하므로 검증 강도 high. 또한 'awards' 섹션에 '대학생 창업 공모전 대상 (2021.11)'이 있어 프로젝트 성과도 확인됨.",
        
        "resume_matches": [
            {{
            "resume_section": "projects",
            "matched_content": "학부생 창업 프로젝트 (2020.03-2021.02) | 팀장 | 패션 큐레이션 플랫폼 개발 및 운영",
            "relevance": "면접 답변의 '창업 프로젝트 이끌었다'와 직접 매칭. 역할(팀장)도 일치."
            }},
            {{
            "resume_section": "awards",
            "matched_content": "대학생 창업 공모전 대상 (2021.11, 중소벤처기업부)",
            "relevance": "프로젝트의 성과를 입증. 시간적으로도 프로젝트 종료 시점과 일치."
            }}
        ],
        
        "confidence_factors": {{
            "direct_evidence": true,
            "multiple_sources": true,
            "time_consistency": true,
            "detail_level": "high"
        }}
        }},
        {{
        "competency": "problem_solving",
        "segment_id": 1,
        "quote_text": "문제를 수요, 공급, 가격 세 축으로 나눠서 분석하는 MECE 방식을 선호합니다",
        
        "resume_verified": true,
        "verification_strength": "medium",
        
        "reasoning": "지원자는 Resume의 'education' 섹션에 '경영학 전공'이 있으며, MECE는 경영학 필수 프레임워크임. 그러나 Resume에 MECE를 직접 언급한 프로젝트는 없음. 'projects'의 '리테일 데이터 분석 공모전'에서 '시장 세분화' 경험이 있어 간접적으로 구조적 분석 능력 추정 가능. 직접 증거는 아니므로 medium.",
        
        "resume_matches": [
            {{
            "resume_section": "education",
            "matched_content": "서울대학교 경영학 전공 (2018.03-2022.02)",
            "relevance": "경영학 커리큘럼에 MECE, Issue Tree 등 구조적 분석 프레임워크 포함됨. 간접 증거."
            }},
            {{
            "resume_section": "projects",
            "matched_content": "리테일 데이터 분석 공모전 (2021.09-2021.11) | 데이터 분석 담당",
            "relevance": "데이터 분석 시 구조적 접근이 필요하나, MECE 직접 언급 없음."
            }}
        ],
        
        "confidence_factors": {{
            "direct_evidence": false,
            "multiple_sources": true,
            "time_consistency": true,
            "detail_level": "medium"
        }}
        }}
    ]
    }}

    ## 중요 지침:
    1. **reasoning은 반드시 구체적으로 작성**
    - Resume의 어느 섹션, 어떤 내용과 매칭되는지 명시
    - 왜 그 verification_strength인지 논리적 근거 제시
    - 시간적 일관성, 역할 일치 여부 등 언급

    2. **resume_matches는 1개 이상 필수**
    - 여러 섹션에서 확인되면 모두 나열
    - matched_content는 Resume 원문 그대로 인용
    - relevance는 면접 답변과의 관련성 설명

    3. **confidence_factors로 검증 신뢰도 근거 명시**
    - direct_evidence: Resume에 직접 언급 여부
    - multiple_sources: 여러 섹션에서 확인 여부
    - time_consistency: 시간적 모순 없음
    - detail_level: Resume 기술의 구체성 수준

    4. **Output ONLY valid JSON**
    - NO markdown code blocks
    - NO explanations outside JSON
    - ALL text in Korean

    """
        return prompt
    
    
    def _merge_verification_results(
        self,
        original_segments: List[Dict],
        verified_list: List[Dict]
    ) -> List[Dict]:
        """
        원본 Segment 평가 + Resume 검증 결과 병합
        """
        # Verification 결과를 dict로 변환 (빠른 조회)
        verification_map = {}
        for v in verified_list:
            key = (v.get("competency"), v.get("segment_id"))
            verification_map[key] = v
        
        # 병합
        merged = []
        for seg in original_segments:
            key = (seg.get("competency"), seg.get("segment_id"))
            verification = verification_map.get(key, {})

            # ✅ 기본 검증 플래그/강도 결정 (빈 매칭이면 검증 실패로 처리)
            resume_matches = verification.get("resume_matches", [])
            resume_verified = bool(verification.get("resume_verified", False)) and bool(resume_matches)
            verification_strength = verification.get("verification_strength", "none") if resume_verified else "none"
            
            merged_seg = {
                **seg,
                
                "resume_verification": {
                    "verified": resume_verified,
                    "strength": verification_strength,
                    "reasoning": verification.get("reasoning", ""),  # 🆕
                    "resume_matches": resume_matches,  # 🆕
                    "confidence_factors": verification.get("confidence_factors", {})  # 🆕
                }
            }
            merged.append(merged_seg)
        
        return merged
    
    
    def _add_empty_verification(
        self,
        segment_evaluations: List[Dict]
    ) -> List[Dict]:
        """
        Resume 검증 없이 빈 필드 추가
        """
        return [
            {
                **seg,
                "resume_verification": { 
                    "verified": False,
                    "strength": "none",
                    "reasoning": "Resume 데이터 없음",
                    "resume_matches": [],
                    "confidence_factors": {
                        "direct_evidence": False,
                        "multiple_sources": False,
                        "time_consistency": False,
                        "detail_level": "none"
                    }
                }
            }
            for seg in segment_evaluations
        ]
