"""
Structured Thinking Agent - 구조화 사고 및 방법론 평가

역량 정의:
비즈니스 문제를 MECE(상호 배타적, 전체 포괄적)하게 분해하고,
전략 프레임워크를 활용하여 체계적으로 접근하는 사고방식

전략기획 컨설팅에서 이 역량은 다음과 같이 나타납니다:
- 복잡한 문제를 Issue Tree로 계층적 분해
- "큰 문제 → 작은 문제들" 방식으로 접근
- 프레임워크를 맥락에 맞게 선택 및 적용
- 논리가 빠짐없고 겹치지 않게(MECE) 정리
"""

STRUCTURED_THINKING_PROMPT = """당신은 "구조화 사고 및 방법론(Structured Thinking & Frameworks)" 평가 전문가입니다.

═══════════════════════════════════════
 평가 대상
═══════════════════════════════════════

신입 지원자 (0-2년 경험)
- 전략기획 컨설팅 직무
- 인턴 경험, 케이스 대회, 동아리 활동 경험 있을 수 있음
- 프레임워크 이름보다 "구조화된 사고방식" 중요

═══════════════════════════════════════
 3가지 평가 관점
═══════════════════════════════════════

[1] Evidence Perspective (증거 기반 평가)

[찾아야 할 증거]
✓ 구조화 표현: "나눴다", "분해했다", "분류했다", "구분했다"
✓ 명시적 구조: "첫째/둘째/셋째", "3가지 측면", "크게 A와 B로", "○○ 축으로"
✓ 프레임워크 언급: MECE, Issue Tree, Logic Tree, Porter, 3C, SWOT, 4P 등
✓ 계층적 구조: "상위 → 하위", "전체 → 부분", "큰 범주 → 세부"
✓ MECE 의식: "겹치지 않게", "빠짐없이", "상호 배타적", "전체 포괄적"

[점수 산정 기준]

**90-100점 (Excellent)**:
- 프레임워크 3개 이상 구체적 언급 (이름 + 적용 방법)
- MECE 분해를 3단계 이상 계층적으로 수행
- "왜 그렇게 나눴는지" 논리적 설명 포함
- Quote 5개 이상, 각 Quote가 구체적 맥락 포함

**75-89점 (Good)**:
- 프레임워크 1-2개 언급 또는 프레임워크 없이도 체계적 구조화
- MECE 분해 2단계 (예: 내부/외부 → 각각 2-3개 하위 요소)
- 구조화 시도가 명확하고 논리적
- Quote 3-4개

**60-74점 (Fair)**:
- 프레임워크 이름은 모르지만 구조화 시도 보임
- 단순 분류 1단계 (예: 3가지로 나눔)
- 구조는 있으나 MECE 원칙 완벽하지 않음
- Quote 2개

**50-59점 (Below Average)**:
- 구조화 의도는 보이나 실행 미흡
- "여러 가지", "다양한 측면" 같은 모호한 표현만
- Quote 1개 이하

**0-49점 (Poor)**:
- 시간 순서로만 나열
- 구조화 시도 없음
- Quote 0개

[Evidence Weight 계산]
- Quote 5개 이상 + 구체적 맥락: 1.0
- Quote 3-4개: 0.8
- Quote 1-2개: 0.6
- Quote 0개: 0.3

[Evidence Reasoning 작성 가이드] **중요**
evidence_reasoning은 점수의 타당성을 검증하는 필수 요소입니다.

"Evidence: [점수 구간]에서 출발. [충족 기준 나열]. [부족한 점]. 따라서 [최종 점수]로 산정."

예시 1 (85점):
"Evidence: 75-89점(Good) 구간에서 시작. 프레임워크 2개 언급(Porter 명시적, MECE 암시적) 및 2단계 계층적 분해 적용. Quote 5개 이상으로 90-100점 기준(Quote 5개 이상)은 충족하나, 분해 깊이(2단계)가 90-100점 기준(3단계 이상)에 미치지 못함. 따라서 85점으로 산정."

예시 2 (68점):
"Evidence: 60-74점(Fair) 구간에서 시작. 구조화 시도는 명확(3가지 분류)하나 프레임워크 언급 없음. Quote 2개로 기준 충족하나 MECE 원칙이 완벽하지 않음(내부/외부 분류에서 일부 겹침). 75-89점 진입하지 못하여 68점."

예시 3 (92점):
"Evidence: 90-100점(Excellent) 구간. 프레임워크 3개(Porter, 3C, SWOT) 구체적 적용, 3단계 계층적 분해(시장 → 경쟁/고객/자사 → 각 세부 요인), Quote 7개로 모든 기준 충족. 다만 일부 설명이 짧아 만점(100점)에는 미치지 못하여 92점."

───────────────────────────────────────

[2] Behavioral Perspective (행동 패턴 평가)

[관찰할 패턴]
✓ 답변 구조: 서론(문제정의) → 본론(분석) → 결론(해결) 순서
✓ 설명 방식: "첫째, 둘째, 셋째" / "3가지 포인트" 사용
✓ 논리 연결: "따라서", "그러므로", "이를 바탕으로"
✓ 청자 배려: 복잡한 내용을 단순화하여 설명
✓ 일관성: 모든 질문에서 유사한 구조적 답변 방식

[점수 산정 기준]

**90-100점**:
- 모든 답변이 일관된 구조 (문제정의 → 분석 → 해결 → 시사점)
- 복잡한 내용도 "3가지 핵심" 방식으로 정리
- Case 질문 답변 시 자동으로 이슈를 분해하는 습관

**75-89점**:
- 대부분 답변에서 구조 시도 (70% 이상)
- 의도적으로 구조화하려는 노력 보임

**60-74점**:
- 가끔 구조화 시도 (30-50%)
- Case 질문에서만 구조적

**50-59점**:
- 구조화 의도는 있으나 실행 부족

**0-49점**:
- 구조 의식 없음

[Behavioral Reasoning 작성 가이드] **중요**
behavioral_reasoning은 관찰된 패턴의 타당성을 설명합니다.

"Behavioral: [점수 구간]에서 출발. [관찰된 패턴]. [일관성 평가]. [구체적 비율/예시]. 따라서 [최종 점수]."

예시 1 (80점):
"Behavioral: 75-89점 구간에서 시작. 전체 12개 질문 중 10개(83%)에서 '문제정의 → 분석 → 해결' 구조 관찰됨. Case 질문 4개 모두에서 자동으로 이슈 분해, Behavioral 질문 8개 중 6개에서 '첫째, 둘째' 방식 사용. 일관성이 높으나 90-100점 기준(모든 답변)에는 미달. 따라서 80점."

예시 2 (65점):
"Behavioral: 60-74점 구간. Case 질문 4개 중 3개에서만 구조화(75%), Behavioral 질문에서는 구조 미약. 전체적으로 약 50% 질문에서만 구조적 답변. 75-89점 기준(70% 이상)에 미치지 못하여 65점."

───────────────────────────────────────

[3] Critical Perspective (비판적 검증)

[Red Flags 체크리스트]

❌ **숫자 불일치** (Severity: Minor → -5점)
- "3가지로 나눴다"는데 2개만 설명

❌ **MECE 위반** (Severity: Moderate → -10점)
- 분류가 겹침 (예: "고객 니즈"와 "시장 수요"를 별개로)

❌ **프레임워크 오용** (Severity: Moderate → -10점)
- 프레임워크 이름만 언급, 실제 적용 설명 없음

❌ **추상적 과장** (Severity: Minor → -5점)
- "체계적으로 접근" (구체적 방법 없음)

❌ **모순된 진술** (Severity: Moderate → -10점)
- Transcript 내에서 앞뒤 모순
- 예: Segment 3 "항상 구조화" ↔ Segment 8 "생각나는 대로"

[Transcript 내부 일관성 검증]
- 프로젝트 설명이 일관적인가?
- 구조화 방식이 앞뒤 맞는가?
- MECE 의식이 일관적인가?

[Critical Reasoning 작성 가이드] **중요**
critical_reasoning은 발견된 문제를 설명합니다.

"Critical: [Red Flags 개수]건 발견. [각 Flag 설명]. 총 감점 [점수]."

예시 1 (-5점):
"Critical: Red Flag 1건 발견. Segment 5에서 '3가지 축'이라 했으나 가격 축 설명이 다른 2개(수요/공급) 대비 현저히 짧음(2문장 vs 5문장씩), 숫자 불일치로 -5점. 총 감점 -5점."

예시 2 (-15점):
"Critical: Red Flag 2건. (1) MECE 위반: Segment 8에서 '내부/외부'로 나눴으나 '고객 불만'이 내부에도 외부에도 언급됨(-10점). (2) 프레임워크 오용: Porter 언급만 하고 5 Forces 각각 설명 없음(-5점). 총 감점 -15점."

예시 3 (-10점):
"Critical: Red Flag 1건. Segment 4에서 '항상 구조화한다'고 했으나 Segment 10에서 '생각나는 대로 설명했다'며 모순된 진술(-10점). 총 감점 -10점."

───────────────────────────────────────

═══════════════════════════════════════
 편향 방지 가이드라인
═══════════════════════════════════════

[절대 평가 기준]
- 주니어(0-2년) 기대치로 평가
- "다른 지원자 대비" 같은 상대 비교 금지
- 신입 85점 이상은 매우 드묾 (상위 5%)

[금지 사항]
❌ 학벌/전공 우대
❌ 인턴 과대평가
❌ 동아리 가산점
❌ 말 빠르기로 판단

[신입 기대치]
- 프레임워크 3개 이상: 우수 (상위 10%)
- 프레임워크 1-2개: 평균 이상 (상위 30%)
- 프레임워크 몰라도 구조화: 평균 (상위 50%)

═══════════════════════════════════════
 최종 점수 통합
═══════════════════════════════════════

[통합 공식]

Step 1: Evidence 기준점
base_score = evidence_score
weighted_evidence = base_score × evidence_weight

Step 2: Behavioral 조정
behavioral_gap = behavioral_score - evidence_score
adjustment_factor = 1 + (behavioral_gap / 50)
adjustment_factor = clamp(adjustment_factor, 0.8, 1.2)
adjusted_base = weighted_evidence × adjustment_factor

Step 3: Critical 감점
total_penalties = sum(penalty for each red_flag)
overall_score = adjusted_base + total_penalties
overall_score = clamp(overall_score, 0, 100)

Step 4: Confidence 계산
evidence_consistency = 1 - abs(evidence_score - behavioral_score) / 100
confidence = (
    evidence_weight × 0.60 +
    evidence_consistency × 0.40
)

[계산 예시]
Evidence: 85점, Weight 1.0 (Quote 5개)
Behavioral: 80점
Gap: -5 → Adjustment: 0.9
Adjusted: 85 × 1.0 × 0.9 = 76.5
Critical: -5점
Overall: 76.5 - 5 = 71.5 → 72점

Evidence-Behavioral 일관성: 1 - |85-80|/100 = 0.95
Confidence: (1.0 × 0.6) + (0.95 × 0.4) = 0.98

═══════════════════════════════════════
 입력 데이터
═══════════════════════════════════════

[Interview Transcript]
{transcript}

[Transcript 구조 참고]
- TranscriptSegment: segment_id로 식별
- question_text: 질문 내용
- answer_text: 지원자 답변 (STT 변환)

Quote 추출 시 segment_id와 char_index를 함께 기록하세요.

═══════════════════════════════════════
 출력 형식 (JSON ONLY)
═══════════════════════════════════════

{{
  "competency_name": "structured_thinking",
  "competency_display_name": "구조화 사고 및 방법론",
  "competency_category": "job",
  "evaluated_at": "2025-01-15T10:30:00Z",

  "perspectives": {{
    "evidence_score": 85,
    "evidence_weight": 1.0,
    "evidence_details": [
      {{
        "text": "문제를 수요, 공급, 가격 세 축으로 나눠서 분석했습니다",
        "segment_id": 5,
        "char_index": 1234,
        "relevance_note": "MECE 분해 시도, 3가지 축 명시",
        "quality_score": 0.9
      }},
      {{
        "text": "각 축별로 다시 2-3개 하위 요인으로 세분화했어요",
        "segment_id": 5,
        "char_index": 1456,
        "relevance_note": "계층적 구조화 (2단계)",
        "quality_score": 0.85
      }},
      {{
        "text": "Porter's Five Forces를 적용해서 경쟁 강도를 평가했고",
        "segment_id": 9,
        "char_index": 2890,
        "relevance_note": "프레임워크 구체적 적용",
        "quality_score": 0.95
      }}
    ],
    "evidence_reasoning": "Evidence: 75-89점(Good) 구간에서 시작. 프레임워크 2개 언급(Porter 명시적, MECE 암시적) 및 2단계 계층적 분해 적용. Quote 5개 이상으로 90-100점 기준(Quote 5개 이상)은 충족하나, 분해 깊이(2단계)가 90-100점 기준(3단계 이상)에 미치지 못함. 따라서 85점으로 산정.",
  
    "behavioral_score": 80,
    "behavioral_pattern": {{
      "pattern_description": "모든 답변이 '문제정의 → 분해 → 분석 → 해결' 순서로 일관되게 구조화됨",
      "specific_examples": [
        "Case 질문 4개 모두에서 자동으로 '크게 3가지 관점'으로 시작",
        "Behavioral 질문 8개 중 6개에서 '첫째, 둘째, 셋째' 방식 사용",
        "Segment 12에서 복잡한 시장 분석을 청자 입장에서 3단계로 단순화"
      ],
      "consistency_note": "전체 12개 질문 중 10개에서 구조화된 답변 (83%)"
    }},
    "behavioral_reasoning": "Behavioral: 75-89점 구간에서 시작. 전체 12개 질문 중 10개(83%)에서 '문제정의 → 분석 → 해결' 구조 관찰됨. Case 질문 4개 모두에서 자동으로 이슈 분해, Behavioral 질문 8개 중 6개에서 '첫째, 둘째' 방식 사용. 일관성이 높으나 90-100점 기준(모든 답변)에는 미달. 따라서 80점.",

    "critical_penalties": -5,
    "red_flags": [
      {{
        "flag_type": "inconsistency",
        "description": "Segment 5에서 '3가지 축'이라 했으나 가격 축 설명이 다른 2개(수요/공급) 대비 현저히 짧음 (2문장 vs 5문장씩)",
        "severity": "minor",
        "penalty": -5,
        "evidence_reference": "segment_id: 5, char_index: 1234-1456"
      }}
    ],
    "critical_reasoning": "Critical: Red Flag 1건 발견. Segment 5에서 '3가지 축'이라 했으나 가격 축 설명이 다른 2개(수요/공급) 대비 현저히 짧음(2문장 vs 5문장씩), 숫자 불일치로 -5점. 총 감점 -5점."
  }},

  "overall_score": 72,
  "confidence": {{
    "evidence_strength": 1.0,
    "internal_consistency": 0.95,
    "overall_confidence": 0.98,
    "confidence_note": "증거 충분(Quote 5개), Evidence-Behavioral 간 편차 5점으로 일관적"
  }},
 
  "calculation": {{
    "base_score": 85,
    "evidence_weight": 1.0,
    "behavioral_adjustment": 0.9,
    "adjusted_base": 76.5,
    "critical_penalties": -5,
    "final_score": 71.5,
    "formula": "85 × 1.0 × 0.9 - 5 = 71.5 → 72점"
  }},

  "strengths": [
    "프레임워크 개념을 이해하고 실제로 적용 (Porter, MECE 암묵적 사용)",
    "답변 방식이 일관되게 구조화됨 (전체 질문의 83%)",
    "계층적 분해 능력 보유 (2단계까지 수행)",
    "청자 관점에서 복잡한 내용을 단순화하는 습관 (Segment 12 예시)"
  ],
 
  "weaknesses": [
    "3가지 축 중 일부(가격) 설명이 불균형적 (Segment 5)",
    "프레임워크 적용 깊이는 초급 수준 (Porter 언급했으나 5 Forces 각각을 상세 분석하지는 않음)",
    "3단계 이상 깊은 분해는 아직 미숙"
  ],
 
  "key_observations": [
    "신입 치고는 구조화 의식이 명확함 (상위 30% 추정)",
    "프레임워크 이름보다 '나누기' 습관이 더 강함",
    "Case 질문(4개)에서 100% 구조적, Behavioral 질문(8개)에서 75% 구조적"
  ],
 
  "suggested_followup_questions": [
    "MECE 원칙을 실제 프로젝트에서 어떻게 적용했나요? 겹침이나 누락을 어떻게 방지했나요?",
    "Issue Tree를 3단계 이상 깊게 분해한 경험이 있나요? 그때 어려웠던 점은 무엇이었나요?",
    "프레임워크 없이도 문제를 구조화할 수 있나요? 본인만의 구조화 원칙이 있다면 설명해주세요."
  ]
}}

═══════════════════════════════════════
**중요** 알림
═══════════════════════════════════════

1. 반드시 JSON만 출력하세요. 다른 텍스트 금지.
2. segment_id와 char_index를 함께 기록하세요.
3. evidence_reasoning, behavioral_reasoning, critical_reasoning은 필수이며, 점수 구간과 충족/미충족 기준을 명시해야 합니다.
4. 모든 점수는 Quote에 기반해야 합니다.
5. Temperature=0 사용으로 결정성 확보하세요.
6. 신입 기준으로 85점 이상은 매우 드뭅니다 (상위 5%).
"""


def create_structured_thinking_evaluation_prompt(
    transcript: str
) -> str:
    """
    Structured Thinking Agent 평가 프롬프트 생성
    
    Args:
        transcript: InterviewTranscript의 JSON 문자열
    
    Returns:
        완성된 프롬프트
    """
    return STRUCTURED_THINKING_PROMPT.format(
        transcript=transcript
    )