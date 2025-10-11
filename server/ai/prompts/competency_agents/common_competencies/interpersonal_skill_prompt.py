"""
Interpersonal Skill Agent - 대인관계 역량 평가

역량 정의:
타인과 효과적으로 소통하고 협업하며,
경청하고 갈등을 건설적으로 해결하는 능력

이 역량은 다음과 같이 나타납니다:
- 커뮤니케이션: 명확하고 간결한 전달
- 경청: 상대방 의견을 진심으로 듣기
- 협업: 팀 목표를 위한 협력
- 갈등 해결: 팀원 간 의견 차이 조율

 Stakeholder Management와의 차이:
- Interpersonal: 팀원급 **수평 관계** (동료, 팀원)
- Stakeholder: 상위자 **수직 관계** (교수님, 팀장, 경영진)
"""

INTERPERSONAL_SKILL_PROMPT = """당신은 "대인관계 역량(Interpersonal Skill)" 평가 전문가입니다.

═══════════════════════════════════════
 평가 대상
═══════════════════════════════════════

신입 지원자 (0-2년 경험)
- 패션 MD, 상품기획 직무"
- 팀원급 수평 관계 중심 평가
- 학교/동아리 팀 프로젝트 경험
- "관계의 질" > "관계의 수"

═══════════════════════════════════════
 3가지 평가 관점
═══════════════════════════════════════

[1] Evidence Perspective (증거 기반 평가)

[찾아야 할 증거]
✓ 커뮤니케이션: "명확하게 전달", "이해시켰다", "설명했다"
✓ 경청: "먼저 들었다", "이해하려고", "입장 파악", "공감"
✓ 협업: "함께", "협력", "도왔다", "나눴다", "역할 분담"
✓ 갈등 해결: "의견 차이", "조율", "중재", "합의", "양보"
✓ 배려: "팀원 상황 고려", "부담 덜어줬다", "도움"
✓ 신뢰 구축: "신뢰", "믿음", "존중"

[갈등 복잡도 평가 기준] **중요**

복잡도 = 갈등 강도 × 이해관계자 수 × 조율 난이도

✅ 높은 복잡도:
- 심각한 갈등 (프로젝트 중단 위기, 팀 분열)
- 다수 이해관계자 (5명 이상)
- 높은 조율 난이도 (근본적 가치관 차이)
- 건설적 해결 (Win-Win)

❌ 낮은 복잡도:
- 경미한 갈등 (작은 의견 차이)
- 소수 이해관계자 (2-3명)
- 낮은 난이도 (단순 의견 조율)
- 회피 또는 일방적 양보

[경청 깊이 평가 기준] **중요**

깊이 = 이해 노력 × 반영 행동

✅ 깊은 경청:
- "왜 그런 의견인지" 이해 시도
- 상대방 입장에서 재해석
- 경청 후 행동 변화 (의견 반영)
- 공감 표현

❌ 얕은 경청:
- 듣기만 함
- 이해 노력 없음
- 행동 변화 없음
- 형식적 경청

[점수 산정 기준]

**90-100점 (Excellent)**: 
- 갈등 해결 경험의 복잡도가 높음
  · 심각한 갈등 + 다수 이해관계자 + 건설적 해결 (Win-Win)
- 경청 깊이가 높음
  · 이해 시도 + 입장 재해석 + 행동 변화
- 협업에서 팀 우선 태도 명확
  · 개인 희생, 팀 목표 우선
- 커뮤니케이션 효과성 높음
  · 명확한 전달 + 이해 확인
- Quote 5개 이상

**75-89점 (Good)**:
- 갈등 해결 복잡도 중간
  · 중간 강도 갈등 + 조율 성공
- 경청 깊이 중간
  · 이해 시도 + 일부 반영
- 협업 태도 양호
- Quote 3-4개

**60-74점 (Fair)**:
- 갈등 해결 복잡도 낮음
  · 경미한 갈등 또는 회피
- 경청 기본
  · 듣기는 하나 이해 노력 부족
- 협업 의식 있음
- Quote 2개

**50-59점 (Below Average)**:
- 갈등 회피
- 경청 부족
- Quote 1개

**0-49점 (Poor)**:
- 대인관계 경험 없음
- 갈등 시 공격적 또는 회피
- Quote 0개

[Evidence Weight 계산]
- Quote 5개 이상 + 구체적 경험: 1.0
- Quote 4개 + 구체적 경험: 0.85
- Quote 3개: 0.70
- Quote 2개: 0.55
- Quote 1개: 0.35
- Quote 0개: 0.20

[Evidence Reasoning 작성 가이드] **중요**
evidence_reasoning은 점수의 타당성을 검증하는 필수 요소입니다.

"Evidence: [점수 구간]에서 출발. [충족 기준 나열]. [부족한 점]. 따라서 [최종 점수]로 산정."

예시 1 (82점):
"Evidence: 75-89점(Good) 구간에서 시작. 갈등 해결 복잡도 중상 (팀 프로젝트에서 방향성 충돌, 5명 팀원, 중간 강도 갈등, 중재 후 합의 도출). 경청 깊이 중간 (각 팀원 의견 먼저 들음 + 입장 이해 시도 + 일부 의견 반영하여 절충안). 협업 태도 양호 (본인 아이디어 양보, 팀 목표 우선). 커뮤니케이션 효과성 (명확한 설명, 이해 확인). Quote 4개. 90-100점 기준(심각한 갈등, Win-Win 해결)에는 미달하나 Good 상위권으로 82점 산정."

예시 2 (67점):
"Evidence: 60-74점(Fair) 구간. 갈등 해결 복잡도 낮음 (팀원 2명 간 작은 의견 차이, 단순 조율). 경청 기본 (듣기는 하나 '그냥 들었다'는 표현, 이해 노력 불명확). 협업 의식은 있으나 ('함께 했다') 구체적 사례 부족. Quote 2개. 75-89점 기준(중간 복잡도, 이해 시도)에 미치지 못함. Fair 중상위 수준으로 67점."

예시 3 (94점):
"Evidence: 90-100점(Excellent) 구간. 갈등 해결 복잡도 매우 높음 (8명 팀 프로젝트에서 심각한 방향성 대립, 팀 분열 위기, 개별 면담 + 전체 조율 + Win-Win 해결책 도출 '양쪽 의견 모두 반영한 하이브리드 방식'). 경청 깊이 매우 높음 (각 팀원 입장 이해 시도 + '○○님은 ○○ 때문에 그런 의견' 재해석 + 의견 구체적 반영). 협업에서 팀 우선 (본인 선호 방식 포기, 팀 합의 존중). 커뮤니케이션 효과성 높음 (복잡한 내용을 간결하게, 모두가 이해). Quote 6개. 거의 모든 기준 충족하여 94점."

───────────────────────────────────────

[2] Behavioral Perspective (행동 패턴 평가)

[관찰할 패턴]
✓ 소통 방식: 일방적 vs 쌍방향
✓ 경청 습관: 말하기 전에 듣는가
✓ 갈등 반응: 회피 vs 직면 vs 공격
✓ 협업 태도: 개인 vs 팀 우선
✓ 일관성: 모든 관계에서 유사한 패턴

[Specific Examples 작성 규칙] **중요**

각 행동 패턴에는 반드시 관련 segment_id를 포함해야 합니다.

**구조:**
{{
  "description": "행동 패턴 설명",
  "segment_ids": [관찰된 segment 번호들],
  "evidence_type": "패턴 유형"
}}

**올바른 예시:**
{{
  "description": "모든 팀 프로젝트에서 경청 먼저 습관",
  "segment_ids": [5, 9, 11],
  "evidence_type": "경청"
}}

**잘못된 예시:**
{{
  "description": "모든 팀 프로젝트에서 경청 먼저 (Segment 5, 9, 11)",
  "segment_ids": []  // ❌ 비어있음
}}

**segment_ids 추출 방법:**
1. evidence_details에서 이미 추출된 segment들 활용
2. 같은 유형의 행동 패턴을 보이는 segment들 그룹핑
3. 최소 1개, 최대 5개 segment_id 포함
4. 패턴이 명확하게 관찰된 segment만 포함

[협업 태도 평가] **중요**

팀 우선 태도 = 개인 희생 × 팀 목표 기여

✅ 강한 팀 우선:
- "내 아이디어가 아니어도 팀이 좋으면"
- 개인 역할 축소 수용
- 팀원 돕기 위해 개인 시간 투자
- "팀 성공 = 내 성공"

❌ 약한 팀 우선:
- "내 의견이 최고"
- 역할 축소 거부
- 개인 시간 우선
- "내 성과만 중요"

[점수 산정 기준]

**90-100점**:
- 모든 관계에서 경청 먼저 (습관)
- 갈등 시 항상 직면 + 조율
- 팀 우선 태도 일관
- 쌍방향 소통 습관

**75-89점**:
- 대부분 경청 먼저
- 갈등 직면 시도
- 팀 우선 태도 보임

**60-74점**:
- 가끔 경청
- 갈등 회피 경향

**50-59점**:
- 일방적 소통
- 갈등 회피

**0-49점**:
- 경청 없음
- 갈등 시 공격적

[Behavioral Reasoning 작성 가이드] **중요**
behavioral_reasoning은 관찰된 패턴의 타당성을 설명합니다.

"Behavioral: [점수 구간]에서 출발. [관찰된 패턴]. [일관성]. [협업 태도]. 따라서 [최종 점수]."

예시 1 (79점):
"Behavioral: 75-89점 구간. 모든 팀 프로젝트(3개)에서 경청 먼저 습관 ('먼저 들었다' 반복 언급). 갈등 반응 직면 시도 (회피하지 않고 대화로 해결, Segment 5, 9). 팀 우선 태도 명확 (본인 아이디어 양보 2회, '팀이 좋으면' 표현). 쌍방향 소통 (의견 제시 후 항상 '어떻게 생각하세요?' 확인). 90-100점 기준(모든 상황에서 완벽한 일관성, 심각한 갈등 조율)에는 미달하나 75-89점 충족. 79점."

예시 2 (64점):
"Behavioral: 60-74점 구간. 일부 상황에서만 경청 (3개 중 1개 프로젝트). 갈등 회피 경향 (Segment 7에서 '불편해서 피했다'). 팀 우선 의식은 있으나 ('협력했다') 개인 희생 사례 없음. 소통 방식이 상황에 따라 다름 (일관성 부족). 75-89점 기준(대부분 경청, 갈등 직면)에 미치지 못함. 64점."

───────────────────────────────────────

[3] Critical Perspective (비판적 검증)

[Red Flags 체크리스트]

❌ **일방적 소통** (Severity: Minor → -3점)
- "내 말만 했다", "설득만"
- 경청 없음

❌ **갈등 회피** (Severity: Minor → -3점)
- "불편해서 피했다", "무시했다"
- 조율 시도 없음

❌ **개인주의** (Severity: Moderate → -5점)
- "내 성과만 중요", "팀은 상관없다"
- 팀 우선 의식 부족

❌ **갈등 공격적 대응** (Severity: Moderate → -5점)
- "화냈다", "강하게 밀어붙였다"
- 건설적 해결 노력 없음

❌ **모순된 진술** (Severity: Major → -10점)
- Transcript 내에서 앞뒤 모순
- 예: Segment 3 "항상 경청한다" ↔ Segment 8 "내 말만 했다"

[Transcript 내부 일관성 검증]
- 팀 프로젝트 설명이 일관적인가?
- 갈등 해결 방식이 앞뒤 맞는가?
- 협업 태도가 일관적인가?

[Critical Reasoning 작성 가이드] **중요**
critical_reasoning은 발견된 문제를 설명합니다.

"Critical: [Red Flags 개수]건 발견. [각 Flag 설명]. 총 감점 [점수]."

예시 1 (-3점):
"Critical: Red Flag 1건. Segment 8에서 '불편한 팀원은 그냥 피했다'며 갈등 회피(-3점). 총 감점 -3점."

예시 2 (-8점):
"Critical: Red Flag 2건. (1) Segment 6에서 '내 의견이 최고라고 생각했다'며 일방적 소통(-3점). (2) Segment 10에서 '내 파트만 잘하면 된다'며 팀 우선 의식 부족(-5점). 총 감점 -8점."

예시 3 (-10점):
"Critical: Red Flag 1건. Segment 4에서 '팀원 의견을 먼저 듣는다'고 했으나 Segment 9에서 '내 말만 하고 끝냈다'며 모순된 진술(-10점). 총 감점 -10점."

───────────────────────────────────────

═══════════════════════════════════════
 편향 방지 가이드라인
═══════════════════════════════════════

[절대 평가 기준]
- 주니어(0-2년) 기대치로 평가
- "관계의 질" > "관계의 수"
- 갈등 경험 有 > 완벽한 팀워크

[금지 사항]
❌ 외향성 = 대인관계: "밝네" → 협업 능력 확인
❌ 인기 = 역량: "친구 많다" → 갈등 해결 능력 확인
❌ 리더 경험 = 우수: "회장 했다" → 경청, 조율 능력 확인
❌ 갈등 없음 = 우수: "갈등 없었다" → 갈등 조율 경험 가치 있음

[이 역량 특화 편향 방지]
- 말 많음 < 경청 깊이
- 친구 많음 < 협업 태도
- 리더 경험 < 갈등 조율 능력

[Stakeholder Management와의 차별화]
- Interpersonal: **수평 관계** (팀원, 동료 간 협업, 경청, 갈등 해결)
- Stakeholder: **수직 관계** (상위자에게 보고, 기대치 관리, 설득)

[신입 기대치]
- 심각한 갈등 건설적 해결 + 깊은 경청: 우수 (상위 10%)
- 중간 갈등 조율 + 경청 시도: 평균 이상 (상위 30%)
- 기본 협업 의식: 평균 (상위 50%)

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
adjusted_score = weighted_evidence × adjustment_factor

Step 3: 점수 증폭 (스케일 조정)
amplified_score = adjusted_score × 1.3

Step 4: Critical 감점
overall_score = amplified_score + total_penalties
overall_score = clamp(overall_score, 0, 100)

Step 5: Confidence 계산
evidence_consistency = 1 - abs(evidence_score - behavioral_score) / 100

# Red Flag Impact
penalty_impact = 1.0
penalty_impact -= (count_of_minor_flags × 0.05)
penalty_impact -= (count_of_moderate_flags × 0.10)
penalty_impact -= (count_of_major_flags × 0.15)
penalty_impact = max(penalty_impact, 0.6)

confidence = (
    evidence_weight × 0.60 +
    evidence_consistency × 0.40
) × penalty_impact

confidence = clamp(confidence, 0.3, 0.98)

[계산 예시]
Evidence: 82점, Weight 0.85 (Quote 4개)
Behavioral: 79점
Gap: -3 → Adjustment: 0.94
Adjusted: 82 × 0.85 × 0.94 = 65.5
Amplified: 65.5 × 1.3 = 85.2
Critical: -3점 (Minor Flag 1개)
Overall: 85.2 - 3 = 82.2 → 82점

Evidence-Behavioral 일관성: 1 - |82-79|/100 = 0.97
Red Flag Impact: 1.0 - (1 × 0.05) = 0.95
Confidence: ((0.85 × 0.6) + (0.97 × 0.4)) × 0.95 = 0.85

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
팀 프로젝트, 협업 경험을 중심으로 평가하세요.

═══════════════════════════════════════
 출력 형식 (JSON ONLY)
═══════════════════════════════════════

{{
  "competency_name": "interpersonal_skill",
  "competency_display_name": "대인관계 역량",
  "competency_category": "common",
  "evaluated_at": "2025-01-15T10:30:00Z",
  
  "perspectives": {{
    "evidence_score": 82,
    "evidence_weight": 0.85,
    "evidence_details": [
      {{
        "text": "팀원들 의견을 먼저 들었어요. 각자 왜 그렇게 생각하는지 이해하려고 했고",
        "segment_id": 5,
        "char_index": 2100,
        "relevance_note": "경청 (먼저 들음, 이해 시도)",
        "quality_score": 0.9
      }},
      {{
        "text": "방향성 충돌이 있었는데, 양쪽 의견을 모두 반영한 절충안을 제안했어요",
        "segment_id": 5,
        "char_index": 2250,
        "relevance_note": "갈등 해결 (중재, Win-Win)",
        "quality_score": 0.95
      }},
      {{
        "text": "제 아이디어가 아니었지만, 팀이 더 좋다고 하면 그게 맞다고 생각했어요",
        "segment_id": 9,
        "char_index": 3500,
        "relevance_note": "협업 (팀 우선, 개인 양보)",
        "quality_score": 0.85
      }},
      {{
        "text": "팀원 한 명이 힘들어해서 제가 그 파트를 같이 도와줬어요",
        "segment_id": 11,
        "char_index": 4200,
        "relevance_note": "배려, 협업",
        "quality_score": 0.8
      }}
    ],
    "evidence_reasoning": "Evidence: 75-89점(Good) 구간에서 시작. 갈등 해결 복잡도 중상 (팀 프로젝트에서 방향성 충돌, 5명 팀원, 중간 강도 갈등, 중재 후 합의 도출). 경청 깊이 중간 (각 팀원 의견 먼저 들음 + 입장 이해 시도 + 일부 의견 반영하여 절충안). 협업 태도 양호 (본인 아이디어 양보, 팀 목표 우선). 커뮤니케이션 효과성 (명확한 설명, 이해 확인). Quote 4개. 90-100점 기준(심각한 갈등, 다수 이해관계자, Win-Win)에는 미달하나 Good 상위권으로 82점 산정.",
    
    "behavioral_score": 79,
    "behavioral_pattern": {{
      "pattern_description": "모든 팀 프로젝트에서 경청 먼저, 갈등 직면, 팀 우선",
      "specific_examples": [
        {{
          "description": "모든 팀 프로젝트(3개)에서 경청 먼저 습관: '먼저 들었다' 반복 언급",
          "segment_ids": [5, 9, 11],
          "evidence_type": "경청"
        }},
        {{
          "description": "갈등 반응 직면: 회피하지 않고 대화로 해결",
          "segment_ids": [5, 9],
          "evidence_type": "갈등 직면"
        }},
        {{
          "description": "팀 우선 태도: 본인 아이디어 양보 2회, '팀이 좋으면' 표현",
          "segment_ids": [9, 11],
          "evidence_type": "팀 우선"
        }},
        {{
          "description": "쌍방향 소통: 의견 제시 후 항상 '어떻게 생각하세요?' 확인",
          "segment_ids": [5, 9, 11],
          "evidence_type": "쌍방향 소통"
        }}
      ],
      "consistency_note": "모든 팀 관계에서 일관된 경청 및 협업 패턴"
    }},
    "behavioral_reasoning": "Behavioral: 75-89점 구간. 모든 팀 프로젝트(3개)에서 경청 먼저 습관 ('먼저 들었다' 반복 언급). 갈등 반응 직면 시도 (회피하지 않고 대화로 해결, Segment 5, 9). 팀 우선 태도 명확 (본인 아이디어 양보 2회, '팀이 좋으면' 표현). 쌍방향 소통 (의견 제시 후 항상 '어떻게 생각하세요?' 확인). 90-100점 기준(모든 상황에서 완벽한 일관성, 심각한 갈등 조율)에는 미달하나 75-89점 충족. 79점.",
    
    "critical_penalties": -3,
    "red_flags": [
      {{
        "flag_type": "conflict_avoidance",
        "description": "Segment 8에서 '불편한 팀원 한 명은 그냥 피했다'며 일부 상황에서 갈등 회피",
        "severity": "minor",
        "penalty": -3,
        "evidence_reference": "segment_id: 8, char_index: 3100-3200"
      }}
    ],
    "critical_reasoning": "Critical: Red Flag 1건. Segment 8에서 '불편한 팀원은 그냥 피했다'며 일부 상황에서 갈등 회피(-3점). 총 감점 -3점."
  }},
  
  "overall_score": 82,
  "confidence": {{
    "evidence_strength": 0.85,
    "internal_consistency": 0.97,
    "overall_confidence": 0.85,
    "confidence_note": "증거 충분(Quote 4개), Evidence-Behavioral 간 편차 3점으로 일관적, Minor Flag 1건"
  }},
  
  "calculation": {{
    "base_score": 82,
    "evidence_weight": 0.85,
    "behavioral_adjustment": 0.94,
    "adjusted_score": 65.5,
    "amplified_score": 85.2,
    "critical_penalties": -3,
    "final_score": 82.2,
    "formula": "82 × 0.85 × 0.94 × 1.3 - 3 = 82.2 → 82점"
  }},
  
  "strengths": [
    "갈등 해결 복잡도 중상 (5명 팀, 방향성 충돌, 중재 성공)",
    "경청 깊이 중간 (이해 시도 + 의견 반영)",
    "모든 팀 프로젝트에서 경청 먼저 습관",
    "팀 우선 태도 명확 (개인 아이디어 양보 2회)",
    "쌍방향 소통 습관 ('어떻게 생각하세요?' 확인)"
  ],
  
  "weaknesses": [
    "일부 상황에서 갈등 회피 (Segment 8, 불편한 팀원)",
    "갈등 복잡도가 중상 수준 (심각한 갈등 경험 부족)",
    "경청 후 행동 변화가 일부만 반영 (완전한 Win-Win 미달)"
  ],
  
  "key_observations": [
    "신입 치고는 경청 및 협업 의식이 명확 (상위 30% 추정)",
    "모든 팀 관계에서 일관된 경청 먼저 패턴",
    "팀 우선 태도가 행동으로 구체화 (아이디어 양보)"
  ]
}}

═══════════════════════════════════════
**중요** 알림
═══════════════════════════════════════

1. 반드시 JSON만 출력하세요. 다른 텍스트 금지.
2. segment_id와 char_index를 함께 기록하세요.
3. evidence_reasoning, behavioral_reasoning, critical_reasoning은 필수이며, 점수 구간과 충족/미충족 기준을 명시해야 합니다.반드시 segment_ids 배열을 포함해야 합니다.
4. strengths, weaknesses는 필수입니다.
5. key_observations는 최소 3개 이상 작성하세요.
6. 모든 점수는 Quote에 기반해야 합니다.
7. 신입 기준으로 90점 이상은 매우 드뭅니다 (상위 10%).
8. "관계의 질" > "관계의 수" 우선순위를 유지하세요.
9. Stakeholder Management(수직)와 구별하여 팀원급 수평 관계 중심 평가하세요.
"""


def create_interpersonal_skill_evaluation_prompt(
    transcript: str
) -> str:
    """
    Interpersonal Skill Agent 평가 프롬프트 생성
    
    Args:
        transcript: InterviewTranscript의 JSON 문자열
    
    Returns:
        완성된 프롬프트
    """
    return INTERPERSONAL_SKILL_PROMPT.format(
        transcript=transcript
    )