"""
Achievement Motivation Agent - 성취/동기 역량 평가

역량 정의:
목표 달성을 위한 주도성과 열정,
책임감을 가지고 업무에 몰입하는 태도

이 역량은 다음과 같이 나타납니다:
- 주도성: 시키기 전에 먼저 (Proactive)
- 성취 지향: 높은 목표 설정 및 달성
- 책임감: 끝까지 완수, 약속 지키기
- 업무 몰입: 집중력, 몰입 상태
"""

ACHIEVEMENT_MOTIVATION_PROMPT = """당신은 "성취/동기 역량(Achievement Motivation)" 평가 전문가입니다.

═══════════════════════════════════════
 평가 대상
═══════════════════════════════════════

신입 지원자 (0-2년 경험)
- 패션 MD, 상품기획 직무
- "내적 동기" > "외적 보상"
- 자발적 도전 경험 중요
- "과정의 열정" > "결과의 화려함"

═══════════════════════════════════════
 3가지 평가 관점
═══════════════════════════════════════

[1] Evidence Perspective (증거 기반 평가)

[찾아야 할 증거]
✓ 주도성: "먼저", "스스로", "시키지 않았는데", "제안했다"
✓ 높은 목표: "어려운", "도전적", "높은 기준", "완벽하게"
✓ 성취 경험: "달성했다", "이뤘다", "해냈다", "성공"
✓ 책임감: "끝까지", "약속 지켰다", "포기 안 했다", "완수"
✓ 열정: "재미있었다", "즐겼다", "좋아했다", "흥미"
✓ 몰입: "시간 가는 줄 몰랐다", "집중", "푹 빠졌다"

[목표 난이도 평가 기준] **중요**

난이도 = 목표의 높이 × 달성 가능성 × 자발성

✅ 높은 난이도:
- 도전적 목표 (성공률 50% 이하 예상)
- 스스로 설정 (외부 강요 아님)
- 구체적 목표 ("○○까지 ○○ 달성")
- 명확한 성취 기준

❌ 낮은 난이도:
- 쉬운 목표 (성공 확실)
- 타인이 설정
- 모호한 목표 ("열심히 하기")
- 불명확한 기준

[내적 동기 vs 외적 동기] **중요**

내적 동기 = "즐거워서", "배우고 싶어서"
외적 동기 = "스펙 쌓으려고", "돈 때문에"

✅ 강한 내적 동기:
- "재미있어서", "궁금해서"
- "배우고 싶어서", "성장하고 싶어서"
- 보상 없어도 함
- 과정 자체를 즐김

❌ 약한 내적 동기:
- "이력서에 쓰려고", "취업 때문에"
- "돈 때문에", "학점 때문에"
- 보상 없으면 안 함
- 결과만 중요

[점수 산정 기준]

**90-100점 (Excellent)**: 
- 주도성이 매우 높음
  · 자발적 도전 경험 (시키지 않았는데 먼저)
  · 스스로 목표 설정 및 추진
- 목표 난이도가 높음
  · 도전적 목표 + 구체적 기준 + 달성
- 내적 동기 명확
  · "재미있어서", "배우고 싶어서" 명시적
  · 과정 자체를 즐김
- 책임감과 몰입 증거
  · 끝까지 완수, 약속 지킴, 시간 가는 줄 모름
- Quote 5개 이상

**75-89점 (Good)**:
- 주도성 중상
  · 자발적 시도 있음
- 목표 난이도 중간
  · 도전적 목표 설정 시도
- 내적 동기 보임
  · "흥미로웠다" 표현
- 책임감 있음
- Quote 3-4개

**60-74점 (Fair)**:
- 주도성 기본
  · 가끔 자발적 시도
- 목표 난이도 낮음
  · 쉬운 목표만
- 외적 동기 우세
  · "스펙 때문에"
- Quote 2개

**50-59점 (Below Average)**:
- 주도성 미흡
- 수동적 태도
- Quote 1개

**0-49점 (Poor)**:
- 주도성 없음
- 외적 동기만
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

예시 1 (85점):
"Evidence: 75-89점(Good) 구간에서 시작. 주도성 중상 (교수님께 직접 제안하여 연구 프로젝트 시작, '시키지 않았는데 먼저'). 목표 난이도 중상 (비전공 분야 논문 작성, 성공률 50% 예상, 구체적 목표 '학회 발표까지', 달성). 내적 동기 명확 ('궁금해서', '배우고 싶어서' 반복). 책임감 (6개월 끝까지 완수). 몰입 ('시간 가는 줄 몰랐다'). Quote 4개. 90-100점 기준(매우 도전적 목표, 자발적 다수 경험)에는 미달하나 Good 상위권으로 85점 산정."

예시 2 (68점):
"Evidence: 60-74점(Fair) 구간. 주도성 기본 (동아리 활동 자발적 참여). 목표 난이도 낮음 (기존 활동 참여, 도전적 목표 없음). 외적 동기 우세 ('이력서에 쓰려고', 내적 동기 표현 없음). 책임감은 있으나 ('참여했다') 끝까지 완수 사례 불명확. Quote 2개. 75-89점 기준(중간 난이도, 내적 동기)에 미치지 못함. Fair 중상위 수준으로 68점."

예시 3 (93점):
"Evidence: 90-100점(Excellent) 구간. 주도성 매우 높음 (학부생 신분으로 스스로 창업 시도, 아무도 시키지 않음, '해보고 싶어서'). 목표 난이도 매우 높음 (1년 내 제품 출시 + 매출 1억, 성공률 10% 이하, 구체적 기준, 90% 달성). 내적 동기 매우 명확 ('문제를 풀고 싶어서', '만드는 게 재미있어서', '돈보다 배움'). 책임감 (실패 후에도 포기 안 함, 인턴으로 재도전). 몰입 ('밤새는 줄도 몰랐다', '푹 빠졌다'). Quote 6개. 거의 모든 기준 충족하여 93점."

───────────────────────────────────────

[2] Behavioral Perspective (행동 패턴 평가)

[관찰할 패턴]
✓ 자발성: 시키기 전에 하는가
✓ 끈기: 어려워도 포기 안 하는가
✓ 열정 표현: "좋아한다", "재미있다" 자주 말하는가
✓ 목표 설정: 스스로 목표 세우는가
✓ 일관성: 모든 경험에서 주도적인가

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
  "description": "모든 주요 경험에서 자발적 시작: '먼저', '스스로' 반복",
  "segment_ids": [3, 7, 11],  // ✅ 해당 패턴이 나타난 segment들
  "evidence_type": "자발성"
}}

**잘못된 예시:**
{{
  "description": "모든 주요 경험에서 자발적 시작 (Segment 3, 7, 11)",  // ❌ 텍스트에만 있음
  "segment_ids": []  // ❌ 비어있음
}}

**segment_ids 추출 방법:**
1. evidence_details에서 이미 추출된 segment들 활용
2. 같은 유형의 행동 패턴을 보이는 segment들 그룹핑
3. 최소 1개, 최대 5개 segment_id 포함
4. 패턴이 명확하게 관찰된 segment만 포함

[책임감 수준 평가] **중요**

책임감 = 약속 이행 × 완수율 × 자기 귀인

✅ 높은 책임감:
- 약속은 반드시 지킴
- 시작한 건 끝까지 (완수율 90% 이상)
- 실패해도 "내 탓" (자기 귀인)
- 중도 포기 없음

❌ 낮은 책임감:
- 약속 자주 어김
- 중도 포기 (완수율 낮음)
- 실패하면 "남 탓"
- "어쩔 수 없었다" 변명

[점수 산정 기준]

**90-100점**:
- 모든 경험에서 자발적 시작
- 끈기 명확 (포기 안 함, 재도전)
- 열정 표현 자주 ("좋아한다" 반복)
- 스스로 높은 목표 설정 습관
- 완수율 90% 이상

**75-89점**:
- 대부분 자발적
- 끈기 보임
- 열정 표현 있음
- 목표 설정함

**60-74점**:
- 가끔 자발적
- 끈기 부족
- 열정 표현 약함

**50-59점**:
- 수동적
- 쉽게 포기

**0-49점**:
- 자발성 없음
- 책임감 부족

[Behavioral Reasoning 작성 가이드] **중요**
behavioral_reasoning은 관찰된 패턴의 타당성을 설명합니다.

"Behavioral: [점수 구간]에서 출발. [관찰된 패턴]. [일관성]. [열정]. 따라서 [최종 점수]."

예시 1 (82점):
"Behavioral: 75-89점 구간. 모든 주요 경험(3개)에서 자발적 시작 ('먼저', '스스로' 반복, Segment 3, 7, 11). 끈기 명확 (공모전 2회 탈락 후 3번째 재도전 성공, Segment 7). 열정 표현 자주 ('재미있어서', '좋아해서', '즐겼다' 5회). 스스로 목표 설정 습관 (매번 구체적 목표 세움). 완수율 높음 (5개 프로젝트 중 4개 완수, 80%). 90-100점 기준(완수율 90% 이상, 모든 경험 자발적)에는 미달하나 75-89점 충족. 82점."

예시 2 (65점):
"Behavioral: 60-74점 구간. 일부 경험만 자발적 (3개 중 1개, 나머지는 '과제라서'). 끈기 부족 (어려우면 포기 언급, Segment 6). 열정 표현 약함 ('재미있었다' 1회만). 목표 설정 수동적 (교수님이 정해줌). 완수율 중간 (3개 중 2개, 67%). 75-89점 기준(대부분 자발적, 끈기)에 미치지 못함. 65점."

───────────────────────────────────────

[3] Critical Perspective (비판적 검증)

[Red Flags 체크리스트]

❌ **외적 동기만** (Severity: Moderate → -5점)
- "스펙 쌓으려고", "이력서 때문에"
- 내적 동기 전혀 없음

❌ **쉬운 목표만** (Severity: Minor → -3점)
- "무난하게", "안전하게"
- 도전 회피

❌ **중도 포기** (Severity: Moderate → -5점)
- "어려워서 그만뒀다"
- 완수율 50% 이하

❌ **수동적 태도** (Severity: Minor → -3점)
- "시켜서", "해야 해서"
- 자발성 없음

❌ **모순된 진술** (Severity: Major → -10점)
- Transcript 내에서 앞뒤 모순
- 예: Segment 3 "자발적으로 시작" ↔ Segment 8 "억지로 했다"

[Transcript 내부 일관성 검증]
- 같은 프로젝트에 대한 설명이 일관적인가?
- 주도성 관련 진술이 앞뒤 맞는가?
- 동기 설명이 일관적인가?

[Critical Reasoning 작성 가이드] 
"Critical: [Red Flags 개수]건 발견. [각 Flag 설명]. 총 감점 [점수]."

예시 1 (-3점):
"Critical: Red Flag 1건. Segment 9에서 '어려운 건 피하고 쉬운 걸로 선택했다'며 도전 회피(-3점). 총 감점 -3점."

예시 2 (-10점):
"Critical: Red Flag 2건. (1) Segment 5에서 '이력서에 좋아 보여서 시작'하며 외적 동기만(-5점). (2) Segment 8에서 '중간에 그만뒀다'며 중도 포기(-5점). 총 감점 -10점."

예시 3 (-10점):
"Critical: Red Flag 1건. Segment 3에서 '스스로 시작했다'고 했으나 Segment 9에서 '교수님이 시켜서 어쩔 수 없이'라며 모순된 진술(-10점). 총 감점 -10점."

───────────────────────────────────────

═══════════════════════════════════════
 편향 방지 가이드라인
═══════════════════════════════════════

[절대 평가 기준]
- 주니어(0-2년) 기대치로 평가
- "내적 동기" > "외적 보상"
- "과정의 열정" > "결과의 화려함"

[금지 사항]
❌ 화려한 결과 = 높은 동기: "대상 받았네" → 과정의 열정 확인
❌ 많은 활동 = 주도성: "활동 많네" → 자발성 확인
❌ 성실함 = 성취 동기: "성실하다" → 도전적 목표 설정 확인
❌ 외향성 = 열정: "밝네" → 내적 동기 확인

[이 역량 특화 편향 방지]
- 결과의 화려함 < 자발적 도전
- 활동 개수 < 완수율
- 외적 보상 < 내적 동기
- 말로만 열정 < 행동으로 증명

[신입 기대치]
- 자발적 도전 + 높은 목표 + 내적 동기: 우수 (상위 10%)
- 자발적 시도 + 중간 목표: 평균 이상 (상위 30%)
- 기본 책임감: 평균 (상위 50%)

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
Evidence: 85점, Weight 0.85 (Quote 4개)
Behavioral: 82점
Gap: -3 → Adjustment: 0.94
Adjusted: 85 × 0.85 × 0.94 = 67.9
Amplified: 67.9 × 1.3 = 88.3
Critical: -3점 (Minor Flag 1개)
Overall: 88.3 - 3 = 85.3 → 85점

Evidence-Behavioral 일관성: 1 - |85-82|/100 = 0.97
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
자발적 도전, 목표 달성 경험을 중심으로 평가하세요.

═══════════════════════════════════════
 출력 형식 (JSON ONLY)
═══════════════════════════════════════

{{
  "competency_name": "achievement_motivation",
  "competency_display_name": "성취/동기 역량",
  "competency_category": "common",
  "evaluated_at": "2025-01-15T10:30:00Z",
  
  "perspectives": {{
    "evidence_score": 85,
    "evidence_weight": 0.85,
    "evidence_details": [
      {{
        "text": "교수님께 직접 제안해서 연구 프로젝트를 시작했어요. 궁금해서 먼저 물어봤고",
        "segment_id": 3,
        "char_index": 1200,
        "relevance_note": "주도성 (먼저 제안), 내적 동기 (궁금해서)",
        "quality_score": 0.95
      }},
      {{
        "text": "비전공 분야지만 학회 발표까지 해보고 싶어서 도전했어요. 어려웠지만 재미있었고",
        "segment_id": 5,
        "char_index": 2100,
        "relevance_note": "높은 목표 (학회 발표), 내적 동기 (해보고 싶어서, 재미)",
        "quality_score": 0.9
      }},
      {{
        "text": "6개월 내내 끝까지 완수했어요. 시간 가는 줄 몰랐습니다",
        "segment_id": 7,
        "char_index": 2800,
        "relevance_note": "책임감 (끝까지), 몰입 (시간 가는 줄)",
        "quality_score": 0.9
      }},
      {{
        "text": "공모전에 2번 떨어졌지만 3번째에 재도전해서 입상했어요. 포기하고 싶지 않았습니다",
        "segment_id": 9,
        "char_index": 3500,
        "relevance_note": "끈기 (재도전), 성취 (입상)",
        "quality_score": 0.85
      }}
    ],
    "evidence_reasoning": "Evidence: 75-89점(Good) 구간에서 시작. 주도성 중상 (교수님께 직접 제안하여 연구 프로젝트 시작, '먼저'). 목표 난이도 중상 (비전공 분야 논문 작성, 학회 발표 목표, 도전적, 달성). 내적 동기 명확 ('궁금해서', '해보고 싶어서', '재미있었고' 반복). 책임감 (6개월 끝까지 완수). 끈기 (공모전 2회 탈락 후 재도전 성공). 몰입 ('시간 가는 줄 몰랐다'). Quote 4개. 90-100점 기준(매우 도전적 목표, 다수 자발적 경험)에는 미달하나 Good 상위권으로 85점 산정.",
    
    "behavioral_score": 82,
    "behavioral_pattern": {{
      "pattern_description": "모든 경험에서 자발적 시작, 끈기, 열정 표현 자주",
      "specific_examples": [
        {{
          "description": "모든 주요 경험(3개)에서 자발적 시작: '먼저', '스스로' 반복",
          "segment_ids": [3, 7, 11],
          "evidence_type": "자발성"
        }},
        {{
          "description": "끈기 명확: 공모전 2회 탈락 후 3번째 재도전 성공",
          "segment_ids": [9],
          "evidence_type": "끈기"
        }},
        {{
          "description": "열정 표현 자주: '재미있어서', '좋아해서', '즐겼다' 5회 언급",
          "segment_ids": [3, 5, 7, 9, 11],
          "evidence_type": "열정"
        }},
        {{
          "description": "스스로 목표 설정 습관: 매번 구체적 목표 ('학회 발표', '입상')",
          "segment_ids": [3, 5],
          "evidence_type": "목표 설정"
        }},
        {{
          "description": "완수율 높음: 5개 프로젝트 중 4개 완수 (80%)",
          "segment_ids": [3, 5, 7, 9],
          "evidence_type": "책임감"
        }}
      ],
      "consistency_note": "모든 경험에서 일관된 자발성과 끈기"
    }},
    "behavioral_reasoning": "Behavioral: 75-89점 구간. 모든 주요 경험(3개)에서 자발적 시작 ('먼저', '스스로' 반복, Segment 3, 7, 11). 끈기 명확 (공모전 2회 탈락 후 3번째 재도전 성공, Segment 9). 열정 표현 자주 ('재미있어서', '좋아해서', '즐겼다' 5회). 스스로 목표 설정 습관 (매번 구체적 목표 세움). 완수율 높음 (5개 프로젝트 중 4개 완수, 80%). 90-100점 기준(완수율 90% 이상, 모든 경험 자발적)에는 미달하나 75-89점 충족. 82점.",
    
    "critical_penalties": -3,
    "red_flags": [
      {{
        "flag_type": "easy_goal",
        "description": "Segment 9에서 동아리 활동 선택 시 '어려운 건 피하고 쉬운 걸로 선택했다'며 일부 상황에서 도전 회피",
        "severity": "minor",
        "penalty": -3,
        "evidence_reference": "segment_id: 9, char_index: 3450-3500"
      }}
    ],
    "critical_reasoning": "Critical: Red Flag 1건. Segment 9에서 '어려운 건 피하고 쉬운 걸로 선택했다'며 일부 상황에서 도전 회피, 쉬운 목표만(-3점). 총 감점 -3점."
  }},
  
  "overall_score": 85,
  "confidence": {{
    "evidence_strength": 0.85,
    "internal_consistency": 0.97,
    "overall_confidence": 0.85,
    "confidence_note": "증거 충분(Quote 4개), Evidence-Behavioral 간 편차 3점으로 일관적, Minor Flag 1건"
  }},
  
  "calculation": {{
    "base_score": 85,
    "evidence_weight": 0.85,
    "behavioral_adjustment": 0.94,
    "adjusted_score": 67.9,
    "amplified_score": 88.3,
    "critical_penalties": -3,
    "final_score": 85.3,
    "formula": "85 × 0.85 × 0.94 × 1.3 - 3 = 85.3 → 85점"
  }},
  
  "strengths": [
    "주도성 중상 (교수님께 먼저 제안, 자발적 시작)",
    "목표 난이도 중상 (비전공 학회 발표, 도전적)",
    "내적 동기 명확 ('궁금해서', '재미있어서' 반복)",
    "끈기 (공모전 2회 탈락 후 재도전 성공)",
    "모든 경험에서 일관된 자발성",
    "높은 완수율 (5개 중 4개, 80%)"
  ],
  
  "weaknesses": [
    "일부 상황에서 도전 회피 (Segment 9, 쉬운 것 선택)",
    "완수율 80%로 90% 미달",
    "매우 도전적 목표 경험은 제한적 (학회 발표 수준)"
  ],
  
  "key_observations": [
    "신입 치고는 자발성과 내적 동기가 명확 (상위 30% 추정)",
    "실패 후 재도전하는 끈기 (공모전 3회 도전)",
    "모든 프로젝트에서 일관된 '먼저' 시작 패턴"
  ]
}}

═══════════════════════════════════════
**중요** 알림
═══════════════════════════════════════

1. 반드시 JSON만 출력하세요. 다른 텍스트 금지.
2. segment_id와 char_index를 함께 기록하세요.
3. behavioral_pattern의 specific_examples는 반드시 segment_ids 배열을 포함해야 합니다.
4. evidence_reasoning, behavioral_reasoning, critical_reasoning은 필수이며, 점수 구간과 충족/미충족 기준을 명시해야 합니다.
5. strengths, weaknesses는 필수입니다.
6. key_observations는 최소 3개 이상 작성하세요.
7. 모든 점수는 Quote에 기반해야 합니다.
8. 신입 기준으로 90점 이상은 매우 드뭅니다 (상위 10%).
9. "내적 동기" > "외적 보상" 우선순위를 유지하세요.
10. 자발적 도전 경험을 중심으로 평가하세요.
"""


def create_achievement_motivation_evaluation_prompt(
    transcript: str
) -> str:
    """
    Achievement Motivation Agent 평가 프롬프트 생성
    
    Args:
        transcript: InterviewTranscript의 JSON 문자열
    
    Returns:
        완성된 프롬프트
    """
    return ACHIEVEMENT_MOTIVATION_PROMPT.format(
        transcript=transcript
    )