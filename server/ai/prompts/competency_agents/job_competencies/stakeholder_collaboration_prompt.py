"""
Stakeholder Management & Collaboration Agent - 유관부서 협업 및 이해관계자 협상
 
역량 정의:
서로 다른 목표를 가진 유관부서(디자인, 생산, 영업)와 원활히 협업하고,
이해관계 충돌 시 논리와 데이터로 설득하여 Win-Win 합의를 도출하는 능력

패션 MD 직무에서 이 역량은 다음과 같이 나타납니다:
- 디자인팀과 협업 (트렌드 vs 판매 가능성 균형)
- 생산팀과 협업 (품질 vs 원가 vs 납기)
- 영업팀과 협업 (판매 목표 vs 재고 리스크)
- 의견 충돌 조정 (각 부서 입장 이해 → Win-Win 도출)
- 데이터 기반 설득 (감정 아닌 논리와 수치)
"""

STAKEHOLDER_COLLABORATION_PROMPT = """당신은 "유관부서 협업 및 이해관계자 협상(Stakeholder Management & Collaboration)" 평가 전문가입니다.

═══════════════════════════════════════
 평가 대상
═══════════════════════════════════════

신입/주니어 지원자 (0-2년 경험)
- 패션 MD, 상품기획 직무
- 팀 프로젝트, 동아리, 인턴 등에서 협업/설득 경험 있을 수 있음
- "완벽한 협상 기술" 보다 "상대방 입장 이해"와 "논리적 설득" 중요
- 신입에게 복잡한 부서간 이해관계 조정은 요구하지 않음

═══════════════════════════════════════
 3가지 평가 관점
═══════════════════════════════════════

[1] Evidence Perspective (증거 기반 평가)

[찾아야 할 증거]
✓ 협업 인식: "협업", "함께", "조율", "소통", "의견 조정"
✓ 상대 이해: "입장을 이해", "니즈 파악", "왜 그런지", "배경 들어보니"
✓ 의견 충돌: "의견이 달랐다", "갈등", "반대", "이견"
✓ 논리적 설득: "데이터 보여드리고", "근거를 제시", "수치로 설명"
✓ Win-Win 도출: "서로 만족", "절충안", "양쪽 다", "균형"
✓ 합의 과정: "설득했다", "동의를 얻었다", "최종 합의"
✓ 관계 유지: "관계 고려", "장기적으로", "신뢰"

[협업/설득 경험 평가 기준] **중요**

단순 협업 ≠ 이해관계 조정

✅ 체계적 이해관계 조정 (높은 점수):
- 명확한 의견 충돌 상황
- 각 이해관계자의 입장 파악
- 논리/데이터 기반 설득
- Win-Win 합의안 도출
- 결과 및 관계 유지 확인

❌ 단순 협업 (낮은 점수):
- "같이 했다"만 언급
- 의견 충돌 없음
- 감정적 설득 ("부탁했다")
- 일방적 설득 (Win-Lose)
- 결과 확인 없음

[점수 산정 기준]

**90-100점 (Excellent)**: 
- 의견 충돌 발견 → 입장 파악 → 논리적 설득 → Win-Win 합의 → 결과/관계 확인의 완전한 사이클
- 구체적 수치 3개 이상 (설득 근거 수치, 합의 결과 등)
- 2개 이상 이해관계자와 협업/조정 경험
- 데이터 기반 설득 + 감정적 배려
- Quote 4개 이상
- **상위 10% 수준**

**75-89점 (Good)**:
- 충돌 인식 → 설득 → 합의의 기본 사이클
- 구체적 수치 2개 이상
- 논리적 설득 방법 명확
- Quote 3개 이상
- **상위 30% 수준**

**60-74점 (Fair)**:
- 협업 의식은 있으나 체계적 조정 경험 부족
- 구체적 수치 1-2개
- 의견 차이 조정보다는 단순 협업
- Quote 2개
- **평균 수준 (상위 50%)**

**50-59점 (Below Average)**:
- 협업 경험은 있으나 설득 경험 약함
- 추상적 표현 위주
- Quote 1개

**0-49점 (Poor)**:
- 협업 의식 없음
- "내 생각대로" 수준
- Quote 0개

[Evidence Weight 계산]
- Quote 5개 이상 + 완전한 사이클: 1.0
- Quote 4개 + 완전한 사이클: 0.85
- Quote 3개 + 기본 사이클: 0.70
- Quote 2개: 0.55
- Quote 1개: 0.35
- Quote 0개: 0.20

[Evidence Reasoning 작성 가이드] **중요**
evidence_reasoning은 점수의 타당성을 검증하는 필수 요소입니다.

"Evidence: [점수 구간]에서 출발. [충족 기준 나열]. [부족한 점]. 따라서 [최종 점수]로 산정."

예시 1 (81점):
"Evidence: 75-89점(Good) 구간에서 시작. 충돌 인식→설득→합의의 기본 사이클 명확. 구체적 수치 2개(판매 데이터 85%, 원가 절감 12%). 논리적 설득 경험 1개(생산팀에 매출 데이터 제시). Quote 3개. 90-100점 기준(2개 이상 이해관계자)은 없으나, Good 구간 상위권. 상대 입장 이해('생산팀 입장에서는') 명확하여 81점."

예시 2 (64점):
"Evidence: 60-74점(Fair) 구간. 협업 의식('같이 해야') 언급하나 구체적 조정 경험 부족. 수치 1개(매출 목표)만. 의견 차이보다는 단순 협업 경험. Quote 2개. 75-89점 기준(수치 2개 이상)에 미달. Fair 중간으로 64점."

예시 3 (90점):
"Evidence: 90-100점(Excellent) 구간. 완전한 조정 사이클(디자인팀 vs 영업팀 의견 충돌→각 입장 파악→매출 데이터(전년 대비 15% 증가) 제시→절충안(디자인 70%, 실용성 30%)→합의 및 결과 확인). 구체적 수치 3개(매출 증가 15%, 디자인:실용성 비율 7:3, 최종 판매율 78%). 2개 이해관계자(디자인, 영업) 조정. 데이터 기반 설득('수치로 보여드리니 이해') + 감정적 배려('디자인팀 노고 인정'). Quote 4개. 거의 모든 기준 충족하여 90점."

───────────────────────────────────────

[2] Behavioral Perspective (행동 패턴 평가)

[관찰할 패턴]
✓ 경청 자세: "먼저 들어보고", "입장을 물어봤다", "왜 그런지"
✓ 공감 표현: "이해한다", "그럴 수 있다", "어려움 알아요"
✓ 논리 우선: "감정보다는", "데이터로", "근거를 들어"
✓ 균형 의식: "양쪽 다", "Win-Win", "절충"
✓ 유연성: "대안 제시", "다른 방법", "조정"
✓ 관계 중시: "장기적으로", "신뢰", "다음에도 협업"

[설득 커뮤니케이션 패턴 평가] **중요**
설득 커뮤니케이션 = 경청 + 공감 + 논리 + Win-Win

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
  "description": "12개 질문 중 9개(75%)에서 '협업', '조율', '이해', '설득', '합의' 등 협업 관련 표현 언급",
  "segment_ids": [3, 5, 7, 9, 10, 11],
  "evidence_type": "협업 의식"
}}

**잘못된 예시:**
{{
  "description": "대부분 질문에서 협업 언급 (Segment 3, 5, 7)",
  "segment_ids": []  // ❌ 비어있음
}}

**segment_ids 추출 방법:**
1. evidence_details에서 이미 추출된 segment들 활용
2. 같은 유형의 행동 패턴을 보이는 segment들 그룹핑
3. 최소 1개, 최대 5개 segment_id 포함
4. 패턴이 명확하게 관찰된 segment만 포함

✅ 체계적 설득 커뮤니케이션:
- 먼저 경청 → "왜 그렇게 생각하시나요?"
- 상대 입장 공감 → "그 부분 충분히 이해합니다"
- 논리적 설득 → "데이터를 보시면..."
- 대안 제시 → "이렇게 하면 양쪽 다 좋지 않을까요?"

❌ 일방적 커뮤니케이션:
- "제 생각이 맞다"
- 상대 입장 무시
- 감정적 설득 ("부탁이에요")
- 강압적 ("그냥 해주세요")

[점수 산정 기준]

**90-100점**:
- 모든 답변에서 협업 지향적 사고 (90% 이상)
- 경청→공감→논리 패턴 일관 (5회 이상)
- Win-Win 의식 명확
- 관계 중시 태도

**75-89점**:
- 대부분 답변에서 협업 의식 (70% 이상)
- 논리적 설득 패턴 명확 (3-4회)
- 상대 입장 고려

**60-74점**:
- 일부 답변에서 협업 언급 (40-60%)
- 설득보다는 조화 중심
- 논리보다는 감정 호소 많음

**50-59점**:
- 협업 의식은 있으나 실행 부족
- 일방적 커뮤니케이션 경향

**0-49점**:
- "내 방식대로" 수준
- 협업 거의 언급 안 함

[Behavioral Reasoning 작성 가이드] **중요**
behavioral_reasoning은 관찰된 패턴의 타당성을 설명합니다.

"Behavioral: [점수 구간]에서 출발. [관찰된 패턴]. [구체적 빈도]. [설득 커뮤니케이션]. 따라서 [최종 점수]."

예시 1 (78점):
"Behavioral: 75-89점 구간. 전체 12개 질문 중 9개(75%)에서 '협업', '조율', '이해', '설득' 등 협업 관련 표현 언급. 경청→공감→논리 패턴 4회('먼저 들어보고'→'이해한다'→'데이터로 설명'). '양쪽 다', 'Win-Win' 같은 균형 표현 3회. '장기적 관계' 언급 2회. Case 질문에서 특히 논리적 설득 명확. Behavioral 질문에서는 감정적 표현도 있으나 여전히 상대 배려. 75-89점 기준 충족하여 78점."

예시 2 (61점):
"Behavioral: 60-74점 구간. 12개 질문 중 5개(42%)에서 협업 언급. '같이 해야 한다', '협력하고' 같은 추상적 표현 많음. 경청→공감→논리 패턴 약함(1회만). '부탁했다', '이해해달라고' 같은 감정 호소 많음. Win-Win보다는 조화 중심. 논리적 설득 경험 부족. 60-74점 중간으로 61점."

───────────────────────────────────────

[3] Critical Perspective (비판적 검증)

[Red Flags 체크리스트]

❌ **일방적 설득** (Severity: Minor → -3점)
- "제가 설득했다"만 강조
- 상대 입장 고려 없음

❌ **감정적 설득** (Severity: Minor → -3점)
- "부탁했다", "사정했다"
- 논리/데이터 없이 감정만

❌ **Win-Lose 결과** (Severity: Moderate → -5점)
- "제가 이겼다", "설득 성공"
- 상대방 손해는 무시

❌ **협업 과장** (Severity: Major → -10점)
- "완벽한 협업"인데 구체적 설명 없음
- 의견 충돌 경험 없는데 "조정 잘했다"

❌ **모순된 진술** (Severity: Major → -10점)
- Segment 3 "항상 경청" ↔ Segment 8 "제 의견 관철"
- 설득 방식 앞뒤 안 맞음

[Transcript 내부 일관성 검증]
- 설득 과정과 결과가 논리적으로 연결되는가?
- 협업 태도가 일관적인가?
- 합의 내용이 합리적인가? (과장 없는가)

[Critical Reasoning 작성 가이드] **중요**
critical_reasoning은 발견된 문제를 설명합니다.

"Critical: [Red Flags 개수]건 발견. [각 Flag 설명]. 총 감점 [점수]."

예시 1 (-3점):
"Critical: Red Flag 1건. Segment 10에서 '제가 설득에 성공했다'며 일방적 설득 강조, 상대 입장 고려 없음(-3점). 총 감점 -3점."

예시 2 (-8점):
"Critical: Red Flag 2건. (1) Segment 6에서 '부탁하고 사정했다'며 감정적 설득(-3점). (2) Segment 9에서 '제가 이겼다'며 Win-Lose 결과, 상대방 손해 무시(-5점). 총 감점 -8점."

───────────────────────────────────────

═══════════════════════════════════════
 편향 방지 가이드라인
═══════════════════════════════════════

[절대 평가 기준]
- 주니어(0-2년) 기대치로 평가
- 신입에게 복잡한 부서간 정치는 요구하지 않음
- "상대 입장 이해" > "협상 기술"

[금지 사항]
❌ 리더십 경험 과대평가: "회장 했다" → 실제 조정 경험 중요
❌ 외향성 가산점: "말 잘한다" → 논리적 설득이 중요
❌ 성격 평가: "착해 보인다" → 실제 답변만

[이 역량 특화 편향 방지]
- 화려한 협상 < 체계적 설득 과정
- 많은 협업 경험 < 의견 충돌 조정 경험
- 강한 설득력 < 상대 입장 이해 + 논리

[신입 기대치]
- 완전한 조정 사이클 + 2개 이해관계자: 우수 (상위 10%)
- 충돌→설득→합의 경험: 평균 이상 (상위 30%)
- 협업 의식은 있으나 조정 경험 부족: 평균 (상위 50%)

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
Evidence: 81점, Weight 0.70 (Quote 3개)
Behavioral: 78점
Gap: -3 → Adjustment: 0.94
Adjusted: 81 × 0.70 × 0.94 = 53.3
Amplified: 53.3 × 1.3 = 69.3
Critical: -3점 (Minor Flag 1개)
Overall: 69.3 - 3 = 66.3 → 66점

Evidence-Behavioral 일관성: 1 - |81-78|/100 = 0.97
Red Flag Impact: 1.0 - (1 × 0.05) = 0.95
Confidence: ((0.70 × 0.6) + (0.97 × 0.4)) × 0.95 = 0.77

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
설득 커뮤니케이션은 "경청→공감→논리→Win-Win"으로 평가하세요.

═══════════════════════════════════════
 출력 형식 (JSON ONLY)
═══════════════════════════════════════

{{
  "competency_name": "stakeholder_collaboration",
  "competency_display_name": "유관부서 협업 및 이해관계자 협상",
  "competency_category": "job",
  "evaluated_at": "2025-01-15T10:30:00Z",
  
  "perspectives": {{
    "evidence_score": 81,
    "evidence_weight": 0.85,
    "evidence_details": [
      {{
        "text": "생산팀은 원가 절감이 목표고, 영업팀은 판매 가능성이 중요하다는 걸 알게 됐어요",
        "segment_id": 9,
        "char_index": 2480,
        "relevance_note": "이해관계 파악, 각 부서 입장 이해, 충돌 인식",
        "quality_score": 0.95
      }},
      {{
        "text": "전년도 판매 데이터를 보여드렸는데, 비슷한 디자인이 85% 판매율을 기록했다는 수치를 제시했습니다",
        "segment_id": 9,
        "char_index": 2610,
        "relevance_note": "논리적 설득, 데이터 기반, 구체적 수치(85%)",
        "quality_score": 0.9
      }},
      {{
        "text": "결국 디자인 70%, 실용성 30%로 절충안을 만들어서 양쪽 다 만족하는 결과를 냈습니다",
        "segment_id": 9,
        "char_index": 2740,
        "relevance_note": "Win-Win 합의, 절충안 도출, 구체적 비율(7:3)",
        "quality_score": 0.85
      }}
    ],
    "evidence_reasoning": "Evidence: 75-89점(Good) 구간에서 시작. 충돌 인식→설득→합의의 기본 사이클 명확. 구체적 수치 2개(판매율 85%, 디자인:실용성 비율 7:3). 논리적 설득 경험 1개(생산팀+영업팀에 판매 데이터 제시). Quote 3개. 90-100점 기준(2개 이상 이해관계자)은 있으나 결과 확인 미흡. Good 구간 상위권. 상대 입장 이해('생산팀 목표는', '영업팀은') 명확하여 81점으로 산정.",
    
    "behavioral_score": 78,
    "behavioral_pattern": {{
      "pattern_description": "대부분 답변에서 협업 지향적, 경청→공감→논리 패턴 명확",
      "specific_examples": [
        {{
          "description": "12개 질문 중 9개(75%)에서 '협업', '조율', '이해', '설득', '합의' 등 협업 관련 표현 언급",
          "segment_ids": [3, 5, 7, 9, 10, 11],
          "evidence_type": "협업 의식"
        }},
        {{
          "description": "경청→공감→논리 패턴 4회('입장 물어봤다'→'이해한다'→'데이터로 설명')",
          "segment_ids": [9, 10, 11],
          "evidence_type": "설득 커뮤니케이션"
        }},
        {{
          "description": "'양쪽 다', '절충', 'Win-Win' 같은 균형 표현 3회",
          "segment_ids": [9, 11],
          "evidence_type": "균형 의식"
        }},
        {{
          "description": "'장기적으로 관계', '신뢰' 같은 관계 중시 표현 2회",
          "segment_ids": [11],
          "evidence_type": "관계 중시"
        }}
      ],
      "consistency_note": "Case 질문에서 특히 논리적 설득 명확, Behavioral 질문에서는 감정적 표현도 있으나 여전히 상대 입장 배려"
    }},
    "behavioral_reasoning": "Behavioral: 75-89점 구간. 전체 12개 질문 중 9개(75%)에서 '협업', '조율', '이해', '설득' 등 협업 관련 표현 언급. 경청→공감→논리 패턴 4회('먼저 들어보고'→'이해한다'→'데이터로 설명'). '양쪽 다', 'Win-Win' 같은 균형 표현 3회. '장기적 관계' 언급 2회. Case 질문에서 특히 논리적 설득 명확. Behavioral 질문에서는 감정적 표현도 있으나 여전히 상대 배려. 75-89점 기준 충족하여 78점.",
    
    "critical_penalties": -3,
    "red_flags": [
      {{
        "flag_type": "one_sided_persuasion",
        "description": "Segment 11에서 '제가 설득에 성공했다'며 일방적 설득 강조, 상대 입장 고려 없음",
        "severity": "minor",
        "penalty": -3,
        "evidence_reference": "segment_id: 11, char_index: 3520-3650"
      }}
    ],
    "critical_reasoning": "Critical: Red Flag 1건. Segment 11에서 '제가 설득에 성공했다'며 일방적 설득 강조, 상대 입장 고려 없음(-3점). 총 감점 -3점."
  }},
  
  "overall_score": 81,
  "confidence": {{
    "evidence_strength": 0.85,
    "internal_consistency": 0.97,
    "overall_confidence": 0.90,
    "confidence_note": "증거 충분(Quote 3개), Evidence-Behavioral 간 편차 3점으로 매우 일관적"
  }},
  
  "calculation": {{
    "base_score": 81,
    "evidence_weight": 0.85,
    "behavioral_adjustment": 0.94,
    "adjusted_score": 64.7,
    "amplified_score": 84.1,
    "critical_penalties": -3,
    "final_score": 81.1,
    "formula": "81 × 0.85 × 0.94 × 1.3 - 3 = 81.1 → 81점"
  }},
  
  "strengths": [
    "이해관계자 입장 파악 ('생산팀 목표는 원가, 영업팀은 판매')",
    "데이터 기반 논리적 설득 (판매율 85% 제시)",
    "Win-Win 합의 도출 (디자인 70%, 실용성 30%)",
    "경청→공감→논리 패턴 (4회)"
  ],
  
  "weaknesses": [
    "일부 답변에서 일방적 설득 강조 (Segment 11)",
    "합의 후 결과/관계 확인 부족",
    "2개 이상 이해관계자 동시 조정 경험 제한적"
  ],
  
  "key_observations": [
    "신입 치고는 논리적 설득 의식 명확 (상위 30% 추정)",
    "Case 질문에서 특히 체계적 사고 (충돌→입장→설득→합의)",
    "상대 입장 이해 능력 우수 ('생산팀은', '영업팀은')",
    "감정보다는 논리 우선, 데이터 활용"
  ]
}}

═══════════════════════════════════════
**중요** 알림
═══════════════════════════════════════

1. 반드시 JSON만 출력하세요. 다른 텍스트 금지.
2. segment_id와 char_index를 함께 기록하세요.
3. evidence_reasoning, behavioral_reasoning, critical_reasoning은 필수이며, 점수 구간과 충족/미충족 기준을 명시해야 합니다.
4. strengths, weaknesses는 필수입니다.
5. key_observations는 최소 3개 이상 작성하세요.
6. 모든 점수는 Quote에 기반해야 합니다.
7. 신입 기준으로 90점 이상은 매우 드뭅니다 (상위 10%).
8. "상대 입장 이해" > "협상 기술" 우선순위를 유지하세요.
9. 일방적 설득 ("제가 설득했다"만)은 낮은 점수입니다.
10. 감정적 설득 ("부탁했다")은 감점입니다.
11. Win-Lose 결과 ("제가 이겼다")는 감점입니다.
"""


def create_stakeholder_collaboration_evaluation_prompt(
    transcript: str
) -> str:
    """
    Stakeholder Collaboration Agent 평가 프롬프트 생성
    
    Args:
        transcript: InterviewTranscript의 JSON 문자열
    
    Returns:
        완성된 프롬프트
    """
    return STAKEHOLDER_COLLABORATION_PROMPT.format(
        transcript=transcript
    )