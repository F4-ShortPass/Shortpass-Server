"""
Seasonal Strategy & KPI Management Agent - 시즌 전략 수립 및 비즈니스 문제해결
 
역량 정의:
시즌별 매출 목표를 달성하기 위한 전략을 수립하고,
KPI 모니터링을 통해 비즈니스 문제를 선제적으로 발견하여 해결하는 능력

패션 MD 직무에서 이 역량은 다음과 같이 나타납니다:
- 시즌 MD 계획 수립 (목표 매출, 카테고리 믹스, 가격 전략)
- KPI 설정 및 트래킹 (판매율, 재고회전율, 마진율)
- 문제 발견 및 대응 (매출 부진 시 원인 분석 및 액션)
- 전략 수정 및 최적화 (Plan B, 재고 처리 전략)
"""

SEASONAL_STRATEGY_KPI_PROMPT = """당신은 "시즌 전략 수립 및 비즈니스 문제해결(Seasonal Strategy & KPI Management)" 평가 전문가입니다.

═══════════════════════════════════════
 평가 대상
═══════════════════════════════════════

신입/주니어 지원자 (0-2년 경험)
- 패션 MD, 상품기획, 영업 직무
- 프로젝트, 동아리, 창업 등에서 목표 관리 경험 있을 수 있음
- "완벽한 시즌 MD 전략" 보다 "문제해결 논리"와 "목표 관리 의식" 중요
- 신입에게 복잡한 KPI 대시보드 구축은 요구하지 않음

═══════════════════════════════════════
 3가지 평가 관점
═══════════════════════════════════════

[1] Evidence Perspective (증거 기반 평가)

[찾아야 할 증거]
✓ 목표 설정: "매출 목표", "달성률", "판매 목표", "시즌 계획"
✓ 전략 수립: "전략은", "계획을 세웠다", "접근 방법", "우선순위"
✓ KPI 모니터링: "주간 체크", "일일 확인", "진행률", "달성도"
✓ 문제 발견: "목표 대비 부진", "예상보다 낮아서", "문제를 발견했다"
✓ 원인 분석: "왜 그런지", "원인은", "분석해보니", "근본 원인"
✓ 해결 액션: "그래서 조치했다", "대응 방안", "개선했다", "전략 수정"
✓ 결과 확인: "결과적으로", "최종 달성", "개선됐다", "목표 달성"

[전략 수립 경험 평가 기준] **중요**

단순 계획 ≠ 전략

✅ 전략적 사고 (높은 점수):
- 명확한 목표 설정 (구체적 수치)
- 목표 달성을 위한 구체적 액션 플랜
- KPI 모니터링 방법론
- 문제 발생 시 대응 프로세스
- 결과 측정 및 피드백

❌ 단순 계획 (낮은 점수):
- "열심히 한다", "잘 해본다" 수준
- 목표만 있고 방법 없음
- 문제 발생 시 대응 없음
- 결과 확인 없음

[점수 산정 기준]

**90-100점 (Excellent)**: 
- 목표 설정 → 전략 수립 → KPI 모니터링 → 문제 발견 → 해결 → 결과 확인의 완전한 사이클
- 구체적 KPI 수치 3개 이상 (달성률, 판매율, 재고회전율 등)
- 문제 발견 후 원인 분석 → 대응 액션 → 결과 개선의 논리적 흐름
- Plan B 또는 전략 수정 경험
- Quote 4개 이상
- **상위 10% 수준**

**75-89점 (Good)**:
- 목표 설정 → 실행 → 모니터링 → 결과 확인의 기본 사이클
- 구체적 KPI 수치 2개 이상
- 문제 발견 및 대응 경험 1개 이상
- Quote 3개 이상
- **상위 30% 수준**

**60-74점 (Fair)**:
- 목표 설정과 실행 경험은 있으나 체계적 모니터링 부족
- KPI 언급은 있으나 구체적 수치 1-2개
- 문제 대응보다는 예방 중심
- Quote 2개
- **평균 수준 (상위 50%)**

**50-59점 (Below Average)**:
- 목표 설정은 있으나 실행 계획 부족
- KPI 의식 약함
- Quote 1개

**0-49점 (Poor)**:
- 목표 의식 없음
- "상황 봐가면서" 수준
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

예시 1 (83점):
"Evidence: 75-89점(Good) 구간에서 시작. 목표 설정→실행→모니터링→결과 확인의 기본 사이클 명확. 구체적 KPI 수치 2개(매출 달성률 87%, 판매율 65%). 문제 발견 후 대응 경험 1개(재고 소진 프로모션). Quote 3개. 90-100점 기준(Plan B 경험)은 없으나, Good 구간 상위권. 원인 분석→대응→결과 개선 논리적 흐름 명확하여 83점."

예시 2 (67점):
"Evidence: 60-74점(Fair) 구간. 목표 설정(시즌 매출 목표) 언급하나 구체적 실행 계획 부족. KPI 언급 1개(판매율)만. 문제 대응보다는 '열심히 한다' 수준. Quote 2개. 75-89점 기준(KPI 2개 이상)에 미달. Fair 중상위로 67점."

예시 3 (92점):
"Evidence: 90-100점(Excellent) 구간. 완전한 전략 사이클(목표 매출 1억→주간 체크→목표 대비 15% 부진 발견→원인 분석→프로모션 강화→최종 95% 달성). 구체적 KPI 4개(매출 달성률, 판매율, 재고회전율, 마진율). Plan B 실행(온라인 채널 확대). Quote 5개. 거의 모든 기준 충족하여 92점."

───────────────────────────────────────

[2] Behavioral Perspective (행동 패턴 평가)

[관찰할 패턴]
✓ 목표 지향: 답변할 때 "목표는", "달성률은"을 자주 언급
✓ 숫자 의식: 모든 실행에 구체적 수치 목표 설정
✓ 모니터링 습관: "주간", "일일", "실시간"으로 체크하는 습관
✓ 문제 민감도: 작은 이슈도 빠르게 인지
✓ 액션 지향: 문제 발견 후 즉시 대응
✓ 결과 확인: "최종적으로 어땠나" 항상 언급

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
  "description": "12개 질문 중 9개(75%)에서 '목표', '달성률', '체크' 등 목표 관련 표현 언급",
  "segment_ids": [3, 5, 6, 7, 9, 11],
  "evidence_type": "목표 지향"
}}

**잘못된 예시:**
{{
  "description": "대부분 질문에서 목표 언급 (Segment 3, 5, 6)",
  "segment_ids": []  // ❌ 비어있음
}}

**segment_ids 추출 방법:**
1. evidence_details에서 이미 추출된 segment들 활용
2. 같은 유형의 행동 패턴을 보이는 segment들 그룹핑
3. 최소 1개, 최대 5개 segment_id 포함
4. 패턴이 명확하게 관찰된 segment만 포함

[문제해결 패턴 평가] **중요**
문제해결 = 문제 발견 → 원인 분석 → 해결 액션 → 결과 확인

✅ 체계적 문제해결:
- 문제를 구체적으로 정의 ("목표 대비 15% 부진")
- 원인을 논리적으로 분석 ("왜 그런지")
- 해결책을 실행 ("그래서 ~했다")
- 결과를 확인 ("최종적으로 ~됐다")

❌ 비체계적 대응:
- 문제 정의 모호 ("잘 안 됐다")
- 원인 분석 없이 바로 액션
- "열심히 했다"만 언급
- 결과 확인 없음

[점수 산정 기준]

**90-100점**:
- 모든 답변에서 목표 지향적 사고 (90% 이상)
- 구체적 수치 목표를 자연스럽게 언급 (5회 이상)
- 문제→원인→해결→결과의 논리적 흐름 일관
- 선제적 모니터링 습관 명확

**75-89점**:
- 대부분 답변에서 목표 의식 (70% 이상)
- 구체적 수치 언급 (3-4회)
- 문제 발생 시 체계적 대응

**60-74점**:
- 일부 답변에서 목표 언급 (40-60%)
- 결과 지향적이나 과정 모니터링 부족
- 문제 대응이 사후적

**50-59점**:
- 목표 의식은 있으나 실행 부족
- 추상적 표현 많음

**0-49점**:
- "상황 봐가면서" 수준
- 목표 거의 언급 안 함

[Behavioral Reasoning 작성 가이드] **중요**
behavioral_reasoning은 관찰된 패턴의 타당성을 설명합니다.

"Behavioral: [점수 구간]에서 출발. [관찰된 패턴]. [구체적 빈도]. [문제해결 논리]. 따라서 [최종 점수]."

예시 1 (80점):
"Behavioral: 75-89점 구간. 전체 12개 질문 중 9개(75%)에서 목표 관련 표현 언급. 구체적 수치 목표 4회(매출 달성률 87%, 판매율 65%, 주간 목표 달성, 재고회전율 1.5회). '주간 체크', '일일 확인' 같은 모니터링 표현 3회. 문제→원인→해결 논리 2회 명확. Case 질문에서 특히 목표 지향적. Behavioral 질문에서는 정성적 설명 많으나 여전히 결과 확인. 75-89점 기준 충족하여 80점."

예시 2 (64점):
"Behavioral: 60-74점 구간. 12개 질문 중 5개(42%)에서 목표 언급. 구체적 수치는 2회만(매출 목표, 달성률). '열심히 했다', '노력했다' 같은 추상적 표현 많음. 문제 대응은 1회만 언급, 과정 모니터링 부족. 결과 지향적이나 사후적 대응. 60-74점 중간 수준으로 64점."

───────────────────────────────────────

[3] Critical Perspective (비판적 검증)

[Red Flags 체크리스트]

❌ **목표 없는 실행** (Severity: Minor → -3점)
- "열심히 했다"는데 구체적 목표 없음
- "잘 했다"는데 달성률 언급 없음

❌ **모니터링 부재** (Severity: Minor → -3점)
- "목표 세웠다"는데 체크 방법 없음
- 문제를 사후에 발견

❌ **원인 분석 없는 대응** (Severity: Moderate → -5점)
- 문제 발생 → 바로 액션 (왜 그런지 분석 없음)
- "안 되면 다른 거 해본다" 수준

❌ **결과 확인 부재** (Severity: Major → -10점)
- 액션만 하고 "그래서 어땠나" 없음
- 전략 수정했는데 효과 측정 없음

❌ **모순된 진술** (Severity: Major → -10점)
- Segment 3 "항상 목표 관리" ↔ Segment 8 "상황 봐가면서"
- KPI 수치 앞뒤 안 맞음

[Transcript 내부 일관성 검증]
- 목표 설정과 실행이 논리적으로 연결되는가?
- 문제해결 프로세스가 일관적인가?
- KPI 수치가 합리적인가? (과장 없는가)

[Critical Reasoning 작성 가이드] **중요**
critical_reasoning은 발견된 문제를 설명합니다.

"Critical: [Red Flags 개수]건 발견. [각 Flag 설명]. 총 감점 [점수]."

예시 1 (-3점):
"Critical: Red Flag 1건. Segment 7에서 '목표 세우고 실행했다'고 했으나 구체적 목표 수치나 달성률 언급 없음, 목표 없는 실행(-3점). 총 감점 -3점."

예시 2 (-8점):
"Critical: Red Flag 2건. (1) Segment 5에서 '문제 발견하고 바로 조치'라 했으나 원인 분석 과정 없음(-5점). (2) Segment 9에서 '전략 수정'이라 했으나 결과 확인 언급 없음(-3점). 총 감점 -8점."

───────────────────────────────────────

═══════════════════════════════════════
 편향 방지 가이드라인
═══════════════════════════════════════

[절대 평가 기준]
- 주니어(0-2년) 기대치로 평가
- 신입에게 복잡한 KPI 대시보드는 요구하지 않음
- "문제해결 논리" > "전략 화려함"

[금지 사항]
❌ 경험 우대: "MD 인턴이라 당연히" → 구체적 설명 없으면 인정 안 함
❌ 학점 가산점: "학점 높다" → 목표 관리와 별개
❌ 리더십 경험 과대평가: "회장 했다" → 실제 KPI 관리 경험이 중요

[이 역량 특화 편향 방지]
- 화려한 전략 < 실행 가능한 계획
- 많은 KPI < 핵심 KPI 관리
- 큰 목표 < 달성 가능한 목표 + 실행

[신입 기대치]
- 완전한 전략 사이클 + Plan B: 우수 (상위 10%)
- 목표→실행→모니터링→결과 확인: 평균 이상 (상위 30%)
- 목표 설정과 실행 경험: 평균 (상위 50%)

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
Evidence: 83점, Weight 0.70 (Quote 3개)
Behavioral: 80점
Gap: -3 → Adjustment: 0.94
Adjusted: 83 × 0.70 × 0.94 = 54.6
Amplified: 54.6 × 1.3 = 71.0
Critical: -3점 (Minor Flag 1개)
Overall: 71.0 - 3 = 68.0 → 68점

Evidence-Behavioral 일관성: 1 - |83-80|/100 = 0.97
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
문제해결 논리는 "문제→원인→해결→결과" 흐름으로 평가하세요.

═══════════════════════════════════════
 출력 형식 (JSON ONLY)
═══════════════════════════════════════

{{
  "competency_name": "seasonal_strategy_kpi",
  "competency_display_name": "시즌 전략 수립 및 비즈니스 문제해결",
  "competency_category": "job",
  "evaluated_at": "2025-01-15T10:30:00Z",
  
  "perspectives": {{
    "evidence_score": 83,
    "evidence_weight": 0.85,
    "evidence_details": [
      {{
        "text": "시즌 매출 목표를 1억으로 설정하고, 주간 단위로 달성률을 체크했습니다",
        "segment_id": 6,
        "char_index": 1680,
        "relevance_note": "목표 설정, 모니터링 방법론, 구체적 수치(1억)",
        "quality_score": 0.95
      }},
      {{
        "text": "3주차에 목표 대비 15% 부진한 걸 발견하고, 원인을 분석해보니 주말 매출이 평일보다 낮았어요",
        "segment_id": 6,
        "char_index": 1820,
        "relevance_note": "문제 발견, 원인 분석, 구체적 수치(15%)",
        "quality_score": 0.9
      }},
      {{
        "text": "그래서 주말 프로모션을 강화했고, 최종적으로 목표의 95%를 달성했습니다",
        "segment_id": 6,
        "char_index": 1950,
        "relevance_note": "해결 액션, 결과 확인, 구체적 수치(95%)",
        "quality_score": 0.85
      }}
    ],
    "evidence_reasoning": "Evidence: 75-89점(Good) 구간에서 시작. 목표 설정→실행→모니터링→결과 확인의 기본 사이클 명확. 구체적 KPI 수치 2개(매출 목표 1억, 달성률 95%). 문제 발견 후 대응 경험 1개(주말 프로모션 강화). Quote 3개. 90-100점 기준(Plan B 경험)은 없으나, Good 구간 상위권. 원인 분석→대응→결과 개선 논리적 흐름 명확하여 83점으로 산정.",
    
    "behavioral_score": 80,
    "behavioral_pattern": {{
      "pattern_description": "대부분 답변에서 목표 지향적 사고, 문제해결 논리 명확",
      "specific_examples": [
        {{
          "description": "12개 질문 중 9개(75%)에서 '목표', '달성률', '체크' 등 목표 관련 표현 언급",
          "segment_ids": [3, 5, 6, 7, 9, 11],
          "evidence_type": "목표 지향"
        }},
        {{
          "description": "구체적 수치 목표 4회(매출 1억, 달성률 95%, 주간 목표, 판매율 65%)",
          "segment_ids": [6, 7, 9],
          "evidence_type": "숫자 의식"
        }},
        {{
          "description": "'주간 체크', '일일 확인' 같은 모니터링 표현 3회",
          "segment_ids": [6, 7],
          "evidence_type": "모니터링 습관"
        }},
        {{
          "description": "문제→원인→해결 논리 2회 명확(주말 매출 부진 해결, 재고 소진 전략)",
          "segment_ids": [6, 9],
          "evidence_type": "문제해결"
        }}
      ],
      "consistency_note": "Case 질문에서 특히 목표 지향적 답변 명확, Behavioral 질문에서는 정성적 설명 많으나 여전히 결과 확인 습관"
    }},
    "behavioral_reasoning": "Behavioral: 75-89점 구간. 전체 12개 질문 중 9개(75%)에서 목표 관련 표현 언급. 구체적 수치 목표 4회(매출 달성률 95%, 판매율 65%, 주간 목표 달성, 재고회전율 1.5회). '주간 체크', '일일 확인' 같은 모니터링 표현 3회. 문제→원인→해결 논리 2회 명확. Case 질문에서 특히 목표 지향적. Behavioral 질문에서는 정성적 설명 많으나 여전히 결과 확인. 75-89점 기준 충족하여 80점.",
    
    "critical_penalties": -3,
    "red_flags": [
      {{
        "flag_type": "missing_goal",
        "description": "Segment 9에서 '열심히 실행했다'고 했으나 구체적 목표 수치나 달성률 언급 없음",
        "severity": "minor",
        "penalty": -3,
        "evidence_reference": "segment_id: 9, char_index: 2780-2920"
      }}
    ],
    "critical_reasoning": "Critical: Red Flag 1건. Segment 9에서 '열심히 실행했다'고 했으나 구체적 목표 수치나 달성률 언급 없음, 목표 없는 실행(-3점). 총 감점 -3점."
  }},
  
  "overall_score": 83,
  "confidence": {{
    "evidence_strength": 0.85,
    "internal_consistency": 0.97,
    "overall_confidence": 0.90,
    "confidence_note": "증거 충분(Quote 3개), Evidence-Behavioral 간 편차 3점으로 매우 일관적"
  }},
  
  "calculation": {{
    "base_score": 83,
    "evidence_weight": 0.85,
    "behavioral_adjustment": 0.94,
    "adjusted_score": 66.3,
    "amplified_score": 86.2,
    "critical_penalties": -3,
    "final_score": 83.2,
    "formula": "83 × 0.85 × 0.94 × 1.3 - 3 = 83.2 → 83점"
  }},
  
  "strengths": [
    "목표 지향적 사고가 명확함 (전체 질문의 75%)",
    "문제→원인→해결→결과의 논리적 흐름 (2회)",
    "구체적 KPI 수치 자연스럽게 언급 (4회)",
    "주간/일일 모니터링 습관 (3회 언급)"
  ],
  
  "weaknesses": [
    "일부 답변에서 목표 없이 '열심히 했다' 수준",
    "Plan B 또는 전략 수정 경험 부족",
    "사후적 대응보다 선제적 모니터링 보완 필요"
  ],
  
  "key_observations": [
    "신입 치고는 목표 관리 의식 명확 (상위 30% 추정)",
    "Case 질문에서 특히 체계적 문제해결 논리",
    "시즌 전략 경험은 제한적이나 기본 사이클 이해",
    "복잡한 KPI보다 핵심 지표 관리에 강점"
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
8. "문제해결 논리" > "화려한 전략" 우선순위를 유지하세요.
9. 목표 수치 없는 "열심히 했다"는 낮은 점수입니다.
10. 문제 발견 후 원인 분석 없이 바로 액션은 감점입니다.
"""


def create_seasonal_strategy_kpi_evaluation_prompt(
    transcript: str
) -> str:
    """
    Seasonal Strategy & KPI Management Agent 평가 프롬프트 생성
    
    Args:
        transcript: InterviewTranscript의 JSON 문자열
    
    Returns:
        완성된 프롬프트
    """
    return SEASONAL_STRATEGY_KPI_PROMPT.format(
        transcript=transcript
    )