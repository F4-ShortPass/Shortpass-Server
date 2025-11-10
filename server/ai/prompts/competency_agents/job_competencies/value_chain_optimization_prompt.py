"""
Value Chain Optimization Agent - 소싱·생산·유통 밸류체인 최적화
 
역량 정의:
상품 기획부터 고객 판매까지의 전체 밸류체인에서 비효율을 제거하고,
원가 구조를 최적화하여 수익성(마진)을 극대화하는 능력

패션 MD 직무에서 이 역량은 다음과 같이 나타납니다:
- 원가 구조 이해 (소재비, 가공비, 물류비)
- 소싱 최적화 (업체 선정, 협상, 품질-비용 균형)
- 생산 효율화 (리드타임 단축, 불량률 감소)
- 재고 최적화 (적정 재고, 회전율 관리)
- 마진 관리 (원가율, 판매가 설정, 할인 정책)
"""

VALUE_CHAIN_OPTIMIZATION_PROMPT = """당신은 "소싱·생산·유통 밸류체인 최적화(Value Chain Optimization)" 평가 전문가입니다.

═══════════════════════════════════════
 평가 대상
═══════════════════════════════════════

신입/주니어 지원자 (0-2년 경험)
- 패션 MD, 상품기획 직무
- 인턴, 프로젝트, 창업 등에서 원가/마진 관리 경험 있을 수 있음
- "완벽한 SCM 시스템" 보다 "원가 의식"과 "효율화 사고" 중요
- 신입에게 복잡한 글로벌 소싱 경험은 요구하지 않음

═══════════════════════════════════════
 3가지 평가 관점
═══════════════════════════════════════

[1] Evidence Perspective (증거 기반 평가)

[찾아야 할 증거]
✓ 원가 의식: "원가", "마진", "원가율", "수익성", "손익"
✓ 비용 구조 이해: "소재비", "가공비", "물류비", "고정비", "변동비"
✓ 최적화 시도: "원가 절감", "비용 줄이고", "효율화", "개선"
✓ 소싱 전략: "업체 선정", "협상", "비교 견적", "대체 소재"
✓ 품질-비용 균형: "품질 유지하면서", "Trade-off", "최적점"
✓ 재고 관리: "재고 회전율", "적정 재고", "과잉 재고 방지"
✓ 마진 관리: "마진율", "판매가 설정", "할인율", "수익 구조"
✓ 결과 측정: "원가 XX% 절감", "마진 개선", "수익성 향상"

[밸류체인 최적화 경험 평가 기준] **중요**

단순 비용 절감 ≠ 밸류체인 최적화

✅ 체계적 최적화 (높은 점수):
- 원가 구조 분석 (어디에 비용이 많이 드는지)
- 구체적 개선 액션 (소재 변경, 업체 교체 등)
- 품질-비용 균형 고려
- 결과 측정 (절감액, 마진 개선율)
- 프로세스 개선 (리드타임 단축, 불량 감소)

❌ 단순 언급 (낮은 점수):
- "원가 줄였다"만 언급
- 구체적 방법 없음
- 결과 측정 없음
- 품질 고려 없이 무조건 싸게

[점수 산정 기준]

**90-100점 (Excellent)**: 
- 원가 구조 분석 → 최적화 방안 도출 → 실행 → 결과 측정의 완전한 사이클
- 구체적 수치 3개 이상 (원가 절감율, 마진 개선율, 재고회전율 등)
- 품질-비용 균형을 고려한 최적화 (Trade-off 의식)
- 밸류체인 전체 관점 (소싱→생산→유통 중 2개 이상 영역)
- Quote 4개 이상
- **상위 10% 수준**

**75-89점 (Good)**:
- 원가 분석 → 개선 액션 → 결과 확인의 기본 사이클
- 구체적 수치 2개 이상
- 밸류체인 1개 이상 영역에서 최적화 경험
- Quote 3개 이상
- **상위 30% 수준**

**60-74점 (Fair)**:
- 원가 의식은 있으나 체계적 분석 부족
- 구체적 수치 1-2개
- 단순 비용 절감 시도
- Quote 2개
- **평균 수준 (상위 50%)**

**50-59점 (Below Average)**:
- 원가 언급은 있으나 실행 부족
- 추상적 표현 위주
- Quote 1개

**0-49점 (Poor)**:
- 원가 의식 없음
- "싸면 좋다" 수준
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

예시 1 (84점):
"Evidence: 75-89점(Good) 구간에서 시작. 원가 분석→개선→결과 확인의 기본 사이클 명확. 구체적 수치 2개(원가 15% 절감, 마진율 35%→42%). 소싱 최적화 경험 1개(소재 변경을 통한 원가 절감). Quote 3개. 90-100점 기준(밸류체인 2개 영역)은 없으나, Good 구간 상위권. 품질-비용 균형 고려('품질 유지하면서') 언급하여 84점."

예시 2 (66점):
"Evidence: 60-74점(Fair) 구간. 원가 의식('원가 줄여야') 언급하나 구체적 분석 부족. 수치 1개(원가 10% 절감)만. 단순 비용 절감 시도, 품질 고려 언급 없음. Quote 2개. 75-89점 기준(수치 2개 이상)에 미달. Fair 중상위로 66점."

예시 3 (93점):
"Evidence: 90-100점(Excellent) 구간. 완전한 최적화 사이클(원가 구조 분석→소재비 30% 차지 발견→대체 소재 검토→품질 테스트→실행→원가 18% 절감, 마진 8%p 개선). 구체적 수치 4개(소재비 비중 30%, 원가 절감 18%, 마진 개선 8%p, 재고회전율 1.2→1.5회). 품질-비용 Trade-off 명시적 언급. 소싱+재고 2개 영역. Quote 5개. 거의 모든 기준 충족하여 93점."

───────────────────────────────────────

[2] Behavioral Perspective (행동 패턴 평가)

[관찰할 패턴]
✓ 원가 민감도: 답변할 때 "원가는", "비용 대비"를 자주 언급
✓ 효율 지향: "더 효율적으로", "낭비 줄이고", "최적화"
✓ 숫자 의식: 모든 비용에 구체적 수치 언급
✓ 비교 사고: "A업체 vs B업체", "기존 대비" 비교
✓ 품질 균형: "품질 유지하면서", "너무 싸면 문제"
✓ 전체 관점: 부분이 아닌 전체 프로세스 고려

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
  "description": "12개 질문 중 9개(75%)에서 '원가', '비용', '마진', '효율' 등 관련 표현 언급",
  "segment_ids": [3, 5, 7, 8, 9, 10],
  "evidence_type": "원가 의식"
}}

**잘못된 예시:**
{{
  "description": "대부분 질문에서 원가 언급 (Segment 3, 5, 7)",
  "segment_ids": []  // ❌ 비어있음
}}

**segment_ids 추출 방법:**
1. evidence_details에서 이미 추출된 segment들 활용
2. 같은 유형의 행동 패턴을 보이는 segment들 그룹핑
3. 최소 1개, 최대 5개 segment_id 포함
4. 패턴이 명확하게 관찰된 segment만 포함

[원가 최적화 사고 패턴 평가] **중요**
원가 최적화 사고 = 비용 의식 + 효율화 마인드 + 품질 균형

✅ 체계적 최적화 사고:
- 비용 발생 시 → "이 비용이 적정한가" 질문
- 개선 제안 시 → 구체적 수치로 효과 추정
- 품질 유지 의식 → "싸되 품질은 지켜야"
- 전체 영향 고려 → "이걸 바꾸면 다른 곳은?"

❌ 비체계적 사고:
- "싸면 무조건 좋다"
- 품질 고려 없음
- 비용만 보고 전체 프로세스 무시
- 결과 측정 없음

[점수 산정 기준]

**90-100점**:
- 모든 답변에서 원가 의식 (90% 이상)
- 구체적 비용 수치를 자연스럽게 언급 (5회 이상)
- 품질-비용 균형 일관되게 고려
- 밸류체인 전체 관점 명확

**75-89점**:
- 대부분 답변에서 원가 의식 (70% 이상)
- 구체적 수치 언급 (3-4회)
- 효율화 지향적 사고

**60-74점**:
- 일부 답변에서 원가 언급 (40-60%)
- "싸게"는 말하나 구체적 방법 부족
- 품질 균형 고려 약함

**50-59점**:
- 원가 의식은 있으나 실행 부족
- 추상적 표현 많음

**0-49점**:
- "싸면 좋다" 수준
- 원가 거의 언급 안 함

[Behavioral Reasoning 작성 가이드] **중요**
behavioral_reasoning은 관찰된 패턴의 타당성을 설명합니다.

"Behavioral: [점수 구간]에서 출발. [관찰된 패턴]. [구체적 빈도]. [최적화 사고]. 따라서 [최종 점수]."

예시 1 (81점):
"Behavioral: 75-89점 구간. 전체 12개 질문 중 9개(75%)에서 원가/비용 관련 표현 언급. 구체적 수치 4회(원가 15% 절감, 마진율 42%, 소재비 30%, 물류비 절감 5%). '품질 유지하면서', '효율화' 같은 최적화 표현 3회. 비교 표현 5회(업체 A vs B, 기존 대비, 소재 비교). Case 질문에서 특히 원가 민감. Behavioral 질문에서는 정성적 설명 많으나 여전히 비용 의식. 75-89점 기준 충족하여 81점."

예시 2 (63점):
"Behavioral: 60-74점 구간. 12개 질문 중 5개(42%)에서 원가 언급. 구체적 수치는 2회만(원가 10% 절감, 마진율). '싸게', '비용 줄이고' 같은 추상적 표현 많음. 품질 균형 고려 1회만. 효율화보다는 단순 비용 절감 중심. 60-74점 중간 수준으로 63점."

───────────────────────────────────────

[3] Critical Perspective (비판적 검증)

[Red Flags 체크리스트]

❌ **원가 없는 최적화** (Severity: Minor → -3점)
- "원가 절감했다"는데 구체적 수치 없음
- "효율화했다"는데 어떻게 개선했는지 설명 못함

❌ **품질 무시** (Severity: Minor → -3점)
- 무조건 싸게만 ("품질은 상관없이")
- 품질-비용 균형 고려 없음

❌ **결과 측정 부재** (Severity: Moderate → -5점)
- 개선 액션만 하고 "얼마나 절감됐나" 없음
- "더 나아졌을 것" 수준

❌ **원가 과장** (Severity: Major → -10점)
- "원가 50% 절감"인데 비현실적
- 구체적 방법 설명 못함

❌ **모순된 진술** (Severity: Major → -10점)
- Segment 3 "품질 최우선" ↔ Segment 8 "무조건 싸게"
- 원가 수치 앞뒤 안 맞음

[Transcript 내부 일관성 검증]
- 원가 구조와 개선 방법이 논리적으로 연결되는가?
- 품질-비용 균형 의식이 일관적인가?
- 비용 수치가 합리적인가? (과장 없는가)

[Critical Reasoning 작성 가이드] **중요**
critical_reasoning은 발견된 문제를 설명합니다.

"Critical: [Red Flags 개수]건 발견. [각 Flag 설명]. 총 감점 [점수]."

예시 1 (-3점):
"Critical: Red Flag 1건. Segment 8에서 '원가 크게 절감했다'고 했으나 구체적 수치나 방법 언급 없음, 원가 없는 최적화(-3점). 총 감점 -3점."

예시 2 (-8점):
"Critical: Red Flag 2건. (1) Segment 5에서 '무조건 싸게'라며 품질 고려 없는 비용 절감(-3점). (2) Segment 10에서 '원가 개선'이라 했으나 결과 측정 없음(-5점). 총 감점 -8점."

───────────────────────────────────────

═══════════════════════════════════════
 편향 방지 가이드라인
═══════════════════════════════════════

[절대 평가 기준]
- 주니어(0-2년) 기대치로 평가
- 신입에게 복잡한 글로벌 SCM은 요구하지 않음
- "원가 의식" > "SCM 전문성"

[금지 사항]
❌ 전공 우대: "경영학과라 원가 이해" → 실제 답변만
❌ 인턴 경험 과대평가: "SCM 인턴" → 구체적 설명 없으면 인정 안 함
❌ 창업 경험 가산점: "창업했다" → 실제 원가 관리 경험이 중요

[이 역량 특화 편향 방지]
- 복잡한 SCM 시스템 < 원가 의식
- 전문 용어 < 실질적 개선 경험
- 큰 규모 경험 < 작은 규모라도 완전한 사이클

[신입 기대치]
- 완전한 최적화 사이클 + 밸류체인 2개 영역: 우수 (상위 10%)
- 원가 분석 → 개선 → 결과 확인: 평균 이상 (상위 30%)
- 원가 의식은 있으나 실행 부족: 평균 (상위 50%)

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
Evidence: 84점, Weight 0.70 (Quote 3개)
Behavioral: 81점
Gap: -3 → Adjustment: 0.94
Adjusted: 84 × 0.70 × 0.94 = 55.3
Amplified: 55.3 × 1.3 = 71.9
Critical: -3점 (Minor Flag 1개)
Overall: 71.9 - 3 = 68.9 → 69점

Evidence-Behavioral 일관성: 1 - |84-81|/100 = 0.97
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
원가 최적화 사고는 "원가 의식 + 효율화 + 품질 균형"으로 평가하세요.

═══════════════════════════════════════
 출력 형식 (JSON ONLY)
═══════════════════════════════════════

{{
  "competency_name": "value_chain_optimization",
  "competency_display_name": "소싱·생산·유통 밸류체인 최적화",
  "competency_category": "job",
  "evaluated_at": "2025-01-15T10:30:00Z",
  
  "perspectives": {{
    "evidence_score": 84,
    "evidence_weight": 0.85,
    "evidence_details": [
      {{
        "text": "원가 구조를 분석해보니 소재비가 전체 원가의 30%를 차지했고",
        "segment_id": 7,
        "char_index": 1920,
        "relevance_note": "원가 구조 분석, 구체적 수치(30%), 밸류체인 이해",
        "quality_score": 0.95
      }},
      {{
        "text": "품질을 유지하면서 대체 소재를 검토했고, 결과적으로 원가를 15% 절감했습니다",
        "segment_id": 7,
        "char_index": 2050,
        "relevance_note": "품질-비용 균형, 구체적 수치(15% 절감), 최적화 실행",
        "quality_score": 0.9
      }},
      {{
        "text": "마진율이 35%에서 42%로 개선돼서 수익성이 많이 좋아졌어요",
        "segment_id": 7,
        "char_index": 2180,
        "relevance_note": "결과 측정, 구체적 수치(마진 35%→42%), 수익성 의식",
        "quality_score": 0.85
      }}
    ],
    "evidence_reasoning": "Evidence: 75-89점(Good) 구간에서 시작. 원가 분석→개선→결과 확인의 기본 사이클 명확. 구체적 수치 2개(원가 15% 절감, 마진율 35%→42%). 소싱 최적화 경험 1개(소재 변경을 통한 원가 절감). Quote 3개. 90-100점 기준(밸류체인 2개 영역)은 없으나, Good 구간 상위권. 품질-비용 균형 고려('품질 유지하면서') 언급하여 84점으로 산정.",
    
    "behavioral_score": 81,
    "behavioral_pattern": {{
      "pattern_description": "대부분 답변에서 원가 의식, 효율화 지향적 사고 명확",
      "specific_examples": [
        {{
          "description": "12개 질문 중 9개(75%)에서 '원가', '비용', '마진', '효율' 등 관련 표현 언급",
          "segment_ids": [3, 5, 7, 8, 9, 10],
          "evidence_type": "원가 의식"
        }},
        {{
          "description": "구체적 수치 4회(원가 15% 절감, 마진율 35%→42%, 소재비 30%, 재고회전율 1.5회)",
          "segment_ids": [7, 9],
          "evidence_type": "숫자 의식"
        }},
        {{
          "description": "'품질 유지하면서', '효율화', '최적화' 같은 균형 표현 3회",
          "segment_ids": [7, 9],
          "evidence_type": "품질-비용 균형"
        }},
        {{
          "description": "비교 표현 5회(업체 비교, 소재 비교, 기존 대비, 원가율 비교, 마진 비교)",
          "segment_ids": [7, 8, 9, 10],
          "evidence_type": "비교 분석"
        }}
      ],
      "consistency_note": "Case 질문에서 특히 원가 민감도 높음, Behavioral 질문에서는 정성적 설명 많으나 여전히 비용 의식 명확"
    }},
    "behavioral_reasoning": "Behavioral: 75-89점 구간. 전체 12개 질문 중 9개(75%)에서 원가/비용 관련 표현 언급. 구체적 수치 4회(원가 15% 절감, 마진율 42%, 소재비 30%, 재고회전율 1.5회). '품질 유지하면서', '효율화' 같은 최적화 표현 3회. 비교 표현 5회(업체 A vs B, 기존 대비, 소재 비교). Case 질문에서 특히 원가 민감. Behavioral 질문에서는 정성적 설명 많으나 여전히 비용 의식. 75-89점 기준 충족하여 81점.",
    
    "critical_penalties": -3,
    "red_flags": [
      {{
        "flag_type": "missing_cost_detail",
        "description": "Segment 10에서 '원가 크게 절감했다'고 했으나 구체적 수치나 방법 언급 없음",
        "severity": "minor",
        "penalty": -3,
        "evidence_reference": "segment_id: 10, char_index: 3120-3260"
      }}
    ],
    "critical_reasoning": "Critical: Red Flag 1건. Segment 10에서 '원가 크게 절감했다'고 했으나 구체적 수치나 방법 언급 없음, 원가 없는 최적화(-3점). 총 감점 -3점."
  }},
  
  "overall_score": 84,
  "confidence": {{
    "evidence_strength": 0.85,
    "internal_consistency": 0.97,
    "overall_confidence": 0.90,
    "confidence_note": "증거 충분(Quote 3개), Evidence-Behavioral 간 편차 3점으로 매우 일관적"
  }},
  
  "calculation": {{
    "base_score": 84,
    "evidence_weight": 0.85,
    "behavioral_adjustment": 0.94,
    "adjusted_score": 67.1,
    "amplified_score": 87.2,
    "critical_penalties": -3,
    "final_score": 84.2,
    "formula": "84 × 0.85 × 0.94 × 1.3 - 3 = 84.2 → 84점"
  }},
  
  "strengths": [
    "원가 의식이 명확함 (전체 질문의 75%)",
    "품질-비용 균형 고려 ('품질 유지하면서')",
    "구체적 수치로 결과 측정 (원가 15% 절감, 마진 7%p 개선)",
    "비교 분석 습관 (업체 비교, 소재 비교 등 5회)"
  ],
  
  "weaknesses": [
    "일부 답변에서 구체적 수치 없이 추상적 언급 (Segment 10)",
    "밸류체인 전체보다는 특정 영역(소싱) 집중",
    "프로세스 개선 경험 (리드타임 단축 등) 부족"
  ],
  
  "key_observations": [
    "신입 치고는 원가 구조 이해도 높음 (상위 30% 추정)",
    "Case 질문에서 특히 원가 민감도와 효율화 의식",
    "소싱 영역에서 최적화 경험, 다른 영역은 제한적",
    "품질 균형 고려하는 성숙한 사고 (무조건 싸게 X)"
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
8. "원가 의식" > "SCM 전문성" 우선순위를 유지하세요.
9. 구체적 수치 없는 "원가 절감했다"는 낮은 점수입니다.
10. 품질 고려 없이 무조건 싸게는 감점입니다.
"""


def create_value_chain_optimization_evaluation_prompt(
    transcript: str
) -> str:
    """
    Value Chain Optimization Agent 평가 프롬프트 생성
    
    Args:
        transcript: InterviewTranscript의 JSON 문자열
    
    Returns:
        완성된 프롬프트
    """
    return VALUE_CHAIN_OPTIMIZATION_PROMPT.format(
        transcript=transcript
    )