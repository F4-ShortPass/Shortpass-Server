 - 루트: 인터뷰 정보가 담긴 JSON 객체여야 하고, segments 배열이 필수입니다.
  - 메타데이터(권장): interview_id, candidate_id, conducted_at, total_duration_sec 등 기본 
    인터뷰 식별/시간 정보.
  - 각 segments 원소(Segment 단위 필수 값):
      - segment_id(고유 식별자)
      - question_text(질문)
      - answer_text(지원자 답변, STT 변환본)
      - 근거 위치 추적용 char_index_start/char_index_end (프롬프트에서 quote 위치를 요구)  
  - 추가로 자주 포함되는 값: segment_order, answer_duration_sec 등 시간/순서 정보.

# char_index_start / char_index_end
  - 역할: 답변 텍스트(answer_text) 안에서 근거가 되는 구간의 시작·끝 위치를 0‑based 인덱스 
    로 표시합니다.
  - 이유: 평가 결과에서 “segment_id X, char_index 1200-1400”처럼 근거 위치를 기록해 HUD/프 
    런트엔드에서 바로 하이라이트(예: <mark>) 할 수 있게 하려는 것.
  - 예시: answer_text가 "Python으로 분석했고 ROI를 계산했습니다"일 때 "Python으로 분석" 구 
    간이 문장 전체 0~28자이면 start=0, end=28로 저장.

# segment_id vs segment_order
  - segment_id: 각 발화(질문+답변) 블록의 고유 식별자. 평가 결과나 로그에서 특정 발화를 참 
    조할 때 사용합니다.
  - segment_order: 면접 진행 순서를 보존하기 위한 정렬 키. STT 후 병합/필터링 등으로       
    segment_id가 불연속이거나 재사용되는 경우에도 원래 면접 흐름 순서를 재구성할 수 있게 별
    도로 둡니다.
  - 정리: segment_id는 “누구(어떤 블록)인가”를 가리키는 안정 ID, segment_order는 “어느 순서
    로 말했는가”를 보존하기 위한 정렬용 값입니다.
