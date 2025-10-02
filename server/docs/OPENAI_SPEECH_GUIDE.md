# OpenAI TTS & Whisper 구현 완료 가이드

## ✅ 구현 완료 사항

### 1. TTS (Text-to-Speech) - OpenAI TTS
- **위치**: `server/services/interview_service.py:99-145`
- **기능**:
  - 텍스트를 자연스러운 음성으로 변환
  - MP3 형식으로 스트리밍
  - 1024바이트 청크 단위로 WebSocket 전송
  - 비동기 처리 (`asyncio.to_thread`)

### 2. STT (Speech-to-Text) - OpenAI Whisper
- **위치**: `server/services/interview_service.py:164-208`
- **기능**:
  - 클라이언트로부터 오디오 수신
  - OpenAI Whisper API로 실시간 변환
  - 한국어 음성 인식 지원
  - WebM, MP3 등 다양한 포맷 지원

---

## 🧪 테스트 결과

### TTS 테스트
```
✓ TTS 생성 성공!
  - 오디오 크기: 84,480 bytes
  - 형식: MP3
  - 텍스트: "안녕하세요. OpenAI TTS를 사용한 음성 합성 테스트입니다."
```

### Whisper STT 테스트
```
✓ 음성 인식 성공!
  - 원본: "이것은 음성 인식 테스트입니다."
  - 인식: "이것은 음성인식 테스트입니다."
  - 정확도: 거의 완벽 (띄어쓰기만 약간 차이)
```

---

## 🔧 구성 파일

### 1. 환경 변수 (.env)
```bash
# OpenAI

**⚠️ 보안 경고**:
- 이 API 키는 이미 노출되었습니다
- 작업 완료 후 **반드시** OpenAI 대시보드에서 키를 삭제하고 새로 발급받으세요
- `.env` 파일은 `.gitignore`에 추가되어 있는지 확인하세요

### 2. Config (core/config.py)
```python
# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

---

## 🎵 TTS 음성 옵션

현재 설정: **nova** (여성 목소리)

### 사용 가능한 음성:
- `alloy` - 중립적인 목소리
- `echo` - 남성 목소리
- `fable` - 영국식 남성 목소리
- `onyx` - 깊은 남성 목소리
- `nova` - 여성 목소리 (현재 사용 중)
- `shimmer` - 부드러운 여성 목소리

### 음성 변경 방법:
`server/services/interview_service.py:114`의 `voice` 파라미터 수정:

```python
response = openai_client.audio.speech.create(
    model="tts-1",  # tts-1-hd for higher quality
    voice="shimmer",  # 여기를 변경
    input=text_to_speak,
    response_format="mp3"
)
```

---

##  서버 실행 방법

### 1. 가상환경 활성화
```bash
cd /home/ec2-user/flex/server
source ../venv1/bin/activate
```

### 2. 서버 실행
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. WebSocket 연결
```
ws://[EC2-PUBLIC-IP]:8000/api/v1/ws/interview
```

---

## 📡 프론트엔드 통합

### WebSocket 메시지 흐름

#### 서버 → 클라이언트

**1. 질문 시작**
```json
{
    "type": "question_start",
    "text": "반갑습니다. AI 면접을 시작하겠습니다."
}
```

**2. 오디오 청크** (Binary)
```javascript
// Blob 형식으로 수신됨
event.data instanceof Blob  // true
```

**3. 질문 종료**
```json
{
    "type": "question_end"
}
```

**4. 음성 인식 시작**
```json
{
    "type": "transcribing_start"
}
```

**5. 인식된 답변**
```json
{
    "type": "answer_text",
    "text": "사용자가 답변한 내용"
}
```

#### 클라이언트 → 서버

**1. 오디오 청크 전송** (Binary)
```javascript
websocket.send(audioChunk);  // Blob or ArrayBuffer
```

**2. 답변 완료 신호**
```json
{
    "type": "answer_done"
}
```

### JavaScript 예시

```javascript
const ws = new WebSocket('ws://YOUR_EC2_IP:8000/api/v1/ws/interview');
let mediaRecorder;
let audioChunks = [];

ws.onopen = () => {
    console.log('면접 WebSocket 연결됨');
};

ws.onmessage = async (event) => {
    // JSON 메시지 처리
    if (typeof event.data === 'string') {
        const message = JSON.parse(event.data);

        switch (message.type) {
            case 'question_start':
                console.log('질문:', message.text);
                break;

            case 'question_end':
                console.log('질문 오디오 재생 완료');
                // 이제 사용자 답변 녹음 시작
                startRecording();
                break;

            case 'answer_text':
                console.log('인식된 답변:', message.text);
                break;

            case 'error':
                console.error('에러:', message.message);
                break;
        }
    }

    // 오디오 바이너리 처리
    if (event.data instanceof Blob) {
        // 오디오 재생
        const audioUrl = URL.createObjectURL(event.data);
        const audio = new Audio(audioUrl);
        await audio.play();

        // 메모리 정리
        audio.onended = () => URL.revokeObjectURL(audioUrl);
    }
};

// 사용자 답변 녹음
async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            audioChunks.push(event.data);
            // 실시간 청크 전송
            ws.send(event.data);
        }
    };

    mediaRecorder.onstop = () => {
        // 녹음 완료 신호
        ws.send(JSON.stringify({ type: 'answer_done' }));
    };

    mediaRecorder.start(1000); // 1초마다 청크 생성
}

// 녹음 중지
function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
    }
}
```

---

## 💰 비용 분석

### OpenAI TTS
- **모델**: tts-1 (Standard)
- **가격**: $15 / 1M 문자
- **예상 비용**:
  - 면접 1회당 약 500자 = $0.0075
  - 월 1000회 면접 = $7.50

### OpenAI Whisper
- **모델**: whisper-1
- **가격**: $0.006 / 분
- **예상 비용**:
  - 면접 1회당 평균 10분 = $0.06
  - 월 1000회 면접 = $60

### 총 예상 비용
- **월 1000회 면접**: 약 $67.50
- **면접 1회당**: 약 $0.068 (약 100원)

---

## 🔄 AWS Polly/Transcribe vs OpenAI 비교

| 항목 | AWS Polly/Transcribe | OpenAI |
|------|---------------------|---------|
| **TTS 품질** | 우수 (Neural) | 우수 |
| **STT 정확도** | 우수 | 매우 우수 |
| **지연 시간** | 낮음 | 낮음 |
| **구현 복잡도** | 높음 (IAM, S3 필요) | **낮음** (API 키만 필요) |
| **비용** | TTS: $16/1M자<br>STT: $0.024/분 | TTS: $15/1M자<br>STT: $0.006/분 |
| **한국어 지원** | 우수 | **매우 우수** |

### OpenAI 선택 장점:
1. ✅ IAM 권한 문제 해결 불필요
2. ✅ S3 버킷 불필요 (Whisper)
3. ✅ 구현 코드 단순화
4. ✅ 한국어 인식 정확도 높음
5. ✅ 비용 효율적 (특히 STT)

---

## 🐛 트러블슈팅

### 문제: "Invalid API key"
**해결**:
1. `.env` 파일의 `OPENAI_API_KEY` 확인
2. OpenAI 대시보드에서 키 상태 확인
3. 서버 재시작

### 문제: TTS 오디오가 재생되지 않음
**해결**:
1. 브라우저 콘솔에서 에러 확인
2. 오디오 자동재생 정책 확인
3. MP3 코덱 지원 확인

### 문제: Whisper 인식이 안 됨
**해결**:
1. 오디오 형식 확인 (WebM, MP3, WAV 등)
2. 오디오 크기 확인 (너무 크면 전송 실패)
3. 녹음 권한 확인

### 문제: WebSocket 연결 실패
**해결**:
1. 서버가 실행 중인지 확인
2. 방화벽/보안 그룹에서 8000 포트 열림 확인
3. CORS 설정 확인

---

##  성능 최적화

### TTS 최적화
```python
# 고품질 TTS 사용
model="tts-1-hd"  # 기본: tts-1

# 더 큰 청크 사이즈 (네트워크 안정 시)
chunk_size = 4096  # 기본: 1024
```

### STT 최적화
```python
# 더 작은 오디오 청크 전송 (실시간성 향상)
mediaRecorder.start(500);  # 0.5초마다

# 또는 연속 전송
mediaRecorder.start();  # 최대한 빠르게
```

---

##  보안 권장사항

### 1. API 키 관리
```bash
# ❌ 하드코딩 금지
OPENAI_API_KEY="sk-..."

# ✅ 환경 변수 사용
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

# ✅ .gitignore에 추가
echo ".env" >> .gitignore
```

### 2. API 키 로테이션
- 정기적으로 API 키 교체 (월 1회 권장)
- 키 노출 시 즉시 삭제 및 재발급

### 3. 사용량 모니터링
- OpenAI 대시보드에서 사용량 확인
- 사용량 제한 설정 (예산 초과 방지)

---

## 📝 다음 단계

1. ✅ OpenAI API 키 보안 처리
2. ⬜ 프론트엔드 WebSocket 클라이언트 구현
3. ⬜ 실제 면접 플로우 테스트
4. ⬜ 에러 핸들링 강화
5. ⬜ 로깅 시스템 추가
6. ⬜ (선택) 다국어 지원 (영어, 중국어 등)

---

## 🔗 참고 자료

- [OpenAI TTS API 문서](https://platform.openai.com/docs/guides/text-to-speech)
- [OpenAI Whisper API 문서](https://platform.openai.com/docs/guides/speech-to-text)
- [FastAPI WebSocket 가이드](https://fastapi.tiangolo.com/advanced/websockets/)
- [MediaRecorder API (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/MediaRecorder)

---

**구현 완료일**: 2025-10-29
**구현자**: Claude Code
**버전**: 2.0.0 (OpenAI)
