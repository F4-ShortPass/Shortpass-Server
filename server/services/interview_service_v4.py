# 20251122 수정 - 페르소나 3개 순차 면접 지원 + 실시간 꼬리질문
# server/services/interview_service_v4.py

import logging
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
import json
import os
import uuid
import wave
from pathlib import Path
from openai import OpenAI
from utils.s3_uploader import upload_file_and_get_url
from utils.stt_tts_translator import stt_tts_translator
from db.database import SessionLocal
from models.interview import InterviewSession

logger = logging.getLogger(__name__)

class InterviewServiceV4:
    def __init__(self):
        self.example_question_list = ["첫번째 질문입니다", "두번째 질문입니다", "세번째 질문입니다"]
        self.interview_results = []
        self.openai_client = OpenAI()  # OPENAI_API_KEY 환경변수 사용

    async def _evaluate_answer_quality(self, question: str, answer: str, intent: str = None) -> bool:
        """
        LLM으로 답변 품질 판단. 약한 답변이면 True 반환.
        - 답변이 너무 짧거나 (100자 미만)
        - 구체적 사례/수치가 없거나
        - 질문 의도에 맞지 않으면 → 꼬리질문 필요
        """
        answer_stripped = answer.strip()

        # 빠른 체크 1: 너무 짧은 답변
        if len(answer_stripped) < 100:
            # print(f"!!! 답변이 너무 짧음 ({len(answer_stripped)}자) → 꼬리질문 필요")
            print(f"!!! 답변 품질 판단 : WEAK → 꼬리질문 필요")
            return True

        # 빠른 체크 2: 회피/무응답 키워드
        evasive_keywords = ["모르겠", "없어요", "없습니다", "잘 모르", "기억이 안", "생각이 안", "그냥", "별로"]
        answer_lower = answer_stripped.lower()
        for keyword in evasive_keywords:
            if keyword in answer_lower and len(answer_stripped) < 150:
                print(f"!!! 회피성 답변 감지 ('{keyword}') → 꼬리질문 필요")
                return True

        # LLM 판단
        try:
            prompt = f"""다음 면접 질문과 답변을 분석해주세요.

질문: {question}
{f'질문 의도: {intent}' if intent else ''}

답변: {answer}

다음 기준으로 답변의 충실도를 **엄격하게** 판단해주세요:
1. 구체적인 사례나 경험이 포함되어 있는가? (구체적 상황, 시기, 배경 등)
2. 수치나 정량적 결과가 언급되어 있는가? (%, 건수, 기간 등)
3. 질문의 핵심을 제대로 답변했는가? (질문과 관련 없는 답변이면 WEAK)
4. STAR 기법(상황-과제-행동-결과)으로 구조화되어 있는가?
5. 답변이 질문과 관련이 있는가? (엉뚱한 답변이면 WEAK)

위 기준 중 3개 이상 충족하지 못하면 "WEAK", 충족하면 "STRONG"으로만 답변하세요.
애매하면 "WEAK"로 판단하세요."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0
            )

            result = response.choices[0].message.content.strip().upper()
            is_weak = "WEAK" in result
            print(f"답변 품질 판단: {result} → 꼬리질문 {'필요' if is_weak else '불필요'}")
            return is_weak

        except Exception as e:
            print(f"답변 품질 판단 실패: {e}")
            return True  # 에러 시에도 꼬리질문 실행 (더 안전)

    def _load_persona_data(self):
        """
        3개 면접관이 정의된 persona_samsung_fashion.json 불러오기
        """
        try:
            # 우선 samsung_fashion 파일 시도
            persona_file = Path(__file__).resolve().parent.parent / "assets" / "persona_samsung_fashion.json"
            if not persona_file.exists():
                # fallback: persona_data.json
                persona_file = Path(__file__).resolve().parent.parent / "assets" / "persona_data.json"

            if not persona_file.exists():
                print(f"!!! 페르소나 파일이 없습니다")
                return None

            with open(persona_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"페르소나 로드: {persona_file.name}")
                return data
        except Exception as e:
            print(f"!!! 페르소나 데이터 로드 실패: {e}")
            return None

    def _get_interviewers(self, persona_data):
        """
        JSON에서 3개 면접관 정보 추출
        """
        if not persona_data:
            return [{
                "id": "DEFAULT",
                "name": "면접관",
                "type": "기본 면접관",
                "tone": "전문적",
                "focus": ["종합 평가"],
                "questions": self.example_question_list
            }]

        # interviewers 배열이 있으면 그대로 사용
        interviewers = persona_data.get("interviewers", [])
        if interviewers:
            print(f"면접관 {len(interviewers)}명 로드됨")
            return interviewers

        # 없으면 기존 방식으로 fallback
        return [{
            "id": "DEFAULT",
            "name": "면접관",
            "type": "기본 면접관",
            "tone": "전문적",
            "focus": ["종합 평가"],
            "questions": persona_data.get("initial_questions", self.example_question_list)
        }]

    def _load_resume_questions(self, applicant_id: int):
        """
        이력서 기반 맞춤 질문 로드 (interview_questions_{applicant_id}.json)
        """
        try:
            questions_file = Path(__file__).resolve().parent.parent / "test_data" / f"interview_questions_{applicant_id}.json"
            if not questions_file.exists():
                print(f"!!! 이력서 기반 질문 파일 없음: {questions_file.name}")
                return None

            with open(questions_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"이력서 기반 질문 로드: {data.get('applicant_name', 'Unknown')}")
                return data
        except Exception as e:
            print(f"!!! 이력서 질문 로드 실패: {e}")
            return None

    def _merge_resume_questions(self, interviewers, resume_data):
        """
        페르소나 면접관에 이력서 기반 질문 병합
        - 이력서 질문이 있으면 기존 질문 대체
        - follow_up_if_weak도 함께 저장
        """
        if not resume_data:
            return interviewers

        resume_interviewers = resume_data.get("interviewers", [])
        resume_map = {ri.get("id"): ri for ri in resume_interviewers}

        for interviewer in interviewers:
            interviewer_id = interviewer.get("id")
            if interviewer_id in resume_map:
                resume_q = resume_map[interviewer_id].get("resume_based_questions", [])
                if resume_q:
                    # 이력서 기반 질문으로 교체
                    interviewer["questions"] = [q["question"] for q in resume_q]
                    interviewer["follow_ups"] = {q["question"]: q.get("follow_up_if_weak") for q in resume_q}
                    interviewer["resume_context"] = [q.get("related_resume") for q in resume_q]
                    print(f"{interviewer.get('name')}: 이력서 기반 질문 {len(resume_q)}개 적용")

        return interviewers

    async def handle_interview_session(self, websocket: WebSocket, interview_id: int, applicant_id: int = None):
        # 세션별 결과 버퍼 초기화
        self.interview_results = []

        # 0. 페르소나 데이터 로드 (3개 면접관)
        persona_data = self._load_persona_data() or {}
        interviewers = self._get_interviewers(persona_data)
        company_info = persona_data.get("company_info", {})

        # 0-1. 이력서 기반 맞춤 질문 로드 및 병합
        if applicant_id:
            resume_data = self._load_resume_questions(applicant_id)
            if resume_data:
                interviewers = self._merge_resume_questions(interviewers, resume_data)
                print(f"이력서 기반 질문 병합 완료 (applicant_id: {applicant_id})")
                # 병합 결과 확인
                for i in interviewers:
                    print(f"{i.get('name')}: 질문 {len(i.get('questions', []))}개, 꼬리질문 {len(i.get('follow_ups', {}))}개")
        
        # [면접 영상 촬영을 위한 일시 수정] 이컬처 매니저(INT_03)를 맨 앞으로 이동
        # =========================================================================
        # INT_03이 있다면, 리스트 정렬 기준: (ID == 'INT_03' ? 0 : 1)
        # 즉, 0번 그룹(이컬처) -> 1번 그룹(나머지) 순서가 됩니다.
        interviewers.sort(key=lambda x: 0 if x.get("id") == "INT_03" else 1)
        
        print("\n[면접 진행 순서]")
        for i, iv in enumerate(interviewers):
            print(f" {i+1}. {iv.get('name')} ({iv.get('type')}) - {iv.get('id')}")
        print("==========================================\n")

        # 1. 연결 성공 메시지 전송
        await websocket.send_json({
            "type": "connection_success",
            "message": f"WebSocket 연결 성공 (Interview ID: {interview_id})",
            "company": company_info.get("company_name", "기업"),
            "job_title": company_info.get("job_title", "직무")
        })

        # 1-1. 전체 면접관 정보 전송 (프론트 표시용)
        await websocket.send_json({
            "type": "interview_info",
            "interviewers": [
                {
                    "id": i.get("id"),
                    "name": i.get("name"),
                    "role": i.get("role"),
                    "type": i.get("type"),
                    "question_count": len(i.get("questions", []))
                } for i in interviewers
            ],
            "total_interviewers": len(interviewers),
            "total_questions": sum(len(i.get("questions", [])) for i in interviewers)
        })

        # 2. 'start_interview' 신호 대기
        await self._wait_for_start_signal(websocket)

        # 3. 면접관별 순차 진행 (메인 루프)
        global_q_idx = 0
        for interviewer_idx, interviewer in enumerate(interviewers):
            interviewer_name = interviewer.get("name", f"면접관 {interviewer_idx + 1}")
            interviewer_type = interviewer.get("type", "면접관")
            questions = interviewer.get("questions", [])

            print(f"\n{'='*50}")
            print(f"[{interviewer_idx + 1}/{len(interviewers)}] {interviewer_name} ({interviewer_type}) 면접 시작")
            print(f"{'='*50}")

            # 면접관 전환 알림
            await websocket.send_json({
                "type": "interviewer_change",
                "interviewer_index": interviewer_idx,
                "interviewer": {
                    "id": interviewer.get("id"),
                    "name": interviewer_name,
                    "role": interviewer.get("role"),
                    "type": interviewer_type,
                    "tone": interviewer.get("tone"),
                    "focus": interviewer.get("focus", []),
                    "style_description": interviewer.get("style_description", "")
                },
                "question_count": len(questions),
                "message": f"{interviewer_name}님의 면접을 시작합니다."
            })

            # 각 면접관의 질문들 진행
            for q_idx, question_text in enumerate(questions):
                print(f"\n--- [{interviewer_name}] Question {q_idx + 1}/{len(questions)} ---")

                # (1) TTS: 질문 텍스트 -> 오디오 URL 생성
                audio_url = stt_tts_translator.text_to_audio(
                    text=question_text,
                    folder=f"interviews/interview_{interview_id}/questions"
                )

                # (2) 질문 전송 (텍스트 + 오디오 URL)
                payload = {
                    "type": "question_audio" if audio_url else "question",
                    "text": question_text,
                    "interviewer_id": interviewer.get("id"),
                    "interviewer_name": interviewer_name,
                    "question_index": q_idx,
                    "global_index": global_q_idx
                }
                if audio_url:
                    payload["audioUrl"] = audio_url

                await websocket.send_json(payload)
                await websocket.send_json({"type": "question_end"})
                print(f"[Q{q_idx}] 질문 전송 완료 : {question_text}")

                # (3) 답변 대기 및 처리
                user_answer_text = await self._process_user_answer(websocket, interview_id, global_q_idx)

                # STT 결과 전송
                print(f"[Q{q_idx}] STT 결과: {user_answer_text[:30]}...")
                await websocket.send_json({
                    "type": "stt_final",
                    "text": user_answer_text
                })

                # (4) 꼬리질문 판단 및 실행
                follow_ups = interviewer.get("follow_ups", {})
                follow_up_question = follow_ups.get(question_text)

                # 디버그 로그 (상세)
                # print(f"\n  === 꼬리질문 생성 ===")
                # print(f"   꼬리질문 생성 중 ...")
                # print(f"   생성된 꼬리질문: {follow_up_question}")
                # print(f"   ===================\n")

                if follow_up_question:
                    # 답변 품질 판단
                    intent = None
                    resume_context = interviewer.get("resume_context", [])
                    if q_idx < len(resume_context):
                        intent = resume_context[q_idx]

                    is_weak = await self._evaluate_answer_quality(question_text, user_answer_text, intent)

                    if is_weak:
                        print(f"\n  === 꼬리질문 생성 ===")
                        print(f"   꼬리질문 생성 중 ...")
                        print(f"   생성된 꼬리질문: {follow_up_question}")
                        print(f"   =====================\n")
                        
                        print(f"[Q{q_idx}] 꼬리질문 실행: {follow_up_question[:30]}...")

                        # 꼬리질문 TTS 생성
                        follow_up_audio_url = stt_tts_translator.text_to_audio(
                            text=follow_up_question,
                            folder=f"interviews/interview_{interview_id}/questions"
                        )

                        # 꼬리질문 전송
                        follow_up_payload = {
                            "type": "follow_up_question",
                            "text": follow_up_question,
                            "interviewer_id": interviewer.get("id"),
                            "interviewer_name": interviewer_name,
                            "original_question": question_text,
                            "question_index": q_idx,
                            "global_index": global_q_idx
                        }
                        if follow_up_audio_url:
                            follow_up_payload["audioUrl"] = follow_up_audio_url

                        await websocket.send_json(follow_up_payload)
                        await websocket.send_json({"type": "question_end"})

                        # 꼬리질문 답변 수신
                        follow_up_answer = await self._process_user_answer(websocket, interview_id, f"{global_q_idx}_followup")

                        # 꼬리질문 STT 결과 전송
                        print(f"[Q{q_idx}] 꼬리질문 STT 결과: {follow_up_answer[:30]}...")
                        await websocket.send_json({
                            "type": "stt_final",
                            "text": follow_up_answer
                        })

                        # 꼬리질문 결과 저장
                        self.interview_results.append({
                            "global_index": f"{global_q_idx}_followup",
                            "interviewer_id": interviewer.get("id"),
                            "interviewer_name": interviewer_name,
                            "interviewer_type": interviewer_type,
                            "question_index": q_idx,
                            "question": follow_up_question,
                            "answer": follow_up_answer,
                            "is_follow_up": True,
                            "original_question": question_text,
                            "target_competencies": interviewer.get("target_competencies", [])
                        })

                # (5) 결과 저장
                self.interview_results.append({
                    "global_index": global_q_idx,
                    "interviewer_id": interviewer.get("id"),
                    "interviewer_name": interviewer_name,
                    "interviewer_type": interviewer_type,
                    "question_index": q_idx,
                    "question": question_text,
                    "answer": user_answer_text,
                    "target_competencies": interviewer.get("target_competencies", [])
                })

                global_q_idx += 1

            # 면접관 종료 알림
            await websocket.send_json({
                "type": "interviewer_complete",
                "interviewer_index": interviewer_idx,
                "interviewer_name": interviewer_name,
                "message": f"{interviewer_name}님의 면접이 종료되었습니다."
            })

        # 4. 전체 결과 JSON 파일로 저장
        result_s3_url = self._save_results_to_json(interview_id)

        # 5. 인터뷰 종료 신호
        await websocket.send_json({
            "type": "interview_end",
            "message": "모든 면접이 종료되었습니다. 수고하셨습니다.",
            "transcriptUrl": result_s3_url,
            "total_interviewers": len(interviewers),
            "total_questions": global_q_idx,
            "results": self.interview_results
        })
        print(f"인터뷰 세션 종료 (ID: {interview_id})")

    async def _wait_for_start_signal(self, websocket: WebSocket):
        """클라이언트로부터 start_interview 신호를 기다립니다."""
        print("SERVICE: 시작 신호 대기 중...")
        while True:
            message = await websocket.receive()
            if "text" in message:
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "start_interview":
                        print("SERVICE: 'start_interview' 신호 수신.")
                        await websocket.send_json({
                            "type": "ack_start",
                            "message": "면접을 시작합니다."
                        })
                        return # 루프 종료
                except json.JSONDecodeError:
                    print("SERVICE: JSON 파싱 에러")

    async def _process_user_answer(self, websocket: WebSocket, interview_id: int, q_idx: int) -> str:
        """
        [수정됨] PCM16 스트림을 받아서 -> WAV 파일로 변환 저장 -> STT 요청
        """
        print(f"[Q{q_idx}] 답변 수신 대기 중...")
        
        # 1. 오디오 데이터를 메모리에 모으기 위한 버퍼
        audio_frames = bytearray()
        
        while True:
            message = await websocket.receive()
            
            # A. 오디오 데이터(PCM Bytes) 수신 -> 버퍼에 추가
            if "bytes" in message:
                audio_frames.extend(message["bytes"])
            
            # B. 텍스트 신호(답변 끝) 수신 -> 루프 탈출
            if "text" in message:
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "answer_end":
                        print(f"[Q{q_idx}] 답변 종료 신호 수신. 수신한 오디오 데이터 {len(audio_frames)} bytes")
                        break
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    print(f"[Q{q_idx}] Warning: Failed to parse WebSocket message: {str(e)}")
                    # Continue receiving - message might not be JSON or might be malformed
        
        # 2. 모은 PCM 데이터를 WAV 파일로 저장
        # (확장자를 .mp3가 아니라 .wav로 해야 합니다!)
        filename = f"answer_{interview_id}_{q_idx}_{uuid.uuid4()}.wav"
        local_path = f"./{filename}"
        
        # [중요] 프론트엔드와 약속한 설정값 (예: 16kHz, Mono, 16bit)
        CHANNELS = 1          # Mono
        SAMPLE_WIDTH = 2      # 16-bit = 2 bytes
        SAMPLE_RATE = 16000   # 16kHz (프론트 설정과 반드시 일치해야 함!)

        try:
            with wave.open(local_path, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(SAMPLE_WIDTH)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_frames)
                
            print(f"[Q{q_idx}] WAV 파일 저장 완료 ({local_path})")

            # 3. Translator에게 변환 요청 
            transcribed_text = stt_tts_translator.audio_to_text(
                local_path=local_path,
                folder=f"interviews/interview_{interview_id}/answers"
            )
            
            # 파일 삭제
            if os.path.exists(local_path):
                os.remove(local_path)

            if transcribed_text:
                print(f"[Q{q_idx}] 변환된 텍스트: {transcribed_text}")
                return transcribed_text
            else:
                return "(인식 실패)"

        except Exception as e:
            print(f"[Q{q_idx}] 오디오 처리 중 에러: {e}")
            return "(오디오 처리 에러)"
        

    def _save_results_to_json(self, interview_id: int):
        # 면접 결과를 JSON 파일로 저장
        filename = f"interview_result_{interview_id}.json"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.interview_results, f, ensure_ascii=False, indent=4)
            print(f"📂 결과 파일 생성됨: {filename}")
            s3_url = upload_file_and_get_url(
                file_path=filename,
                folder=f"interviews/interview_{interview_id}"
            )

            # DB에 transcript_s3_url 업데이트
            if s3_url:
                try:
                    db = SessionLocal()
                    session = db.query(InterviewSession).filter(InterviewSession.id == interview_id).first()
                    if session:
                        session.transcript_s3_url = s3_url
                        db.commit()
                        print(f"DB 업데이트 완료: transcript_s3_url = {s3_url}")
                    else:
                        print(f"InterviewSession {interview_id} 를 찾을 수 없음")
                    db.close()
                except Exception as db_error:
                    print(f"DB 업데이트 실패: {db_error}")

            return s3_url
        except Exception as e:
            print(f"!!! 결과 저장 실패: {e}")
            return None

interview_service_v4 = InterviewServiceV4()
