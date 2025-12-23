# server/services/evaluation_pipeline_service.py
"""
면접 평가 파이프라인 서비스
- 실제 DB/S3와 EvaluationService를 사용하도록 모의 로직 제거
"""
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from sqlalchemy.orm import Session

from models.interview import InterviewSession, SessionTranscript
from services.evaluation.evaluation_service import EvaluationService, PROMPT_GENERATORS
from services.storage.s3_service import S3Service


class EvaluationPipelineService:
    def __init__(self, s3_service: Any):
        self.s3: Any = s3_service
        self.evaluation_service = EvaluationService()

    async def run_pipeline(
        self,
        company_id: int,
        job_id: int,
        applicant_id: int,
        interview_id: int,
        db: Optional[Session] = None,
    ) -> dict:
        """
        실제 평가 파이프라인 실행:
        1) Transcript/Job/Applicant 정보 로드
        2) EvaluationService 호출
        3) 결과 및 입력을 S3(or Local) 로그로 저장
        """
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        base_key = f"evaluations/{interview_id}/{timestamp}"

        owned_session = False
        if db is None:
            from db.database import SessionLocal
            db = SessionLocal()
            owned_session = True

        try:
            session = self._load_session(db, interview_id, company_id, job_id, applicant_id)
            transcripts = self._load_transcripts(db, interview_id)

            transcript_payload = self._build_transcript_payload(session, transcripts)
            competency_weights = self._build_competency_weights(session.job)

            # 평가 실행
            evaluation_result = await self.evaluation_service.evaluate_interview(
                interview_id=interview_id,
                applicant_id=applicant_id,
                job_id=job_id,
                transcript=transcript_payload,
                competency_weights=competency_weights,
                resume_data=None,
                session_id=session.id if session else None,
            )

            # 로그 저장 (입력/출력 모두)
            self._save_json(transcript_payload, f"{base_key}/transcript.json")
            self._save_json(evaluation_result, f"{base_key}/evaluation_result.json")

            return {
                "message": "Evaluation pipeline completed successfully.",
                "log_path": self.s3.get_log_path(base_key),
            }
        finally:
            if owned_session and db:
                db.close()

    def _load_session(
        self, db: Session, interview_id: int, company_id: int, job_id: int, applicant_id: int
    ) -> InterviewSession:
        session = db.query(InterviewSession).filter(InterviewSession.id == interview_id).first()
        if not session:
            raise ValueError(f"InterviewSession {interview_id} not found")

        # 필수 필드 검증 및 보정
        if session.company_id != company_id:
            raise ValueError(f"Company mismatch for session {interview_id}")
        if session.job_id and session.job_id != job_id:
            raise ValueError(f"Job mismatch for session {interview_id}")
        if session.applicant_id and session.applicant_id != applicant_id:
            raise ValueError(f"Applicant mismatch for session {interview_id}")
        return session

    def _load_transcripts(self, db: Session, interview_id: int) -> Tuple[SessionTranscript, ...]:
        transcripts = (
            db.query(SessionTranscript)
            .filter(SessionTranscript.session_id == interview_id)
            .order_by(SessionTranscript.turn)
            .all()
        )
        if not transcripts:
            raise ValueError(f"No transcripts found for interview {interview_id}")
        return tuple(transcripts)

    def _build_transcript_payload(
        self, session: InterviewSession, transcripts: Tuple[SessionTranscript, ...]
    ) -> Dict[str, Any]:
        full_text = "\n".join([t.text for t in transcripts])
        return {
            "interview_id": session.id,
            "applicant_id": session.applicant_id,
            "company_id": session.company_id,
            "job_id": session.job_id,
            "started_at": self._to_iso(session.started_at),
            "completed_at": self._to_iso(session.completed_at),
            "segments": [
                {
                    "segment_id": t.id,
                    "turn": t.turn,
                    "speaker": self._detect_speaker(t),
                    "text": t.text,
                    "timestamp": self._to_iso(t.created_at),
                    "meta": t.meta_json,
                }
                for t in transcripts
            ],
            "full_transcript": full_text,
        }

    def _build_competency_weights(self, job) -> Dict[str, float]:
        """
        가중치 생성:
        - job.dynamic_evaluation_criteria에 weight 필드가 있으면 사용
        - 없으면 PROMPT_GENERATORS 키 기준 균등 가중치
        """
        weights: Dict[str, float] = {}

        if job and getattr(job, "dynamic_evaluation_criteria", None):
            for comp in job.dynamic_evaluation_criteria:
                name = comp.get("id") or comp.get("name")
                if not name:
                    continue
                weight = float(comp.get("weight", 1.0))
                weights[name] = weight

        if not weights:
            uniform = 1.0 / len(PROMPT_GENERATORS)
            weights = {name: uniform for name in PROMPT_GENERATORS.keys()}

        # normalize
        total = sum(weights.values()) or 1.0
        return {k: v / total for k, v in weights.items()}

    @staticmethod
    def _detect_speaker(transcript: SessionTranscript) -> str:
        meta = transcript.meta_json if isinstance(transcript.meta_json, dict) else {}
        speaker = meta.get("speaker")
        if isinstance(speaker, str) and speaker.strip():
            return speaker
        text = (transcript.text or "").strip()
        return "Interviewer" if text.startswith("Q:") else "Applicant"

    @staticmethod
    def _to_iso(dt) -> Optional[str]:
        if not dt:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat()

    def _save_json(self, payload: Dict[str, Any], key: str) -> None:
        """
        s3_service 인터페이스에 따라 JSON 저장. 실패 시 예외로 전파.
        """
        if isinstance(self.s3, S3Service):
            # 실제 S3 서비스인 경우
            self.s3.upload_json(key, payload)
        elif hasattr(self.s3, "save_json_log"):
            self.s3.save_json_log(payload, key)
        else:
            raise RuntimeError("Unsupported S3 service interface")
