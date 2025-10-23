"""
Transcript API (옵션)
- 수동 transcript 업로드/조회
- 주로 테스트 및 관리 목적
"""
import logging
from datetime import timezone
from typing import Optional
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from db.database import get_db
from models.interview import InterviewSession, SessionTranscript
from services.storage.s3_service import S3Service
from core.config import settings

router = APIRouter(prefix="/api/v1/transcripts", tags=["transcripts"])
logger = logging.getLogger(__name__)

s3_service = S3Service(
    bucket_name=settings.S3_BUCKET_NAME,
    region_name=settings.AWS_REGION
)


def _to_iso(dt) -> Optional[str]:
    """Ensure datetime is serialized with timezone info (UTC fallback)."""
    if not dt:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat()


def _resolve_s3_key(interview_id: int, s3_url: Optional[str]) -> str:
    """
    Derive S3 key from stored URL when possible; fall back to deterministic key.
    Supports s3://bucket/key or https://bucket.s3.../key forms.
    """
    if not s3_url:
        return f"transcripts/{interview_id}.json"

    parsed = urlparse(s3_url)
    if parsed.scheme in {"s3", "http", "https"}:
        path = parsed.path.lstrip("/")
        if path:
            return path
    return s3_url.lstrip("/")


def _detect_speaker(transcript: SessionTranscript) -> str:
    """
    Speaker 정보 우선순위:
    1) meta_json 내 speaker 필드
    2) 텍스트 포맷 힌트 (Q: 프리픽스)
    """
    meta = transcript.meta_json if isinstance(transcript.meta_json, dict) else {}
    speaker = meta.get("speaker") if meta else None
    if isinstance(speaker, str) and speaker.strip():
        return speaker
    text = (transcript.text or "").strip()
    return "Interviewer" if text.startswith("Q:") else "Applicant"


def _build_transcript_payload(session: InterviewSession, transcripts) -> dict:
    full_transcript_text = "\n".join([t.text for t in transcripts])
    return {
        "interview_id": session.id,
        "applicant_id": session.applicant_id,
        "company_id": session.job.company_id if session.job else None,
        "job_id": session.job_id,
        "started_at": _to_iso(session.started_at),
        "completed_at": _to_iso(session.completed_at),
        "segments": [
            {
                "segment_id": t.id,
                "turn": t.turn,
                "speaker": _detect_speaker(t),
                "text": t.text,
                "timestamp": _to_iso(t.created_at),
                "meta": t.meta_json,
            }
            for t in transcripts
        ],
        "full_transcript": full_transcript_text,
    }


@router.get("/{interview_id}")
async def get_transcript(interview_id: int, db: Session = Depends(get_db)):
    """
    특정 면접의 transcript 조회

    1. DB에 S3 URL이 있으면 → S3에서 다운로드
    2. 없으면 → DB에서 직접 조회하여 반환
    """
    session = db.query(InterviewSession).filter(
        InterviewSession.id == interview_id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="면접 세션을 찾을 수 없습니다.")

    # S3에 저장되어 있는 경우 우선 조회 (키 불일치 방지)
    if session.transcript_s3_url:
        s3_key = _resolve_s3_key(interview_id, session.transcript_s3_url)
        try:
            transcript_data = s3_service.download_json(s3_key)
        except Exception as exc:  # 다운로드 실패 시 로깅 후 DB로 폴백
            logger.warning(
                "Transcript S3 download failed; falling back to DB",
                extra={"interview_id": interview_id, "s3_key": s3_key, "error": str(exc)},
            )
            transcript_data = None

        if transcript_data:
            return {
                "source": "s3",
                "s3_url": session.transcript_s3_url,
                "data": transcript_data
            }

    # S3에 없는 경우 - DB에서 직접 조회
    transcripts = (
        db.query(SessionTranscript)
        .filter(SessionTranscript.session_id == interview_id)
        .order_by(SessionTranscript.turn)
        .all()
    )

    if not transcripts:
        raise HTTPException(status_code=404, detail="Transcript가 없습니다.")

    transcript_data = _build_transcript_payload(session, transcripts)

    return {
        "source": "database",
        "s3_url": None,
        "data": transcript_data
    }


@router.post("/{interview_id}/upload-to-s3")
async def upload_transcript_to_s3(interview_id: int, db: Session = Depends(get_db)):
    """
    DB에 있는 transcript를 수동으로 S3에 업로드
    (주로 마이그레이션 또는 재업로드 목적)
    """
    session = db.query(InterviewSession).filter(
        InterviewSession.id == interview_id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="면접 세션을 찾을 수 없습니다.")

    # 이미 S3에 있는 경우
    if session.transcript_s3_url:
        return {
            "message": "이미 S3에 업로드되어 있습니다.",
            "s3_url": session.transcript_s3_url
        }

    # DB에서 transcript 조회
    transcripts = (
        db.query(SessionTranscript)
        .filter(SessionTranscript.session_id == interview_id)
        .order_by(SessionTranscript.turn)
        .all()
    )

    if not transcripts:
        raise HTTPException(status_code=404, detail="Transcript가 없습니다.")

    # 데이터 구성
    transcript_data = _build_transcript_payload(session, transcripts)

    # S3 업로드
    try:
        s3_key = _resolve_s3_key(interview_id, None)
        s3_uri = s3_service.upload_json(s3_key, transcript_data)

        # DB 업데이트
        session.transcript_s3_url = s3_uri
        db.commit()

        return {
            "message": "S3 업로드 성공",
            "s3_url": s3_uri,
            "segments_count": len(transcripts)
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"S3 업로드 실패: {str(e)}")
