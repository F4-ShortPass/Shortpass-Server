#!/usr/bin/env python3
"""
기존 지원자 데이터 삭제
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy.orm import Session
from db.database import SessionLocal
from models.interview import Applicant, InterviewSession, InterviewResult

def reset_data():
    """기존 지원자, 세션, 결과 삭제"""
    db: Session = SessionLocal()

    try:
        print("\n기존 데이터 삭제 중...")

        # InterviewResult 삭제
        result_count = db.query(InterviewResult).delete()
        print(f"✓ InterviewResult {result_count}개 삭제")

        # InterviewSession 삭제
        session_count = db.query(InterviewSession).delete()
        print(f"✓ InterviewSession {session_count}개 삭제")

        # Applicant 삭제
        applicant_count = db.query(Applicant).delete()
        print(f"✓ Applicant {applicant_count}개 삭제")

        db.commit()
        print("\n✅ 모든 데이터 삭제 완료!\n")

    except Exception as e:
        db.rollback()
        print(f"\n❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    reset_data()
