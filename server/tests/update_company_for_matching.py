#!/usr/bin/env python3
"""
Company 데이터 업데이트 - 매칭 점수 계산을 위한 설정
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy.orm import Session
from db.database import SessionLocal
from models.interview import Company

def update_company_data():
    """Company에 가중치 및 핵심 가치 설정"""
    db: Session = SessionLocal()

    try:
        print("\n" + "="*60)
        print("Company 데이터 업데이트")
        print("="*60 + "\n")

        # Company ID 1 업데이트
        company = db.query(Company).filter(Company.id == 1).first()
        if not company:
            print("❌ Company ID 1을 찾을 수 없습니다.")
            return

        # 핵심 가치 설정
        company.core_values = [
            "안전 최우선",
            "기술 혁신",
            "고객 만족",
            "팀워크",
            "지속적 성장"
        ]

        # 카테고리 가중치 설정 (합=1.0)
        company.category_weights = {
            "technical": 0.35,  # 기술 역량 35%
            "cultural": 0.30,   # 문화 적합도 30%
            "experience": 0.20, # 경험 20%
            "soft_skills": 0.15 # 소프트 스킬 15%
        }

        # 우선순위 가중치 설정 (세부 기술/역량별)
        company.priority_weights = {
            "technical": {
                "python": 1.5,
                "fastapi": 1.3,
                "system_design": 1.4,
                "problem_solving": 1.2,
                "default": 1.0
            },
            "cultural": {
                "teamwork": 1.3,
                "communication": 1.2,
                "growth_mindset": 1.2,
                "default": 1.0
            },
            "experience": {
                "backend": 1.3,
                "leadership": 1.2,
                "default": 1.0
            },
            "soft_skills": {
                "communication": 1.2,
                "problem_solving": 1.3,
                "default": 1.0
            }
        }

        db.commit()

        print(f"✓ Company 업데이트 완료: {company.name}")
        print(f"  - 핵심 가치: {len(company.core_values)}개")
        print(f"  - 카테고리 가중치: {company.category_weights}")
        print("\n" + "="*60)
        print("✅ 완료!")
        print("="*60 + "\n")

    except Exception as e:
        db.rollback()
        print(f"\n❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    update_company_data()
