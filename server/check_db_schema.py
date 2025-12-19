#!/usr/bin/env python3
"""
DB 스키마 확인 스크립트
jobs 테이블의 현재 컬럼을 확인합니다.
"""

import asyncio
from sqlalchemy import create_engine, inspect, text
from core.config import settings


def check_jobs_table_schema():
    """jobs 테이블 스키마 확인"""
    print("=" * 80)
    print("DB 연결 및 스키마 확인")
    print("=" * 80)

    try:
        # 엔진 생성
        engine = create_engine(settings.DATABASE_URL)

        # 연결 테스트
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ DB 연결 성공!")

        # 인스펙터 생성
        inspector = inspect(engine)

        # jobs 테이블 존재 확인
        if 'jobs' not in inspector.get_table_names():
            print("❌ jobs 테이블이 존재하지 않습니다.")
            return False

        print("\n✅ jobs 테이블 존재")
        print("\n" + "=" * 80)
        print("현재 jobs 테이블 컬럼:")
        print("=" * 80)

        columns = inspector.get_columns('jobs')

        # RAG Agent 관련 필드 체크
        rag_fields = {
            'required_skills': False,
            'preferred_skills': False,
            'domain_requirements': False,
            'dynamic_evaluation_criteria': False,
            'seniority_level': False,
            'main_responsibilities': False,
            'competency_weights': False,
            'position_type': False
        }

        print(f"\n{'컬럼명':<35} {'타입':<20} {'Nullable'}")
        print("-" * 80)

        for col in columns:
            col_name = col['name']
            col_type = str(col['type'])
            nullable = 'NULL' if col['nullable'] else 'NOT NULL'

            print(f"{col_name:<35} {col_type:<20} {nullable}")

            # RAG 필드 체크
            if col_name in rag_fields:
                rag_fields[col_name] = True

        # RAG 필드 상태 확인
        print("\n" + "=" * 80)
        print("RAG Agent 관련 필드 상태:")
        print("=" * 80)

        missing_fields = []
        for field, exists in rag_fields.items():
            status = "✅" if exists else "❌"
            print(f"{status} {field:<35} {'존재' if exists else '누락'}")
            if not exists:
                missing_fields.append(field)

        if missing_fields:
            print(f"\n⚠️  마이그레이션 필요: {len(missing_fields)}개 필드 누락")
            print(f"   누락된 필드: {', '.join(missing_fields)}")
            return False
        else:
            print(f"\n✅ 모든 RAG Agent 필드가 존재합니다!")
            return True

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    check_jobs_table_schema()
