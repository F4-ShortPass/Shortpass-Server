"""
Run embedding column migration for RAG implementation
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import text
from dotenv import load_dotenv

# Load .env
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path, override=True)

from db.database import engine


def run_migration():
    """Run the embedding column migration"""

    print("\n" + "="*80)
    print("  RAG 임베딩 컬럼 마이그레이션")
    print("="*80)

    migration_sql = """
    -- 1. Enable pgvector extension (if not already enabled)
    CREATE EXTENSION IF NOT EXISTS vector;

    -- 2. Add embedding column to session_transcripts table
    ALTER TABLE session_transcripts
    ADD COLUMN IF NOT EXISTS embedding vector(1536);

    -- 3. Create HNSW index for efficient vector similarity search
    CREATE INDEX IF NOT EXISTS session_transcripts_embedding_idx
    ON session_transcripts
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
    """

    try:
        with engine.connect() as conn:
            print("\n✓ 데이터베이스 연결 성공")

            # Execute each statement separately
            statements = [s.strip() for s in migration_sql.split(';') if s.strip()]

            for i, statement in enumerate(statements, 1):
                print(f"\n[{i}/{len(statements)}] 실행 중...")
                print(f"  {statement[:80]}...")

                conn.execute(text(statement))
                conn.commit()

                print(f"  ✓ 완료")

            print("\n" + "="*80)
            print("  ✅ 마이그레이션 성공!")
            print("="*80)

            # Verify
            print("\n[검증] embedding 컬럼 확인:")
            result = conn.execute(text("""
                SELECT column_name, data_type, udt_name
                FROM information_schema.columns
                WHERE table_name = 'session_transcripts'
                AND column_name = 'embedding'
            """))

            row = result.fetchone()
            if row:
                print(f"  ✓ embedding 컬럼 존재")
                print(f"    - column_name: {row[0]}")
                print(f"    - data_type: {row[1]}")
                print(f"    - udt_name: {row[2]}")
            else:
                print("  ⚠️  embedding 컬럼을 찾을 수 없습니다")

            # Check index
            print("\n[검증] HNSW 인덱스 확인:")
            result = conn.execute(text("""
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE tablename = 'session_transcripts'
                AND indexname = 'session_transcripts_embedding_idx'
            """))

            row = result.fetchone()
            if row:
                print(f"  ✓ HNSW 인덱스 존재")
                print(f"    - indexname: {row[0]}")
            else:
                print("  ⚠️  HNSW 인덱스를 찾을 수 없습니다")

    except Exception as e:
        print(f"\n❌ 마이그레이션 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_migration()
