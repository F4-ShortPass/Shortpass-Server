"""
Embedding 차원 변경: 1536 → 1024 (Bedrock Titan)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import text
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path, override=True)

from db.database import engine


def migrate_to_1024():
    """Embedding 차원을 1024로 변경"""
    
    print("\n" + "="*80)
    print("  Embedding 차원 변경: 1536 → 1024 (Bedrock Titan)")
    print("="*80)
    
    migration_sql = """
    -- 1. 기존 컬럼 삭제
    ALTER TABLE session_transcripts DROP COLUMN IF EXISTS embedding;
    
    -- 2. 1024차원 컬럼 추가
    ALTER TABLE session_transcripts ADD COLUMN embedding vector(1024);
    
    -- 3. 기존 인덱스 삭제
    DROP INDEX IF EXISTS session_transcripts_embedding_idx;
    
    -- 4. 새 HNSW 인덱스 생성
    CREATE INDEX session_transcripts_embedding_idx
    ON session_transcripts
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
    """
    
    try:
        with engine.connect() as conn:
            print("\n✓ 데이터베이스 연결 성공")
            
            statements = [s.strip() for s in migration_sql.split(';') if s.strip()]
            
            for i, statement in enumerate(statements, 1):
                print(f"\n[{i}/{len(statements)}] 실행 중...")
                conn.execute(text(statement))
                conn.commit()
                print(f"  ✓ 완료")
            
            print("\n" + "="*80)
            print("  ✅ 마이그레이션 성공!")
            print("="*80)
            
    except Exception as e:
        print(f"\n❌ 마이그레이션 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    migrate_to_1024()
