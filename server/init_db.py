#!/usr/bin/env python3
# server/init_db.py
"""
Database initialization script.

This script:
1. Creates all tables defined in SQLAlchemy models
2. Enables pgvector extension in PostgreSQL
3. Creates vector indexes for efficient similarity search

Usage:
    python init_db.py
"""
import sys
from sqlalchemy import text
from db.database import engine, Base, check_db_connection
from models import Job, JobChunk, InterviewSession, InterviewResult, Question, PersonaDB, Company, Applicant, ResumeChunk


def enable_pgvector():
    """Enable pgvector extension in PostgreSQL."""
    print("Enabling pgvector extension...")
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()
        print("✓ pgvector extension enabled")
        return True
    except Exception as e:
        print(f"✗ Failed to enable pgvector extension: {e}")
        return False


def create_tables():
    """Create all tables defined in models."""
    print("\nCreating database tables...")
    try:
        Base.metadata.create_all(bind=engine)
        print("✓ All tables created successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to create tables: {e}")
        return False


def create_vector_indexes():
    """Create vector similarity search indexes."""
    print("\nCreating vector indexes...")
    try:
        with engine.connect() as conn:
            # Create HNSW index for job_chunks
            # HNSW (Hierarchical Navigable Small World) is recommended for high-recall scenarios
            conn.execute(text(
                """
                CREATE INDEX IF NOT EXISTS idx_job_chunks_embedding_hnsw
                ON job_chunks
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
                """
            ))

            # Create HNSW index for resume_chunks (이력서/포트폴리오)
            conn.execute(text(
                """
                CREATE INDEX IF NOT EXISTS idx_resume_chunks_embedding_hnsw
                ON resume_chunks
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
                """
            ))

            conn.commit()
        print("✓ Vector indexes created successfully")
        print("  - HNSW index on job_chunks.embedding (cosine similarity)")
        print("  - HNSW index on resume_chunks.embedding (cosine similarity)")
        return True
    except Exception as e:
        print(f"✗ Failed to create vector indexes: {e}")
        print("  Note: You may need to create these indexes manually if you encounter errors")
        return False


def verify_setup():
    """Verify database setup."""
    print("\nVerifying database setup...")
    try:
        with engine.connect() as conn:
            # Check if pgvector is installed
            result = conn.execute(text(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');"
            ))
            pgvector_installed = result.scalar()

            # Check table count
            result = conn.execute(text(
                """
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema = 'public';
                """
            ))
            table_count = result.scalar()

        print(f"✓ pgvector extension: {'installed' if pgvector_installed else 'NOT installed'}")
        print(f"✓ Tables created: {table_count}")
        return True
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False


def main():
    """Main initialization function."""
    print("=" * 60)
    print("Database Initialization Script")
    print("=" * 60)

    # Check database connection
    print("\nChecking database connection...")
    if not check_db_connection():
        print("✗ Database connection failed!")
        print("\nPlease check:")
        print("  1. DATABASE_URL environment variable is set correctly")
        print("  2. PostgreSQL server is running")
        print("  3. Database credentials are correct")
        sys.exit(1)
    print("✓ Database connection successful")

    # Enable pgvector extension
    if not enable_pgvector():
        print("\n⚠ Warning: pgvector extension not enabled")
        print("  You may need superuser privileges to enable extensions")
        print("  Run manually: CREATE EXTENSION IF NOT EXISTS vector;")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Create tables
    if not create_tables():
        print("\n✗ Failed to create tables")
        sys.exit(1)

    # Create vector indexes
    if not create_vector_indexes():
        print("\n⚠ Warning: Vector indexes not created")
        print("  You may need to create these manually for optimal performance")

    # Verify setup
    verify_setup()

    print("\n" + "=" * 60)
    print("Database initialization completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Verify tables: psql -d <database> -c '\\dt'")
    print("  2. Check pgvector: psql -d <database> -c '\\dx vector'")
    print("  3. Test vector search: Query job_chunks with embedding similarity")
    print("\nFor manual index creation, run:")
    print("  CREATE INDEX ON job_chunks USING hnsw (embedding vector_cosine_ops);")


if __name__ == "__main__":
    main()
