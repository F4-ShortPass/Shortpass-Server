# server/db/database.py
# @@지원
from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
from core.config import settings

# Create SQLAlchemy engine with connection pooling
engine = create_engine(
    settings.DATABASE_URL,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_timeout=settings.DB_POOL_TIMEOUT,
    pool_recycle=settings.DB_POOL_RECYCLE,
    echo=settings.DB_ECHO,
    pool_pre_ping=True  # Verify connections before using
)

# Create SessionLocal class for database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class for SQLAlchemy models
Base = declarative_base()


def get_db():
    """
    Dependency function to get database session.
    Used in FastAPI route dependencies.

    Yields:
        Session: SQLAlchemy database session

    Example:
        @app.get("/items/")
        def read_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize database by creating all tables.
    Should be called on application startup.

    Note:
        Requires pgvector extension to be enabled in PostgreSQL:
        CREATE EXTENSION IF NOT EXISTS vector;
    """
    # Import all models here to ensure they are registered with Base
    from models import job, interview

    # Create all tables
    Base.metadata.create_all(bind=engine)


def check_db_connection():
    """
    Check if database connection is healthy.

    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False
