"""
Database schema migrations for PostgreSQL and SQLite.
Creates and manages database tables with support for both databases.
"""
import logging
from src.database.connection import DatabaseManager

logger = logging.getLogger(__name__)


def create_schema(db_manager: DatabaseManager):
    """
    Create database schema (tables, indexes).
    Automatically adapts to PostgreSQL or SQLite.

    Args:
        db_manager: Database connection manager
    """
    try:
        with db_manager.get_cursor() as cursor:
            if db_manager.is_postgres:
                _create_postgres_schema(cursor)
            elif db_manager.is_sqlite:
                _create_sqlite_schema(cursor)
            else:
                raise ValueError("Unsupported database type")
            
            logger.info(f"Database schema created successfully ({'PostgreSQL' if db_manager.is_postgres else 'SQLite'})")

    except Exception as e:
        logger.error(f"Failed to create schema: {e}")
        raise


def _create_postgres_schema(cursor):
    """Create PostgreSQL schema with JSONB support."""
    # Create contributors table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS contributors (
            email VARCHAR(255) PRIMARY KEY,
            contributor_id VARCHAR(50) UNIQUE NOT NULL,
            processed_data JSONB NOT NULL,
            intelligence_summary TEXT,
            processing_status VARCHAR(20) DEFAULT 'pending',
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            intelligence_extracted_at TIMESTAMP
        )
    """)

    # Create indexes for performance
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_processing_status
        ON contributors(processing_status)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_active_90d
        ON contributors((processed_data->'activity_summary'->>'is_active_90d'))
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_country
        ON contributors((processed_data->'location'->>'country'))
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_contributor_id
        ON contributors(contributor_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_updated_at
        ON contributors(updated_at DESC)
    """)


def _create_sqlite_schema(cursor):
    """Create SQLite schema with JSON stored as TEXT."""
    # Create contributors table (TEXT for JSON instead of JSONB)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS contributors (
            email TEXT PRIMARY KEY,
            contributor_id TEXT UNIQUE NOT NULL,
            processed_data TEXT NOT NULL,
            intelligence_summary TEXT,
            processing_status TEXT DEFAULT 'pending',
            error_message TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            intelligence_extracted_at TEXT
        )
    """)

    # Create indexes for performance
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_processing_status
        ON contributors(processing_status)
    """)

    # SQLite JSON indexes use json_extract
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_active_90d
        ON contributors(json_extract(processed_data, '$.activity_summary.is_active_90d'))
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_country
        ON contributors(json_extract(processed_data, '$.location.country'))
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_contributor_id
        ON contributors(contributor_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_updated_at
        ON contributors(updated_at DESC)
    """)


def drop_schema(db_manager: DatabaseManager):
    """
    Drop all tables (use with caution!).

    Args:
        db_manager: Database connection manager
    """
    try:
        with db_manager.get_cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS contributors CASCADE")
            logger.info("Database schema dropped")

    except Exception as e:
        logger.error(f"Failed to drop schema: {e}")
        raise
