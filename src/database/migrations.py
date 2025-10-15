"""
Database schema migrations for SQLite.
Creates and manages database tables.
"""
import logging
from src.database.connection import DatabaseManager

logger = logging.getLogger(__name__)


def create_schema(db_manager: DatabaseManager):
    """
    Create SQLite database schema (tables, indexes).

    Args:
        db_manager: Database connection manager
    """
    try:
        with db_manager.get_cursor() as cursor:
            # Create contributors table (JSON stored as TEXT)
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

            # JSON indexes using json_extract
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

            logger.info("SQLite database schema created successfully")

    except Exception as e:
        logger.error(f"Failed to create schema: {e}")
        raise


def drop_schema(db_manager: DatabaseManager):
    """
    Drop all tables (use with caution!).

    Args:
        db_manager: Database connection manager
    """
    try:
        with db_manager.get_cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS contributors")
            logger.info("Database schema dropped")

    except Exception as e:
        logger.error(f"Failed to drop schema: {e}")
        raise
