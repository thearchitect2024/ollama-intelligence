"""
SQLite database connection management.
Provides simple, file-based database connectivity for SageMaker Studio.
"""
import sqlite3
from contextlib import contextmanager
from typing import Generator, Optional
import logging

from src.config import Settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages SQLite database connections.
    Simple file-based database for testing and development.
    """

    def __init__(self, settings: Settings):
        """
        Initialize database manager.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.db_url = settings.get_database_url()
        
        # Extract SQLite file path from URL
        self._sqlite_path = self.db_url.replace('sqlite:///', '')
        logger.info(f"SQLite database configured: {self._sqlite_path}")
        
        # Test connection
        self._verify_connection()

    def _verify_connection(self):
        """Verify SQLite database is accessible."""
        try:
            with self._get_sqlite_connection() as conn:
                conn.execute("SELECT 1")
            logger.info("SQLite database connection verified")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def _get_sqlite_connection(self):
        """Get a new SQLite connection."""
        conn = sqlite3.connect(self._sqlite_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Return dict-like rows
        return conn

    @contextmanager
    def get_connection(self) -> Generator:
        """
        Context manager for database connections with automatic cleanup.

        Yields:
            sqlite3 connection object

        Example:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM contributors")
        """
        conn = None
        try:
            conn = self._get_sqlite_connection()
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    @contextmanager
    def get_cursor(self, commit: bool = True) -> Generator:
        """
        Context manager for database cursor with automatic cleanup.

        Args:
            commit: Whether to commit transaction on success (default: True)

        Yields:
            sqlite3 cursor object

        Example:
            with db_manager.get_cursor() as cursor:
                cursor.execute("SELECT * FROM contributors WHERE email = ?", (email,))
                result = cursor.fetchone()
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
                if commit:
                    conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                cursor.close()

    def execute_query(self, query: str, params: tuple = None, fetch_one: bool = False, fetch_all: bool = False):
        """
        Execute a query and return results.

        Args:
            query: SQL query string
            params: Query parameters (tuple)
            fetch_one: Fetch single result
            fetch_all: Fetch all results

        Returns:
            Query results or None
        """
        with self.get_cursor() as cursor:
            # SQLite doesn't accept None for params
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            if fetch_one:
                return cursor.fetchone()
            elif fetch_all:
                return cursor.fetchall()
            return None

    def execute_many(self, query: str, params_list: list):
        """
        Execute same query with multiple parameter sets (bulk insert).

        Args:
            query: SQL query string
            params_list: List of parameter tuples
        """
        with self.get_cursor() as cursor:
            cursor.executemany(query, params_list)

    def close_all(self):
        """Close database connections (no-op for SQLite, connections are per-request)."""
        logger.info("SQLite connections closed")

    def __del__(self):
        """Cleanup on object destruction."""
        self.close_all()
