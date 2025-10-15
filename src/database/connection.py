"""
PostgreSQL database connection management with connection pooling.
Provides production-grade database connectivity with proper resource management.
"""
import psycopg2
from psycopg2 import pool, extras
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from contextlib import contextmanager
from typing import Generator, Optional
import logging

from src.config import Settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages PostgreSQL database connections with pooling.
    Thread-safe connection pool for production use.
    """

    def __init__(self, settings: Settings):
        """
        Initialize database manager with connection pool.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self._pool: Optional[pool.ThreadedConnectionPool] = None
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize PostgreSQL connection pool."""
        try:
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=20,
                host=self.settings.postgres_host,
                port=self.settings.postgres_port,
                database=self.settings.postgres_db,
                user=self.settings.postgres_user,
                password=self.settings.postgres_password,
                cursor_factory=extras.RealDictCursor  # Return dicts instead of tuples
            )
            logger.info(f"Database connection pool initialized: {self.settings.postgres_host}:{self.settings.postgres_port}/{self.settings.postgres_db}")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise

    @contextmanager
    def get_connection(self) -> Generator:
        """
        Context manager for database connections with automatic cleanup.

        Yields:
            psycopg2 connection object

        Example:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM contributors")
        """
        if self._pool is None:
            raise RuntimeError("Database pool not initialized")

        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                self._pool.putconn(conn)

    @contextmanager
    def get_cursor(self, commit: bool = True) -> Generator:
        """
        Context manager for database cursor with automatic cleanup.

        Args:
            commit: Whether to commit transaction on success (default: True)

        Yields:
            psycopg2 cursor object

        Example:
            with db_manager.get_cursor() as cursor:
                cursor.execute("SELECT * FROM contributors WHERE email = %s", (email,))
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
            cursor.execute(query, params)

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
        """Close all database connections in the pool."""
        if self._pool:
            self._pool.closeall()
            logger.info("Database connection pool closed")

    def __del__(self):
        """Cleanup on object destruction."""
        self.close_all()
