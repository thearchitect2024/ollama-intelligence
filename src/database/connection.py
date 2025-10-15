"""
Database connection management with support for PostgreSQL and SQLite.
Provides production-grade database connectivity with proper resource management.
"""
import sqlite3
from contextlib import contextmanager
from typing import Generator, Optional, Any
import logging

from src.config import Settings

logger = logging.getLogger(__name__)

# Try to import PostgreSQL driver, but don't fail if it's not available
try:
    import psycopg2
    from psycopg2 import pool, extras
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    logger.warning("psycopg2 not available - PostgreSQL connections disabled")


class DatabaseManager:
    """
    Manages database connections with pooling.
    Supports both PostgreSQL (with connection pooling) and SQLite (file-based).
    Thread-safe for production use.
    """

    def __init__(self, settings: Settings):
        """
        Initialize database manager.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.db_url = settings.get_database_url()
        self.is_sqlite = self.db_url.startswith('sqlite:///')
        self.is_postgres = self.db_url.startswith('postgresql://')
        
        self._pool: Optional[Any] = None
        self._sqlite_path: Optional[str] = None
        
        self._initialize_connection()

    def _initialize_connection(self):
        """Initialize database connection (PostgreSQL pool or SQLite path)."""
        try:
            if self.is_sqlite:
                # Extract SQLite file path from URL
                self._sqlite_path = self.db_url.replace('sqlite:///', '')
                logger.info(f"SQLite database configured: {self._sqlite_path}")
                # Test connection
                with self._get_sqlite_connection() as conn:
                    conn.execute("SELECT 1")
                logger.info("SQLite database connection verified")
                
            elif self.is_postgres:
                if not POSTGRES_AVAILABLE:
                    raise RuntimeError("PostgreSQL driver (psycopg2) not installed")
                
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
                logger.info(f"PostgreSQL connection pool initialized: {self.settings.postgres_host}:{self.settings.postgres_port}/{self.settings.postgres_db}")
            else:
                raise ValueError(f"Unsupported database URL: {self.db_url}")
                
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
            Database connection object (psycopg2 or sqlite3)

        Example:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM contributors")
        """
        if self.is_sqlite:
            # SQLite connection
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
        
        elif self.is_postgres:
            # PostgreSQL pooled connection
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
            # SQLite doesn't accept None for params, PostgreSQL does
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
        """Close all database connections."""
        if self.is_postgres and self._pool:
            self._pool.closeall()
            logger.info("PostgreSQL connection pool closed")
        elif self.is_sqlite:
            logger.info("SQLite connections closed (no pool to close)")

    def __del__(self):
        """Cleanup on object destruction."""
        self.close_all()
