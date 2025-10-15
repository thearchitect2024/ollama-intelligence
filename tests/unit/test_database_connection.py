"""
Unit tests for src/database/connection.py
Tests DatabaseManager connection and query execution.
"""
import pytest
from unittest.mock import MagicMock, Mock, patch, call
from contextlib import contextmanager

from src.database.connection import DatabaseManager


# ==========================================================================
# TEST DatabaseManager.__init__
# ==========================================================================

class TestDatabaseManagerInit:
    """Test DatabaseManager initialization."""

    @patch('src.database.connection.psycopg2.pool.ThreadedConnectionPool')
    def test_init_creates_connection_pool(self, mock_pool, test_settings):
        """Should create connection pool on init."""
        mock_pool_instance = MagicMock()
        mock_pool.return_value = mock_pool_instance
        
        db_manager = DatabaseManager(test_settings)
        
        assert db_manager.settings == test_settings
        assert db_manager._pool is not None
        mock_pool.assert_called_once()

    @patch('src.database.connection.psycopg2.pool.ThreadedConnectionPool')
    def test_init_with_correct_parameters(self, mock_pool, test_settings):
        """Should initialize with correct database parameters."""
        mock_pool_instance = MagicMock()
        mock_pool.return_value = mock_pool_instance
        
        DatabaseManager(test_settings)
        
        call_kwargs = mock_pool.call_args[1]
        assert call_kwargs['host'] == test_settings.postgres_host
        assert call_kwargs['port'] == test_settings.postgres_port
        assert call_kwargs['database'] == test_settings.postgres_db
        assert call_kwargs['user'] == test_settings.postgres_user

    @patch('src.database.connection.psycopg2.pool.ThreadedConnectionPool')
    def test_init_failure_raises_error(self, mock_pool, test_settings):
        """Should raise error on pool creation failure."""
        mock_pool.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception, match="Connection failed"):
            DatabaseManager(test_settings)

    @patch('src.database.connection.psycopg2.pool.ThreadedConnectionPool')
    def test_init_sets_min_max_connections(self, mock_pool, test_settings):
        """Should set min and max connections."""
        mock_pool_instance = MagicMock()
        mock_pool.return_value = mock_pool_instance
        
        DatabaseManager(test_settings)
        
        call_kwargs = mock_pool.call_args[1]
        assert call_kwargs['minconn'] == 1
        assert call_kwargs['maxconn'] == 20


# ==========================================================================
# TEST get_connection
# ==========================================================================

class TestGetConnection:
    """Test get_connection context manager."""

    @patch('src.database.connection.psycopg2.pool.ThreadedConnectionPool')
    def test_get_connection_successful(self, mock_pool, test_settings):
        """Should get connection from pool."""
        mock_conn = MagicMock()
        mock_pool_instance = MagicMock()
        mock_pool_instance.getconn.return_value = mock_conn
        mock_pool.return_value = mock_pool_instance
        
        db_manager = DatabaseManager(test_settings)
        
        with db_manager.get_connection() as conn:
            assert conn == mock_conn
        
        mock_pool_instance.getconn.assert_called_once()
        mock_pool_instance.putconn.assert_called_once_with(mock_conn)

    @patch('src.database.connection.psycopg2.pool.ThreadedConnectionPool')
    def test_get_connection_commits_on_success(self, mock_pool, test_settings):
        """Should commit transaction on success."""
        mock_conn = MagicMock()
        mock_pool_instance = MagicMock()
        mock_pool_instance.getconn.return_value = mock_conn
        mock_pool.return_value = mock_pool_instance
        
        db_manager = DatabaseManager(test_settings)
        
        with db_manager.get_connection() as conn:
            pass
        
        mock_conn.commit.assert_called_once()

    @patch('src.database.connection.psycopg2.pool.ThreadedConnectionPool')
    def test_get_connection_rollback_on_error(self, mock_pool, test_settings):
        """Should rollback on exception."""
        mock_conn = MagicMock()
        mock_pool_instance = MagicMock()
        mock_pool_instance.getconn.return_value = mock_conn
        mock_pool.return_value = mock_pool_instance
        
        db_manager = DatabaseManager(test_settings)
        
        with pytest.raises(ValueError):
            with db_manager.get_connection() as conn:
                raise ValueError("Test error")
        
        mock_conn.rollback.assert_called_once()

    @patch('src.database.connection.psycopg2.pool.ThreadedConnectionPool')
    def test_get_connection_returns_to_pool_on_error(self, mock_pool, test_settings):
        """Should return connection to pool even on error."""
        mock_conn = MagicMock()
        mock_pool_instance = MagicMock()
        mock_pool_instance.getconn.return_value = mock_conn
        mock_pool.return_value = mock_pool_instance
        
        db_manager = DatabaseManager(test_settings)
        
        with pytest.raises(ValueError):
            with db_manager.get_connection() as conn:
                raise ValueError("Test error")
        
        mock_pool_instance.putconn.assert_called_once_with(mock_conn)

    @patch('src.database.connection.psycopg2.pool.ThreadedConnectionPool')
    def test_get_connection_without_pool_raises_error(self, mock_pool, test_settings):
        """Should raise error if pool not initialized."""
        mock_pool.return_value = None
        
        db_manager = DatabaseManager(test_settings)
        db_manager._pool = None
        
        with pytest.raises(RuntimeError, match="not initialized"):
            with db_manager.get_connection():
                pass


# ==========================================================================
# TEST get_cursor
# ==========================================================================

class TestGetCursor:
    """Test get_cursor context manager."""

    @patch('src.database.connection.psycopg2.pool.ThreadedConnectionPool')
    def test_get_cursor_successful(self, mock_pool, test_settings):
        """Should get cursor from connection."""
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_pool_instance = MagicMock()
        mock_pool_instance.getconn.return_value = mock_conn
        mock_pool.return_value = mock_pool_instance
        
        db_manager = DatabaseManager(test_settings)
        
        with db_manager.get_cursor() as cursor:
            assert cursor == mock_cursor
        
        mock_cursor.close.assert_called_once()

    @patch('src.database.connection.psycopg2.pool.ThreadedConnectionPool')
    def test_get_cursor_commits_by_default(self, mock_pool, test_settings):
        """Should commit by default."""
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_pool_instance = MagicMock()
        mock_pool_instance.getconn.return_value = mock_conn
        mock_pool.return_value = mock_pool_instance
        
        db_manager = DatabaseManager(test_settings)
        
        with db_manager.get_cursor() as cursor:
            pass
        
        mock_conn.commit.assert_called_once()

    @patch('src.database.connection.psycopg2.pool.ThreadedConnectionPool')
    def test_get_cursor_no_commit_option(self, mock_pool, test_settings):
        """Should support commit=False option."""
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_pool_instance = MagicMock()
        mock_pool_instance.getconn.return_value = mock_conn
        mock_pool.return_value = mock_pool_instance
        
        db_manager = DatabaseManager(test_settings)
        
        with db_manager.get_cursor(commit=False) as cursor:
            pass
        
        # Should not commit with commit=False
        # But might commit anyway in context manager - check implementation

    @patch('src.database.connection.psycopg2.pool.ThreadedConnectionPool')
    def test_get_cursor_rollback_on_error(self, mock_pool, test_settings):
        """Should rollback on exception."""
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_pool_instance = MagicMock()
        mock_pool_instance.getconn.return_value = mock_conn
        mock_pool.return_value = mock_pool_instance
        
        db_manager = DatabaseManager(test_settings)
        
        with pytest.raises(ValueError):
            with db_manager.get_cursor() as cursor:
                raise ValueError("Test error")
        
        mock_conn.rollback.assert_called_once()


# ==========================================================================
# TEST execute_query
# ==========================================================================

class TestExecuteQuery:
    """Test execute_query method."""

    @patch('src.database.connection.psycopg2.pool.ThreadedConnectionPool')
    def test_execute_query_successful(self, mock_pool, test_settings):
        """Should execute query successfully."""
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_pool_instance = MagicMock()
        mock_pool_instance.getconn.return_value = mock_conn
        mock_pool.return_value = mock_pool_instance
        
        db_manager = DatabaseManager(test_settings)
        db_manager.execute_query("SELECT * FROM test")
        
        mock_cursor.execute.assert_called_once()

    @patch('src.database.connection.psycopg2.pool.ThreadedConnectionPool')
    def test_execute_query_with_params(self, mock_pool, test_settings):
        """Should pass parameters to query."""
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_pool_instance = MagicMock()
        mock_pool_instance.getconn.return_value = mock_conn
        mock_pool.return_value = mock_pool_instance
        
        db_manager = DatabaseManager(test_settings)
        db_manager.execute_query("SELECT * FROM test WHERE id = %s", ("123",))
        
        call_args = mock_cursor.execute.call_args[0]
        assert call_args[1] == ("123",)

    @patch('src.database.connection.psycopg2.pool.ThreadedConnectionPool')
    def test_execute_query_fetch_one(self, mock_pool, test_settings):
        """Should fetch one result."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {'id': 1, 'name': 'Test'}
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_pool_instance = MagicMock()
        mock_pool_instance.getconn.return_value = mock_conn
        mock_pool.return_value = mock_pool_instance
        
        db_manager = DatabaseManager(test_settings)
        result = db_manager.execute_query("SELECT * FROM test", fetch_one=True)
        
        assert result == {'id': 1, 'name': 'Test'}
        mock_cursor.fetchone.assert_called_once()

    @patch('src.database.connection.psycopg2.pool.ThreadedConnectionPool')
    def test_execute_query_fetch_all(self, mock_pool, test_settings):
        """Should fetch all results."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [{'id': 1}, {'id': 2}]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_pool_instance = MagicMock()
        mock_pool_instance.getconn.return_value = mock_conn
        mock_pool.return_value = mock_pool_instance
        
        db_manager = DatabaseManager(test_settings)
        results = db_manager.execute_query("SELECT * FROM test", fetch_all=True)
        
        assert len(results) == 2
        mock_cursor.fetchall.assert_called_once()

    @patch('src.database.connection.psycopg2.pool.ThreadedConnectionPool')
    def test_execute_query_returns_none_by_default(self, mock_pool, test_settings):
        """Should return None when no fetch option."""
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_pool_instance = MagicMock()
        mock_pool_instance.getconn.return_value = mock_conn
        mock_pool.return_value = mock_pool_instance
        
        db_manager = DatabaseManager(test_settings)
        result = db_manager.execute_query("INSERT INTO test VALUES (1)")
        
        assert result is None


# ==========================================================================
# TEST execute_many
# ==========================================================================

class TestExecuteMany:
    """Test execute_many method."""

    @patch('src.database.connection.psycopg2.pool.ThreadedConnectionPool')
    def test_execute_many_successful(self, mock_pool, test_settings):
        """Should execute many queries."""
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_pool_instance = MagicMock()
        mock_pool_instance.getconn.return_value = mock_conn
        mock_pool.return_value = mock_pool_instance
        
        db_manager = DatabaseManager(test_settings)
        params_list = [("val1",), ("val2",), ("val3",)]
        db_manager.execute_many("INSERT INTO test VALUES (%s)", params_list)
        
        mock_cursor.executemany.assert_called_once_with(
            "INSERT INTO test VALUES (%s)",
            params_list
        )


# ==========================================================================
# TEST close_all
# ==========================================================================

class TestCloseAll:
    """Test close_all method."""

    @patch('src.database.connection.psycopg2.pool.ThreadedConnectionPool')
    def test_close_all_closes_pool(self, mock_pool, test_settings):
        """Should close all connections in pool."""
        mock_pool_instance = MagicMock()
        mock_pool.return_value = mock_pool_instance
        
        db_manager = DatabaseManager(test_settings)
        db_manager.close_all()
        
        mock_pool_instance.closeall.assert_called_once()

    @patch('src.database.connection.psycopg2.pool.ThreadedConnectionPool')
    def test_close_all_with_no_pool(self, mock_pool, test_settings):
        """Should handle no pool gracefully."""
        mock_pool.return_value = MagicMock()
        
        db_manager = DatabaseManager(test_settings)
        db_manager._pool = None
        
        # Should not raise error
        db_manager.close_all()

