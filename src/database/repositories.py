"""
Data access layer for contributor profiles.
Provides repository pattern for database operations.
"""
import json
import logging
from typing import List, Optional
from datetime import datetime

from src.database.connection import DatabaseManager
from src.models import ContributorProfile, ProcessingStatus

logger = logging.getLogger(__name__)


class ContributorRepository:
    """Repository for contributor data operations."""

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize repository with database manager.

        Args:
            db_manager: Database connection manager
        """
        self.db = db_manager

    def _json_extract(self, field: str, path: str) -> str:
        """
        Generate JSON extraction SQL for both PostgreSQL and SQLite.
        
        Args:
            field: Column name (e.g., 'processed_data')
            path: JSON path (e.g., 'activity_summary.is_active_90d')
        
        Returns:
            str: Database-specific JSON extraction syntax
        """
        if self.db.is_postgres:
            # PostgreSQL: processed_data->'activity_summary'->>'is_active_90d'
            parts = path.split('.')
            result = field
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    result += f"->>{part!r}"  # Last part uses ->> to get text
                else:
                    result += f"->{part!r}"   # Intermediate parts use ->
            return result
        elif self.db.is_sqlite:
            # SQLite: json_extract(processed_data, '$.activity_summary.is_active_90d')
            json_path = '$.' + path.replace('.', '.')
            return f"json_extract({field}, '{json_path}')"
        else:
            raise ValueError("Unsupported database type")

    def _json_cast_float(self, json_expr: str) -> str:
        """
        Generate SQL to cast JSON value to float.
        
        Args:
            json_expr: JSON extraction expression
        
        Returns:
            str: Database-specific cast syntax
        """
        if self.db.is_postgres:
            return f"({json_expr})::float"
        elif self.db.is_sqlite:
            return f"CAST({json_expr} AS REAL)"
        else:
            raise ValueError("Unsupported database type")

    @property
    def _param_placeholder(self) -> str:
        """Get parameter placeholder for SQL queries."""
        if self.db.is_postgres:
            return "%s"
        elif self.db.is_sqlite:
            return "?"
        else:
            raise ValueError("Unsupported database type")

    def _case_insensitive_like(self) -> str:
        """Get case-insensitive LIKE operator."""
        if self.db.is_postgres:
            return "ILIKE"
        elif self.db.is_sqlite:
            return "LIKE"  # SQLite's LIKE is case-insensitive by default
        else:
            raise ValueError("Unsupported database type")

    def upsert_contributor(self, profile: ContributorProfile) -> bool:
        """
        Insert or update contributor profile by email.

        Args:
            profile: ContributorProfile to save

        Returns:
            bool: True if successful
        """
        try:
            processed_data = profile.model_dump_json_safe()
            ph = self._param_placeholder  # %s or ?
            
            if self.db.is_postgres:
                query = f"""
                    INSERT INTO contributors (
                        email, contributor_id, processed_data,
                        intelligence_summary, processing_status,
                        created_at, updated_at
                    )
                    VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
                    ON CONFLICT (email)
                    DO UPDATE SET
                        processed_data = EXCLUDED.processed_data,
                        intelligence_summary = EXCLUDED.intelligence_summary,
                        processing_status = EXCLUDED.processing_status,
                        updated_at = EXCLUDED.updated_at
                """
            else:  # SQLite
                query = f"""
                    INSERT OR REPLACE INTO contributors (
                        email, contributor_id, processed_data,
                        intelligence_summary, processing_status,
                        created_at, updated_at
                    )
                    VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
                """

            params = (
                profile.contributor_email,
                profile.contributor_id,
                json.dumps(processed_data),
                profile.intelligence_summary,
                profile.processing_status if isinstance(profile.processing_status, str) else profile.processing_status.value,
                datetime.now().isoformat(),
                datetime.now().isoformat()
            )

            self.db.execute_query(query, params)
            logger.info(f"Upserted contributor: {profile.contributor_email}")
            return True

        except Exception as e:
            logger.error(f"Failed to upsert contributor {profile.contributor_email}: {e}")
            return False

    def get_by_email(self, email: str) -> Optional[dict]:
        """
        Fetch contributor by email.

        Args:
            email: Contributor email

        Returns:
            dict: Contributor record or None
        """
        ph = self._param_placeholder
        
        query = f"""
            SELECT email, contributor_id, processed_data,
                   intelligence_summary, processing_status,
                   created_at, updated_at, intelligence_extracted_at
            FROM contributors
            WHERE email = {ph}
        """

        result = self.db.execute_query(query, (email,), fetch_one=True)
        return dict(result) if result else None

    def get_all_emails(self) -> List[str]:
        """
        Get list of all contributor emails.

        Returns:
            List[str]: List of emails
        """
        query = "SELECT email FROM contributors ORDER BY email"
        results = self.db.execute_query(query, fetch_all=True)
        return [row['email'] for row in results] if results else []

    def get_all_contributors(self) -> List[dict]:
        """
        Get all contributors.

        Returns:
            List[dict]: List of contributor records
        """
        query = """
            SELECT email, contributor_id, processed_data,
                   intelligence_summary, processing_status,
                   created_at, updated_at
            FROM contributors
            ORDER BY updated_at DESC
        """
        results = self.db.execute_query(query, fetch_all=True)
        return [dict(row) for row in results] if results else []

    def get_active_90d(self) -> List[dict]:
        """
        Get contributors active in past 90 days.

        Returns:
            List[dict]: List of active contributors
        """
        is_active_expr = self._json_extract('processed_data', 'activity_summary.is_active_90d')
        hours_expr = self._json_extract('processed_data', 'activity_summary.weekly_hours_avg')
        hours_cast = self._json_cast_float(hours_expr)
        
        query = f"""
            SELECT email, contributor_id, processed_data, intelligence_summary
            FROM contributors
            WHERE {is_active_expr} = 'true'
            ORDER BY {hours_cast} DESC
        """
        results = self.db.execute_query(query, fetch_all=True)
        return [dict(row) for row in results] if results else []

    def get_inactive_90d(self) -> List[dict]:
        """
        Get contributors inactive in past 90 days.

        Returns:
            List[dict]: List of inactive contributors
        """
        is_active_expr = self._json_extract('processed_data', 'activity_summary.is_active_90d')
        
        query = f"""
            SELECT email, contributor_id, processed_data
            FROM contributors
            WHERE {is_active_expr} = 'false'
        """
        results = self.db.execute_query(query, fetch_all=True)
        return [dict(row) for row in results] if results else []

    def update_intelligence(self, email: str, summary: str) -> bool:
        """
        Update intelligence summary for contributor.

        Args:
            email: Contributor email
            summary: Intelligence summary text

        Returns:
            bool: True if successful
        """
        try:
            ph = self._param_placeholder
            
            query = f"""
                UPDATE contributors
                SET intelligence_summary = {ph},
                    intelligence_extracted_at = {ph},
                    updated_at = {ph}
                WHERE email = {ph}
            """

            params = (summary, datetime.now().isoformat(), datetime.now().isoformat(), email)
            self.db.execute_query(query, params)
            logger.info(f"Updated intelligence for: {email}")
            return True

        except Exception as e:
            logger.error(f"Failed to update intelligence for {email}: {e}")
            return False

    def mark_as_failed(self, email: str, error_message: str) -> bool:
        """
        Mark contributor as failed processing.

        Args:
            email: Contributor email
            error_message: Error description

        Returns:
            bool: True if successful
        """
        try:
            ph = self._param_placeholder
            
            query = f"""
                UPDATE contributors
                SET processing_status = {ph},
                    error_message = {ph},
                    updated_at = {ph}
                WHERE email = {ph}
            """

            params = (ProcessingStatus.FAILED.value, error_message, datetime.now().isoformat(), email)
            self.db.execute_query(query, params)
            return True

        except Exception as e:
            logger.error(f"Failed to mark {email} as failed: {e}")
            return False

    def get_statistics(self) -> dict:
        """
        Get database statistics.

        Returns:
            dict: Statistics summary
        """
        is_active_expr = self._json_extract('processed_data', 'activity_summary.is_active_90d')
        
        query = f"""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN processing_status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN processing_status = 'failed' THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN {is_active_expr} = 'true' THEN 1 ELSE 0 END) as active_90d,
                SUM(CASE WHEN intelligence_summary IS NOT NULL THEN 1 ELSE 0 END) as with_intelligence
            FROM contributors
        """

        result = self.db.execute_query(query, fetch_one=True)
        return dict(result) if result else {}

    def search_by_email(self, search_term: str) -> List[dict]:
        """
        Search contributors by email pattern.

        Args:
            search_term: Email search pattern

        Returns:
            List[dict]: Matching contributors
        """
        like_op = self._case_insensitive_like()
        placeholder = self._param_placeholder
        
        query = f"""
            SELECT email, contributor_id, processed_data, intelligence_summary
            FROM contributors
            WHERE email {like_op} {placeholder}
            ORDER BY email
            LIMIT 50
        """

        results = self.db.execute_query(query, (f"%{search_term}%",), fetch_all=True)
        return [dict(row) for row in results] if results else []
