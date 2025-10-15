"""
Data access layer for contributor profiles.
Provides repository pattern for SQLite database operations.
"""
import json
import logging
from typing import List, Optional
from datetime import datetime

from src.database.connection import DatabaseManager
from src.models import ContributorProfile, ProcessingStatus

logger = logging.getLogger(__name__)


class ContributorRepository:
    """Repository for contributor data operations with SQLite."""

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize repository with database manager.

        Args:
            db_manager: Database connection manager
        """
        self.db = db_manager

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
            
            query = """
                INSERT OR REPLACE INTO contributors (
                    email, contributor_id, processed_data,
                    intelligence_summary, processing_status,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
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
        query = """
            SELECT email, contributor_id, processed_data,
                   intelligence_summary, processing_status,
                   created_at, updated_at, intelligence_extracted_at
            FROM contributors
            WHERE email = ?
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
        Get all contributors with basic info.

        Returns:
            List[dict]: List of all contributors
        """
        query = """
            SELECT email, contributor_id, processed_data, intelligence_summary
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
        query = """
            SELECT email, contributor_id, processed_data, intelligence_summary
            FROM contributors
            WHERE json_extract(processed_data, '$.activity_summary.is_active_90d') = 'true'
            ORDER BY CAST(json_extract(processed_data, '$.activity_summary.weekly_hours_avg') AS REAL) DESC
        """
        results = self.db.execute_query(query, fetch_all=True)
        return [dict(row) for row in results] if results else []

    def get_inactive_90d(self) -> List[dict]:
        """
        Get contributors inactive in past 90 days.

        Returns:
            List[dict]: List of inactive contributors
        """
        query = """
            SELECT email, contributor_id, processed_data
            FROM contributors
            WHERE json_extract(processed_data, '$.activity_summary.is_active_90d') = 'false'
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
            query = """
                UPDATE contributors
                SET intelligence_summary = ?,
                    intelligence_extracted_at = ?,
                    updated_at = ?
                WHERE email = ?
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
            query = """
                UPDATE contributors
                SET processing_status = ?,
                    error_message = ?,
                    updated_at = ?
                WHERE email = ?
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
        query = """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN processing_status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN processing_status = 'failed' THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN json_extract(processed_data, '$.activity_summary.is_active_90d') = 'true' THEN 1 ELSE 0 END) as active_90d,
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
        query = """
            SELECT email, contributor_id, processed_data, intelligence_summary
            FROM contributors
            WHERE email LIKE ?
            ORDER BY email
            LIMIT 50
        """

        results = self.db.execute_query(query, (f"%{search_term}%",), fetch_all=True)
        return [dict(row) for row in results] if results else []

    def get_contributors_without_intelligence(self) -> List[dict]:
        """
        Get contributors that need intelligence extraction.

        Returns:
            List[dict]: List of contributors without intelligence
        """
        query = """
            SELECT email, contributor_id, processed_data
            FROM contributors
            WHERE intelligence_summary IS NULL
            OR intelligence_summary = ''
            ORDER BY created_at ASC
        """

        results = self.db.execute_query(query, fetch_all=True)
        return [dict(row) for row in results] if results else []
