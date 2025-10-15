"""
Integration tests for src/database/repositories.py
Tests ContributorRepository with mocked database.
"""
import pytest
from unittest.mock import MagicMock, Mock
from datetime import datetime

from src.database.repositories import ContributorRepository
from src.models import ProcessingStatus


# ==========================================================================
# TEST ContributorRepository.__init__
# ==========================================================================

class TestContributorRepositoryInit:
    """Test ContributorRepository initialization."""

    def test_init_with_db_manager(self, mock_db_manager):
        """Should initialize with database manager."""
        repo = ContributorRepository(mock_db_manager)
        assert repo.db == mock_db_manager


# ==========================================================================
# TEST upsert_contributor
# ==========================================================================

class TestUpsertContributor:
    """Test upsert_contributor method."""

    def test_upsert_new_contributor(self, mock_db_manager, sample_contributor_profile):
        """Should insert new contributor."""
        mock_db_manager.execute_query.return_value = None
        repo = ContributorRepository(mock_db_manager)
        
        result = repo.upsert_contributor(sample_contributor_profile)
        
        assert result is True
        mock_db_manager.execute_query.assert_called_once()
        
        # Verify query parameters
        call_args = mock_db_manager.execute_query.call_args
        assert sample_contributor_profile.contributor_email in call_args[0][1]

    def test_upsert_existing_contributor(self, mock_db_manager, sample_contributor_profile):
        """Should update existing contributor."""
        mock_db_manager.execute_query.return_value = None
        repo = ContributorRepository(mock_db_manager)
        
        result = repo.upsert_contributor(sample_contributor_profile)
        
        assert result is True

    def test_upsert_with_intelligence_summary(self, mock_db_manager, sample_contributor_profile):
        """Should store intelligence summary."""
        sample_contributor_profile.intelligence_summary = "Test summary"
        mock_db_manager.execute_query.return_value = None
        repo = ContributorRepository(mock_db_manager)
        
        result = repo.upsert_contributor(sample_contributor_profile)
        
        assert result is True
        call_args = mock_db_manager.execute_query.call_args[0][1]
        assert "Test summary" in call_args

    def test_upsert_handles_error(self, mock_db_manager, sample_contributor_profile):
        """Should handle database errors gracefully."""
        mock_db_manager.execute_query.side_effect = Exception("DB error")
        repo = ContributorRepository(mock_db_manager)
        
        result = repo.upsert_contributor(sample_contributor_profile)
        
        assert result is False

    def test_upsert_serializes_profile_data(self, mock_db_manager, sample_contributor_profile):
        """Should serialize profile to JSON."""
        mock_db_manager.execute_query.return_value = None
        repo = ContributorRepository(mock_db_manager)
        
        repo.upsert_contributor(sample_contributor_profile)
        
        call_args = mock_db_manager.execute_query.call_args[0][1]
        # Should have JSON string in params
        import json
        json_data = call_args[2]
        parsed = json.loads(json_data)
        assert "contributor_email" in parsed


# ==========================================================================
# TEST get_by_email
# ==========================================================================

class TestGetByEmail:
    """Test get_by_email method."""

    def test_get_existing_contributor(self, mock_db_manager):
        """Should fetch contributor by email."""
        mock_db_manager.execute_query.return_value = {
            'email': 'test@example.com',
            'contributor_id': 'C123',
            'processed_data': {},
            'intelligence_summary': 'Test summary'
        }
        repo = ContributorRepository(mock_db_manager)
        
        result = repo.get_by_email('test@example.com')
        
        assert result is not None
        assert result['email'] == 'test@example.com'
        mock_db_manager.execute_query.assert_called_once()

    def test_get_non_existing_contributor(self, mock_db_manager):
        """Should return None for non-existing contributor."""
        mock_db_manager.execute_query.return_value = None
        repo = ContributorRepository(mock_db_manager)
        
        result = repo.get_by_email('nonexistent@example.com')
        
        assert result is None

    def test_get_by_email_query_parameters(self, mock_db_manager):
        """Should use correct query parameters."""
        mock_db_manager.execute_query.return_value = None
        repo = ContributorRepository(mock_db_manager)
        
        repo.get_by_email('test@test.com')
        
        call_args = mock_db_manager.execute_query.call_args
        assert 'test@test.com' in call_args[0][1]
        assert call_args[1]['fetch_one'] is True


# ==========================================================================
# TEST get_all_emails
# ==========================================================================

class TestGetAllEmails:
    """Test get_all_emails method."""

    def test_get_all_emails_with_data(self, mock_db_manager):
        """Should return list of all emails."""
        mock_db_manager.execute_query.return_value = [
            {'email': 'user1@test.com'},
            {'email': 'user2@test.com'},
            {'email': 'user3@test.com'}
        ]
        repo = ContributorRepository(mock_db_manager)
        
        emails = repo.get_all_emails()
        
        assert len(emails) == 3
        assert 'user1@test.com' in emails

    def test_get_all_emails_empty_database(self, mock_db_manager):
        """Should return empty list for empty database."""
        mock_db_manager.execute_query.return_value = []
        repo = ContributorRepository(mock_db_manager)
        
        emails = repo.get_all_emails()
        
        assert emails == []

    def test_get_all_emails_none_result(self, mock_db_manager):
        """Should handle None result."""
        mock_db_manager.execute_query.return_value = None
        repo = ContributorRepository(mock_db_manager)
        
        emails = repo.get_all_emails()
        
        assert emails == []


# ==========================================================================
# TEST get_all_contributors
# ==========================================================================

class TestGetAllContributors:
    """Test get_all_contributors method."""

    def test_get_all_contributors_with_data(self, mock_db_manager):
        """Should return list of all contributors."""
        mock_db_manager.execute_query.return_value = [
            {'email': 'user1@test.com', 'contributor_id': 'C1', 'processed_data': {}},
            {'email': 'user2@test.com', 'contributor_id': 'C2', 'processed_data': {}}
        ]
        repo = ContributorRepository(mock_db_manager)
        
        contributors = repo.get_all_contributors()
        
        assert len(contributors) == 2
        assert contributors[0]['email'] == 'user1@test.com'

    def test_get_all_contributors_empty(self, mock_db_manager):
        """Should handle empty result."""
        mock_db_manager.execute_query.return_value = []
        repo = ContributorRepository(mock_db_manager)
        
        contributors = repo.get_all_contributors()
        
        assert contributors == []


# ==========================================================================
# TEST get_active_90d
# ==========================================================================

class TestGetActive90d:
    """Test get_active_90d method."""

    def test_get_active_contributors(self, mock_db_manager):
        """Should return active contributors."""
        mock_db_manager.execute_query.return_value = [
            {'email': 'active@test.com', 'processed_data': {'activity_summary': {'is_active_90d': True}}}
        ]
        repo = ContributorRepository(mock_db_manager)
        
        actives = repo.get_active_90d()
        
        assert len(actives) == 1

    def test_get_active_empty(self, mock_db_manager):
        """Should handle no active contributors."""
        mock_db_manager.execute_query.return_value = []
        repo = ContributorRepository(mock_db_manager)
        
        actives = repo.get_active_90d()
        
        assert actives == []


# ==========================================================================
# TEST get_inactive_90d
# ==========================================================================

class TestGetInactive90d:
    """Test get_inactive_90d method."""

    def test_get_inactive_contributors(self, mock_db_manager):
        """Should return inactive contributors."""
        mock_db_manager.execute_query.return_value = [
            {'email': 'inactive@test.com', 'processed_data': {'activity_summary': {'is_active_90d': False}}}
        ]
        repo = ContributorRepository(mock_db_manager)
        
        inactives = repo.get_inactive_90d()
        
        assert len(inactives) == 1


# ==========================================================================
# TEST update_intelligence
# ==========================================================================

class TestUpdateIntelligence:
    """Test update_intelligence method."""

    def test_update_intelligence_successful(self, mock_db_manager):
        """Should update intelligence summary."""
        mock_db_manager.execute_query.return_value = None
        repo = ContributorRepository(mock_db_manager)
        
        result = repo.update_intelligence('test@test.com', 'New summary')
        
        assert result is True
        mock_db_manager.execute_query.assert_called_once()

    def test_update_intelligence_with_timestamp(self, mock_db_manager):
        """Should set intelligence_extracted_at timestamp."""
        mock_db_manager.execute_query.return_value = None
        repo = ContributorRepository(mock_db_manager)
        
        repo.update_intelligence('test@test.com', 'Summary')
        
        # Should include timestamp in query
        call_args = mock_db_manager.execute_query.call_args[0][1]
        assert len(call_args) == 4  # summary, extracted_at, updated_at, email

    def test_update_intelligence_error_handling(self, mock_db_manager):
        """Should handle errors gracefully."""
        mock_db_manager.execute_query.side_effect = Exception("DB error")
        repo = ContributorRepository(mock_db_manager)
        
        result = repo.update_intelligence('test@test.com', 'Summary')
        
        assert result is False


# ==========================================================================
# TEST mark_as_failed
# ==========================================================================

class TestMarkAsFailed:
    """Test mark_as_failed method."""

    def test_mark_as_failed_successful(self, mock_db_manager):
        """Should mark contributor as failed."""
        mock_db_manager.execute_query.return_value = None
        repo = ContributorRepository(mock_db_manager)
        
        result = repo.mark_as_failed('test@test.com', 'Error message')
        
        assert result is True

    def test_mark_as_failed_sets_status(self, mock_db_manager):
        """Should set status to FAILED."""
        mock_db_manager.execute_query.return_value = None
        repo = ContributorRepository(mock_db_manager)
        
        repo.mark_as_failed('test@test.com', 'Error')
        
        call_args = mock_db_manager.execute_query.call_args[0][1]
        assert ProcessingStatus.FAILED.value in call_args

    def test_mark_as_failed_error_handling(self, mock_db_manager):
        """Should handle errors."""
        mock_db_manager.execute_query.side_effect = Exception("DB error")
        repo = ContributorRepository(mock_db_manager)
        
        result = repo.mark_as_failed('test@test.com', 'Error')
        
        assert result is False


# ==========================================================================
# TEST get_statistics
# ==========================================================================

class TestGetStatistics:
    """Test get_statistics method."""

    def test_get_statistics_with_data(self, mock_db_manager):
        """Should return statistics."""
        mock_db_manager.execute_query.return_value = {
            'total': 100,
            'completed': 95,
            'failed': 5,
            'active_90d': 50,
            'with_intelligence': 80
        }
        repo = ContributorRepository(mock_db_manager)
        
        stats = repo.get_statistics()
        
        assert stats['total'] == 100
        assert stats['completed'] == 95
        assert stats['with_intelligence'] == 80

    def test_get_statistics_empty_database(self, mock_db_manager):
        """Should handle empty database."""
        mock_db_manager.execute_query.return_value = None
        repo = ContributorRepository(mock_db_manager)
        
        stats = repo.get_statistics()
        
        assert stats == {}


# ==========================================================================
# TEST search_by_email
# ==========================================================================

class TestSearchByEmail:
    """Test search_by_email method."""

    def test_search_finds_matches(self, mock_db_manager):
        """Should find matching contributors."""
        mock_db_manager.execute_query.return_value = [
            {'email': 'john@test.com', 'contributor_id': 'C1'},
            {'email': 'johnny@test.com', 'contributor_id': 'C2'}
        ]
        repo = ContributorRepository(mock_db_manager)
        
        results = repo.search_by_email('john')
        
        assert len(results) == 2

    def test_search_no_matches(self, mock_db_manager):
        """Should return empty list for no matches."""
        mock_db_manager.execute_query.return_value = []
        repo = ContributorRepository(mock_db_manager)
        
        results = repo.search_by_email('nonexistent')
        
        assert results == []

    def test_search_uses_like_pattern(self, mock_db_manager):
        """Should use LIKE pattern in query."""
        mock_db_manager.execute_query.return_value = []
        repo = ContributorRepository(mock_db_manager)
        
        repo.search_by_email('test')
        
        call_args = mock_db_manager.execute_query.call_args[0][1]
        assert '%test%' in call_args

    def test_search_limits_results(self, mock_db_manager):
        """Should limit results to 50."""
        mock_db_manager.execute_query.return_value = []
        repo = ContributorRepository(mock_db_manager)
        
        repo.search_by_email('test')
        
        # Query should contain LIMIT 50
        query = mock_db_manager.execute_query.call_args[0][0]
        assert 'LIMIT 50' in query

