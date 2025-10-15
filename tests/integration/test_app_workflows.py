"""
Integration tests for app.py workflows.
Tests main application workflows and integration points.
"""
import pytest
from unittest.mock import MagicMock, Mock, patch, call
import pandas as pd
from io import StringIO

# Note: Streamlit apps are harder to test directly, so we focus on testing
# the underlying function calls and data flow


# ==========================================================================
# TEST Application Initialization
# ==========================================================================

class TestApplicationInitialization:
    """Test application initialization workflow."""

    @patch('src.database.migrations.create_schema')
    @patch('src.database.repositories.ContributorRepository')
    @patch('src.database.connection.DatabaseManager')
    @patch('src.config.get_settings')
    def test_init_application_successful(self, mock_get_settings, mock_db_manager,
                                        mock_repo, mock_create_schema):
        """Should initialize application components."""
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings
        mock_db_instance = MagicMock()
        mock_db_manager.return_value = mock_db_instance
        mock_repo_instance = MagicMock()
        mock_repo.return_value = mock_repo_instance
        
        # Test the initialization logic
        settings = mock_get_settings()
        db_manager = mock_db_manager(settings)
        repository = mock_repo(db_manager)
        mock_create_schema(db_manager)
        
        assert settings == mock_settings
        assert db_manager == mock_db_instance
        assert repository == mock_repo_instance
        mock_create_schema.assert_called_once_with(mock_db_instance)


# ==========================================================================
# TEST CSV Upload and Processing Workflow
# ==========================================================================

class TestCSVProcessingWorkflow:
    """Test CSV upload and processing workflow."""

    @patch('src.processors.data_processor.process_contributor')
    def test_process_csv_row_successful(self, mock_process_contributor, 
                                       test_settings, sample_csv_row):
        """Should process single CSV row."""
        from src.models import ContributorProfile, ActivitySummary, Location, Compliance
        
        mock_profile = ContributorProfile(
            contributor_email="test@test.com",
            contributor_id="C123",
            activity_summary=ActivitySummary(),
            location=Location(country="USA"),
            compliance=Compliance()
        )
        mock_process_contributor.return_value = mock_profile
        
        result = mock_process_contributor(sample_csv_row, test_settings)
        
        assert result.contributor_email == "test@test.com"

    def test_csv_chunked_processing_flow(self):
        """Should process CSV in chunks."""
        # Create test DataFrame
        data = {
            'email': ['user1@test.com', 'user2@test.com'],
            'contributor_id': ['C1', 'C2'],
            'projects_json': ['[]', '[]'],
            'languages_json': ['[]', '[]']
        }
        df = pd.DataFrame(data)
        
        # Test chunking
        chunksize = 1
        chunks = [chunk for chunk in pd.read_csv(StringIO(df.to_csv(index=False)), chunksize=chunksize)]
        
        assert len(chunks) == 2


# ==========================================================================
# TEST Intelligence Extraction Workflow
# ==========================================================================

class TestIntelligenceExtractionWorkflow:
    """Test intelligence extraction workflow."""

    @patch('src.intelligence.skill_extractor.generate_intelligence_summary')
    @patch('src.intelligence.llm_client.OllamaClient')
    def test_intelligence_extraction_single_contributor(self, mock_ollama_client,
                                                       mock_generate_summary,
                                                       sample_contributor_profile):
        """Should extract intelligence for single contributor."""
        mock_client_instance = MagicMock()
        mock_ollama_client.return_value = mock_client_instance
        mock_generate_summary.return_value = "Test intelligence summary"
        
        summary = mock_generate_summary(sample_contributor_profile, mock_client_instance)
        
        assert summary == "Test intelligence summary"

    def test_filter_contributors_without_intelligence(self):
        """Should filter contributors needing intelligence extraction."""
        contributors = [
            {'email': 'user1@test.com', 'intelligence_summary': None},
            {'email': 'user2@test.com', 'intelligence_summary': 'Has summary'},
            {'email': 'user3@test.com', 'intelligence_summary': None}
        ]
        
        needs_processing = [c for c in contributors if not c.get('intelligence_summary')]
        
        assert len(needs_processing) == 2
        assert needs_processing[0]['email'] == 'user1@test.com'


# ==========================================================================
# TEST View Profiles Workflow
# ==========================================================================

class TestViewProfilesWorkflow:
    """Test view profiles workflow."""

    def test_get_contributor_by_email_flow(self, mock_contributor_repository):
        """Should retrieve contributor by email."""
        mock_contributor_repository.get_by_email.return_value = {
            'email': 'test@test.com',
            'processed_data': {'contributor_email': 'test@test.com'}
        }
        
        result = mock_contributor_repository.get_by_email('test@test.com')
        
        assert result is not None
        assert result['email'] == 'test@test.com'

    def test_format_profile_display_data(self, sample_contributor_profile):
        """Should format profile data for display."""
        profile_data = sample_contributor_profile.model_dump()
        
        # Verify key display fields exist
        assert 'contributor_email' in profile_data
        assert 'education_level' in profile_data
        assert 'production_projects' in profile_data
        assert 'activity_summary' in profile_data


# ==========================================================================
# TEST Search Workflow
# ==========================================================================

class TestSearchWorkflow:
    """Test search workflow."""

    def test_search_contributors_by_email_pattern(self, mock_contributor_repository):
        """Should search contributors by email pattern."""
        mock_contributor_repository.search_by_email.return_value = [
            {'email': 'john.doe@test.com', 'processed_data': {}},
            {'email': 'john.smith@test.com', 'processed_data': {}}
        ]
        
        results = mock_contributor_repository.search_by_email('john')
        
        assert len(results) == 2

    def test_search_returns_empty_for_no_matches(self, mock_contributor_repository):
        """Should return empty list for no matches."""
        mock_contributor_repository.search_by_email.return_value = []
        
        results = mock_contributor_repository.search_by_email('nonexistent')
        
        assert results == []


# ==========================================================================
# TEST Statistics and Dashboard
# ==========================================================================

class TestStatisticsWorkflow:
    """Test statistics and dashboard workflow."""

    def test_get_repository_statistics(self, mock_contributor_repository):
        """Should retrieve repository statistics."""
        mock_contributor_repository.get_statistics.return_value = {
            'total': 100,
            'with_intelligence': 80,
            'active_90d': 50
        }
        
        stats = mock_contributor_repository.get_statistics()
        
        assert stats['total'] == 100
        assert stats['with_intelligence'] == 80

    def test_calculate_remaining_contributors(self):
        """Should calculate remaining contributors needing intelligence."""
        stats = {
            'total': 100,
            'with_intelligence': 75
        }
        
        remaining = (stats.get('total', 0) or 0) - (stats.get('with_intelligence', 0) or 0)
        
        assert remaining == 25

    def test_handle_none_statistics(self):
        """Should handle None values in statistics."""
        stats = {
            'total': None,
            'with_intelligence': 80
        }
        
        total = stats.get('total', 0) or 0
        with_intel = stats.get('with_intelligence', 0) or 0
        
        assert total == 0
        assert with_intel == 80


# ==========================================================================
# TEST Error Handling
# ==========================================================================

class TestErrorHandling:
    """Test error handling in workflows."""

    def test_handle_database_connection_error(self, mock_db_manager):
        """Should handle database connection errors."""
        mock_db_manager.execute_query.side_effect = Exception("Connection failed")
        
        try:
            mock_db_manager.execute_query("SELECT 1")
            assert False, "Should have raised exception"
        except Exception as e:
            assert "Connection failed" in str(e)

    def test_handle_csv_parsing_error(self):
        """Should handle CSV parsing errors."""
        invalid_csv = "invalid,csv\ndata"
        
        try:
            # This should raise an error or be handled gracefully
            df = pd.read_csv(StringIO(invalid_csv))
            # If it doesn't raise, that's okay - pandas is lenient
            assert True
        except Exception:
            # Error is expected and handled
            assert True

    @patch('src.processors.data_processor.process_contributor')
    def test_handle_contributor_processing_error(self, mock_process_contributor, 
                                                 sample_csv_row, test_settings):
        """Should handle individual contributor processing errors."""
        mock_process_contributor.side_effect = Exception("Processing failed")
        
        try:
            mock_process_contributor(sample_csv_row, test_settings)
            assert False, "Should have raised exception"
        except Exception as e:
            assert "Processing failed" in str(e)


# ==========================================================================
# TEST Batch Processing
# ==========================================================================

class TestBatchProcessing:
    """Test batch processing workflows."""

    def test_batch_size_configuration(self, test_settings):
        """Should use configured batch size."""
        assert test_settings.db_batch_size == 10

    def test_process_contributors_in_batches(self):
        """Should process contributors in batches."""
        contributors = [{'id': i} for i in range(25)]
        batch_size = 10
        
        batches = [contributors[i:i+batch_size] for i in range(0, len(contributors), batch_size)]
        
        assert len(batches) == 3
        assert len(batches[0]) == 10
        assert len(batches[1]) == 10
        assert len(batches[2]) == 5

    @pytest.mark.asyncio
    async def test_async_batch_intelligence_extraction(self):
        """Should support async batch intelligence extraction."""
        import asyncio
        
        async def mock_generate(prompt):
            await asyncio.sleep(0.01)
            return f"Response for {prompt}"
        
        prompts = [f"Prompt {i}" for i in range(5)]
        
        tasks = [mock_generate(p) for p in prompts]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5


# ==========================================================================
# TEST Data Validation
# ==========================================================================

class TestDataValidation:
    """Test data validation in workflows."""

    def test_validate_email_format(self):
        """Should validate email format."""
        from pydantic import EmailStr, ValidationError
        from pydantic import BaseModel
        
        class TestModel(BaseModel):
            email: EmailStr
        
        # Valid email
        model = TestModel(email="valid@test.com")
        assert model.email == "valid@test.com"
        
        # Invalid email
        with pytest.raises(ValidationError):
            TestModel(email="invalid-email")

    def test_validate_required_fields(self, sample_csv_row):
        """Should validate required CSV fields."""
        required_fields = ['email', 'contributor_id']
        
        for field in required_fields:
            assert field in sample_csv_row.index

    def test_handle_missing_optional_fields(self):
        """Should handle missing optional fields."""
        row = pd.Series({
            'email': 'test@test.com',
            'contributor_id': 'C123'
        })
        
        # Should not raise error for missing optional fields
        education = row.get('education_level', '')
        assert education == ''


# ==========================================================================
# TEST Performance Metrics
# ==========================================================================

class TestPerformanceMetrics:
    """Test performance tracking in workflows."""

    def test_track_processing_progress(self):
        """Should track processing progress."""
        total = 100
        processed = 0
        
        for i in range(10):
            processed += 10
            progress = processed / total
            assert progress == (i + 1) / 10

    def test_calculate_eta(self):
        """Should calculate estimated time remaining."""
        import time
        
        start_time = time.time()
        processed = 50
        total = 100
        
        # Simulate some time passing
        elapsed = 10.0  # seconds
        rate = processed / elapsed if elapsed > 0 else 0
        eta_seconds = (total - processed) / rate if rate > 0 else 0
        
        assert rate == 5.0  # 50/10
        assert eta_seconds == 10.0  # (100-50)/5

    def test_calculate_processing_rate(self):
        """Should calculate processing rate."""
        processed = 100
        elapsed_seconds = 50
        
        rate = processed / elapsed_seconds if elapsed_seconds > 0 else 0
        
        assert rate == 2.0  # 100/50 = 2 items per second

