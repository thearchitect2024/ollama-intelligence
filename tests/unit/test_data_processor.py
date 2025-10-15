"""
Unit tests for src/processors/data_processor.py
Tests CSV parsing and contributor processing.
"""
import pytest
import pandas as pd
from src.processors.data_processor import (
    parse_projects_json,
    parse_languages,
    process_contributor
)
from tests.fixtures.mock_data import (
    PROJECTS_JSON_PRODUCTION,
    PROJECTS_JSON_NO_DESCRIPTIONS,
    PROJECTS_JSON_EMPTY,
    PROJECTS_JSON_MALFORMED,
    LANGUAGES_JSON_VALID,
    LANGUAGES_JSON_NO_LEVEL,
    LANGUAGES_JSON_EMPTY
)


# ==========================================================================
# TEST parse_projects_json
# ==========================================================================

class TestParseProjectsJson:
    """Test parse_projects_json function."""

    def test_parse_valid_production_projects(self):
        """Should parse and filter production status projects."""
        projects = parse_projects_json(PROJECTS_JSON_PRODUCTION)
        
        # Should only return production projects (P1 and P2, not P3 which is Closed)
        assert len(projects) == 2
        assert projects[0].project_id == "P1"
        assert projects[0].project_type == "AI Training"
        assert projects[1].project_id == "P2"

    def test_parse_prioritizes_objective_job_description(self):
        """Should prioritize objective.job_description over project_long_description."""
        projects = parse_projects_json(PROJECTS_JSON_PRODUCTION)
        
        # First project has objective.job_description
        assert projects[0].long_desc == "AI model training with Python and TensorFlow"
        
        # Second project has only project_long_description
        assert projects[1].long_desc == "Data collection and validation tasks"

    def test_parse_empty_string(self):
        """Should handle empty string."""
        projects = parse_projects_json("")
        assert projects == []

    def test_parse_none_value(self):
        """Should handle None value."""
        projects = parse_projects_json(None)
        assert projects == []

    def test_parse_nan_value(self):
        """Should handle pandas NaN."""
        import numpy as np
        projects = parse_projects_json(np.nan)
        assert projects == []

    def test_parse_malformed_json(self):
        """Should handle malformed JSON gracefully."""
        projects = parse_projects_json(PROJECTS_JSON_MALFORMED)
        assert projects == []

    def test_parse_empty_json_array(self):
        """Should handle empty JSON array."""
        projects = parse_projects_json(PROJECTS_JSON_EMPTY)
        assert projects == []

    def test_parse_non_array_json(self):
        """Should handle non-array JSON."""
        projects = parse_projects_json('{"key": "value"}')
        assert projects == []

    def test_parse_filters_non_production(self):
        """Should filter out non-Production status projects."""
        json_str = '''[
            {"project_id": "P1", "project_status": "Production", "project_type": "AI"},
            {"project_id": "P2", "project_status": "Closed", "project_type": "Data"},
            {"project_id": "P3", "project_status": "Draft", "project_type": "Test"}
        ]'''
        projects = parse_projects_json(json_str)
        assert len(projects) == 1
        assert projects[0].project_id == "P1"

    def test_parse_handles_missing_fields(self):
        """Should handle missing optional fields."""
        json_str = '''[
            {
                "project_id": "P1",
                "project_status": "Production"
            }
        ]'''
        projects = parse_projects_json(json_str)
        assert len(projects) == 1
        assert projects[0].project_type == ""
        assert projects[0].account_name == ""
        assert projects[0].long_desc == ""

    def test_parse_handles_empty_objective(self):
        """Should fallback to project_long_description when objective is empty."""
        json_str = '''[
            {
                "project_id": "P1",
                "project_status": "Production",
                "objective": {},
                "project_long_description": "Fallback description"
            }
        ]'''
        projects = parse_projects_json(json_str)
        assert projects[0].long_desc == "Fallback description"

    def test_parse_handles_non_dict_account(self):
        """Should handle non-dict account field."""
        json_str = '''[
            {
                "project_id": "P1",
                "project_status": "Production",
                "account": "not a dict"
            }
        ]'''
        projects = parse_projects_json(json_str)
        assert len(projects) == 1
        assert projects[0].account_name == ""

    def test_parse_extracts_account_info(self):
        """Should extract account name and ID."""
        json_str = '''[
            {
                "project_id": "P1",
                "project_status": "Production",
                "account": {
                    "account_name": "Test Corp",
                    "account_id": "ACC123"
                }
            }
        ]'''
        projects = parse_projects_json(json_str)
        assert projects[0].account_name == "Test Corp"
        assert projects[0].account_id == "ACC123"


# ==========================================================================
# TEST parse_languages
# ==========================================================================

class TestParseLanguages:
    """Test parse_languages function."""

    def test_parse_valid_languages(self):
        """Should parse valid languages JSON."""
        languages = parse_languages(LANGUAGES_JSON_VALID)
        
        assert len(languages) == 3
        assert languages[0].language == "English"
        assert languages[0].proficiency == "Native"
        assert languages[1].language == "Spanish"
        assert languages[1].proficiency == "Fluent"

    def test_parse_maps_level_to_proficiency(self):
        """Should map 'level' field to 'proficiency'."""
        json_str = '{"language": "English", "level": "Advanced"}'
        # Parse as array
        json_array = f'[{json_str}]'
        languages = parse_languages(json_array)
        
        assert languages[0].proficiency == "Advanced"

    def test_parse_empty_string(self):
        """Should handle empty string."""
        languages = parse_languages("")
        assert languages == []

    def test_parse_none_value(self):
        """Should handle None."""
        languages = parse_languages(None)
        assert languages == []

    def test_parse_nan_value(self):
        """Should handle pandas NaN."""
        import numpy as np
        languages = parse_languages(np.nan)
        assert languages == []

    def test_parse_malformed_json(self):
        """Should handle malformed JSON."""
        languages = parse_languages("{invalid json}")
        assert languages == []

    def test_parse_empty_json_array(self):
        """Should handle empty array."""
        languages = parse_languages(LANGUAGES_JSON_EMPTY)
        assert languages == []

    def test_parse_non_array_json(self):
        """Should handle non-array JSON."""
        languages = parse_languages('{"language": "English"}')
        assert languages == []

    def test_parse_handles_missing_level(self):
        """Should handle missing level field."""
        languages = parse_languages(LANGUAGES_JSON_NO_LEVEL)
        assert len(languages) == 2
        assert languages[0].proficiency == ""

    def test_parse_removes_duplicates(self):
        """Should filter duplicate languages."""
        json_str = '''[
            {"language": "English", "level": "Native"},
            {"language": "English", "level": "Fluent"}
        ]'''
        languages = parse_languages(json_str)
        
        # Should keep only first occurrence
        assert len(languages) == 1
        assert languages[0].language == "English"

    def test_parse_strips_whitespace(self):
        """Should strip whitespace from language names."""
        json_str = '[{"language": "  English  ", "level": "  Native  "}]'
        languages = parse_languages(json_str)
        
        assert languages[0].language == "English"
        assert languages[0].proficiency == "Native"

    def test_parse_skips_empty_language_names(self):
        """Should skip entries with empty language names."""
        json_str = '''[
            {"language": "", "level": "Native"},
            {"language": "Spanish", "level": "Fluent"}
        ]'''
        languages = parse_languages(json_str)
        
        assert len(languages) == 1
        assert languages[0].language == "Spanish"


# ==========================================================================
# TEST process_contributor
# ==========================================================================

class TestProcessContributor:
    """Test process_contributor function."""

    def test_process_valid_contributor(self, sample_csv_row, test_settings):
        """Should process valid contributor row."""
        profile = process_contributor(sample_csv_row, test_settings)
        
        assert profile.contributor_email == "test@example.com"
        assert profile.contributor_id == "C123"
        assert profile.education_level == "Bachelor's Degree"
        assert profile.location.country == "United States"
        assert profile.location.us_state == "California"
        assert len(profile.languages) == 2

    def test_process_filters_production_projects(self, sample_csv_row, test_settings):
        """Should only include production status projects."""
        profile = process_contributor(sample_csv_row, test_settings)
        
        # CSV has 2 projects, only 1 is Production status
        assert len(profile.production_projects) == 1
        assert profile.production_projects[0].project_id == "P1"

    def test_process_calculates_activity_summary(self, sample_csv_row, test_settings):
        """Should calculate activity summary."""
        profile = process_contributor(sample_csv_row, test_settings)
        
        assert profile.activity_summary.total_production_projects == 1
        assert "AI Training" in profile.activity_summary.project_types_distribution

    def test_process_empty_row(self, sample_csv_row_empty, test_settings):
        """Should process row with minimal data."""
        profile = process_contributor(sample_csv_row_empty, test_settings)
        
        assert profile.contributor_email == "empty@example.com"
        assert len(profile.production_projects) == 0
        assert len(profile.languages) == 0

    def test_process_missing_email_raises_error(self, test_settings):
        """Should raise error for missing email."""
        row = pd.Series({'contributor_id': 'C123'})
        
        with pytest.raises(ValueError, match="Missing email"):
            process_contributor(row, test_settings)

    def test_process_empty_email_raises_error(self, test_settings):
        """Should raise error for empty email."""
        row = pd.Series({'email': '', 'contributor_id': 'C123'})
        
        with pytest.raises(ValueError, match="Missing email"):
            process_contributor(row, test_settings)

    def test_process_normalizes_email(self, test_settings):
        """Should normalize email to lowercase."""
        row = pd.Series({
            'email': 'TEST@EXAMPLE.COM',
            'contributor_id': 'C123',
            'projects_json': '[]',
            'languages_json': '[]'
        })
        profile = process_contributor(row, test_settings)
        
        assert profile.contributor_email == "test@example.com"

    def test_process_handles_missing_optional_fields(self, test_settings):
        """Should handle missing optional fields with defaults."""
        row = pd.Series({
            'email': 'test@test.com',
            'contributor_id': 'C123'
        })
        profile = process_contributor(row, test_settings)
        
        assert profile.education_level == ""
        assert profile.location.country == ""

    def test_process_sets_completed_status(self, sample_csv_row, test_settings):
        """Should set processing status to COMPLETED."""
        profile = process_contributor(sample_csv_row, test_settings)
        
        from src.models import ProcessingStatus
        assert profile.processing_status == ProcessingStatus.COMPLETED

    def test_process_handles_none_us_state(self, test_settings):
        """Should handle None US state."""
        row = pd.Series({
            'email': 'test@test.com',
            'contributor_id': 'C123',
            'currently_residing_country_c': 'Canada',
            'currently_residing_us_state_c': None,
            'projects_json': '[]',
            'languages_json': '[]'
        })
        profile = process_contributor(row, test_settings)
        
        assert profile.location.country == "Canada"
        assert profile.location.us_state is None

    def test_process_parses_compliance_fields(self, sample_csv_row, test_settings):
        """Should parse compliance fields."""
        profile = process_contributor(sample_csv_row, test_settings)
        
        assert profile.compliance.kyc_status == "Passed"
        assert profile.compliance.dots_status == "Verified"
        assert profile.compliance.risk_tier == "Low"

    def test_process_with_malformed_json_continues(self, test_settings):
        """Should handle malformed JSON gracefully."""
        row = pd.Series({
            'email': 'test@test.com',
            'contributor_id': 'C123',
            'projects_json': '{invalid}',
            'languages_json': '{bad}',
            'currently_residing_country_c': 'USA'
        })
        profile = process_contributor(row, test_settings)
        
        # Should still create profile with empty lists
        assert profile.production_projects == []
        assert profile.languages == []

