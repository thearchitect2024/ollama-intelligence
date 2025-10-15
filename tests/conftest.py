"""
Pytest configuration and shared fixtures.
"""
import pytest
import os
import sys
from unittest.mock import Mock, MagicMock
from datetime import datetime
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Settings
from src.models import (
    ContributorProfile, ActivitySummary, ProjectInfo,
    Location, Language, Compliance, ProcessingStatus
)


# ==========================================================================
# CONFIGURATION FIXTURES
# ==========================================================================

@pytest.fixture
def test_settings():
    """Return test settings."""
    return Settings(
        postgres_host="localhost",
        postgres_port=5432,
        postgres_db="test_db",
        postgres_user="test_user",
        postgres_password="test_password",
        ollama_model="test-model",
        ollama_base_url="http://localhost:11434",
        max_tokens=256,
        temperature=0.3,
        top_p=0.9,
        max_concurrent_llm=5
    )


# ==========================================================================
# MODEL FIXTURES
# ==========================================================================

@pytest.fixture
def sample_project_info():
    """Return sample ProjectInfo."""
    return ProjectInfo(
        project_id="P123",
        project_type="AI Training",
        account_name="Acme Corp",
        account_id="A001",
        long_desc="This is a test project for AI training tasks including data labeling and annotation."
    )


@pytest.fixture
def sample_projects_list():
    """Return list of sample projects."""
    return [
        ProjectInfo(
            project_id="P1",
            project_type="AI Training",
            account_name="Acme Corp",
            account_id="A001",
            long_desc="AI data labeling project"
        ),
        ProjectInfo(
            project_id="P2",
            project_type="Data Collection",
            account_name="Tech Inc",
            account_id="A002",
            long_desc="Data collection and validation"
        ),
        ProjectInfo(
            project_id="P3",
            project_type="AI Training",
            account_name="Acme Corp",
            account_id="A001",
            long_desc="Another AI training task"
        )
    ]


@pytest.fixture
def sample_activity_summary():
    """Return sample ActivitySummary."""
    return ActivitySummary(
        total_production_projects=3,
        project_types_distribution={"AI Training": 2, "Data Collection": 1},
        unique_clients=["Acme Corp", "Tech Inc"]
    )


@pytest.fixture
def sample_location():
    """Return sample Location."""
    return Location(country="United States", us_state="California")


@pytest.fixture
def sample_languages():
    """Return sample Language list."""
    return [
        Language(language="English", proficiency="Native"),
        Language(language="Spanish", proficiency="Fluent")
    ]


@pytest.fixture
def sample_compliance():
    """Return sample Compliance."""
    return Compliance(
        kyc_status="Passed",
        dots_status="Verified",
        risk_tier="Low"
    )


@pytest.fixture
def sample_contributor_profile(sample_projects_list, sample_activity_summary,
                                sample_location, sample_languages, sample_compliance):
    """Return complete sample ContributorProfile."""
    return ContributorProfile(
        contributor_email="test@example.com",
        contributor_id="C123",
        education_level="Bachelor's Degree",
        production_projects=sample_projects_list,
        activity_summary=sample_activity_summary,
        location=sample_location,
        languages=sample_languages,
        compliance=sample_compliance,
        extracted_skills=["Python", "Data Analysis", "Machine Learning"],
        intelligence_summary="Test summary",
        processing_status=ProcessingStatus.COMPLETED
    )


# ==========================================================================
# CSV DATA FIXTURES
# ==========================================================================

@pytest.fixture
def sample_csv_row():
    """Return sample CSV row as pandas Series."""
    return pd.Series({
        'email': 'test@example.com',
        'contributor_id': 'C123',
        'highest_education_level_c': "Bachelor's Degree",
        'currently_residing_country_c': 'United States',
        'currently_residing_us_state_c': 'California',
        'kyc_status_c': 'Passed',
        'dots_status_c': 'Verified',
        'risk_tier_c': 'Low',
        'languages_json': '''[
            {"language": "English", "level": "Native"},
            {"language": "Spanish", "level": "Fluent"}
        ]''',
        'projects_json': '''[
            {
                "project_id": "P1",
                "project_status": "Production",
                "project_type": "AI Training",
                "account": {"account_name": "Acme Corp", "account_id": "A001"},
                "objective": {"job_description": "AI data labeling project"}
            },
            {
                "project_id": "P2",
                "project_status": "Closed",
                "project_type": "Other",
                "account": {"account_name": "Test", "account_id": "A002"}
            }
        ]'''
    })


@pytest.fixture
def sample_csv_row_empty():
    """Return CSV row with minimal data."""
    return pd.Series({
        'email': 'empty@example.com',
        'contributor_id': 'C456',
        'highest_education_level_c': '',
        'currently_residing_country_c': '',
        'languages_json': '[]',
        'projects_json': '[]'
    })


# ==========================================================================
# DATABASE MOCK FIXTURES
# ==========================================================================

@pytest.fixture
def mock_db_manager():
    """Return mock DatabaseManager."""
    mock_db = MagicMock()
    mock_db.execute_query = MagicMock()
    return mock_db


@pytest.fixture
def mock_contributor_repository(mock_db_manager):
    """Return mock ContributorRepository."""
    from src.database.repositories import ContributorRepository
    return ContributorRepository(mock_db_manager)


# ==========================================================================
# LLM MOCK FIXTURES
# ==========================================================================

@pytest.fixture
def mock_llm_client(test_settings):
    """Return mock OllamaClient."""
    mock_client = MagicMock()
    mock_client.settings = test_settings
    mock_client.generate = MagicMock(return_value="""SUMMARY:
This contributor is based in California, United States and speaks English (Native), Spanish (Fluent). Holds Bachelor's Degree. Has worked on 3 production projects across 2 project types including AI Training (2), Data Collection (1). Notable clients include: Acme Corp, Tech Inc. Compliance: KYC: Passed, DOTS: Verified, Risk Tier: Low.

SKILLS:
- Python
- Data Analysis
- Machine Learning
- Data Labeling
- AI Training""")
    mock_client.generate_async = MagicMock()
    mock_client.generate_batch = MagicMock()
    mock_client.is_available = MagicMock(return_value=True)
    return mock_client


@pytest.fixture
def mock_llm_response_valid():
    """Return valid LLM response."""
    return """SUMMARY:
This contributor is based in California, United States and speaks 2 languages. Has 5 years experience.

SKILLS:
- Python
- React
- AWS"""


@pytest.fixture
def mock_llm_response_no_skills():
    """Return LLM response without skills section."""
    return "This is just a summary without skills section."


# ==========================================================================
# JSON FIXTURES
# ==========================================================================

@pytest.fixture
def valid_projects_json():
    """Return valid projects JSON string."""
    return '''[
        {
            "project_id": "P1",
            "project_status": "Production",
            "project_type": "AI Training",
            "account": {"account_name": "Acme", "account_id": "A1"},
            "objective": {"job_description": "Test description"},
            "project_long_description": "Fallback description"
        }
    ]'''


@pytest.fixture
def invalid_json():
    """Return malformed JSON string."""
    return '{invalid json: true'


@pytest.fixture
def valid_languages_json():
    """Return valid languages JSON string."""
    return '''[
        {"language": "English", "level": "Native"},
        {"language": "Spanish", "level": "Intermediate"}
    ]'''

