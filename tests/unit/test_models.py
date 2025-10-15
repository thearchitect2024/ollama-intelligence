"""
Unit tests for src/models.py
Tests Pydantic models, validation, and serialization.
"""
import pytest
from datetime import datetime
from pydantic import ValidationError

from src.models import (
    ContributorProfile, ActivitySummary, ProjectInfo,
    Location, Language, Compliance, ProcessingStatus,
    ContributorRecord
)


# ==========================================================================
# TEST ProjectInfo
# ==========================================================================

class TestProjectInfo:
    """Test ProjectInfo model."""

    def test_valid_project_info(self):
        """Should create valid ProjectInfo."""
        project = ProjectInfo(
            project_id="P123",
            project_type="AI Training",
            account_name="Acme Corp",
            account_id="A001",
            long_desc="Test description"
        )
        assert project.project_id == "P123"
        assert project.project_type == "AI Training"
        assert project.account_name == "Acme Corp"
        assert project.long_desc == "Test description"

    def test_project_info_with_defaults(self):
        """Should handle default values."""
        project = ProjectInfo(project_id="P123")
        assert project.project_id == "P123"
        assert project.project_type == ""
        assert project.account_name == ""
        assert project.long_desc == ""

    def test_project_info_missing_required_field(self):
        """Should raise error for missing required field."""
        with pytest.raises(ValidationError):
            ProjectInfo()


# ==========================================================================
# TEST ActivitySummary
# ==========================================================================

class TestActivitySummary:
    """Test ActivitySummary model."""

    def test_valid_activity_summary(self):
        """Should create valid ActivitySummary."""
        activity = ActivitySummary(
            total_production_projects=5,
            project_types_distribution={"AI Training": 3, "Data Collection": 2},
            unique_clients=["Acme", "Tech Inc"]
        )
        assert activity.total_production_projects == 5
        assert len(activity.project_types_distribution) == 2
        assert len(activity.unique_clients) == 2

    def test_activity_summary_with_defaults(self):
        """Should use default values."""
        activity = ActivitySummary()
        assert activity.total_production_projects == 0
        assert activity.project_types_distribution == {}
        assert activity.unique_clients == []

    def test_activity_summary_negative_projects(self):
        """Should reject negative project count."""
        with pytest.raises(ValidationError):
            ActivitySummary(total_production_projects=-1)

    def test_activity_summary_zero_projects(self):
        """Should allow zero projects."""
        activity = ActivitySummary(total_production_projects=0)
        assert activity.total_production_projects == 0


# ==========================================================================
# TEST Location
# ==========================================================================

class TestLocation:
    """Test Location model."""

    def test_location_with_us_state(self):
        """Should create location with US state."""
        location = Location(country="United States", us_state="California")
        assert location.country == "United States"
        assert location.us_state == "California"
        assert str(location) == "California, United States"

    def test_location_without_us_state(self):
        """Should create location without US state."""
        location = Location(country="Canada")
        assert location.country == "Canada"
        assert location.us_state is None
        assert str(location) == "Canada"

    def test_location_empty_country(self):
        """Should allow empty country."""
        location = Location(country="")
        assert location.country == ""

    def test_location_str_representation(self):
        """Should format string correctly."""
        loc1 = Location(country="USA", us_state="TX")
        assert str(loc1) == "TX, USA"
        
        loc2 = Location(country="UK")
        assert str(loc2) == "UK"


# ==========================================================================
# TEST Language
# ==========================================================================

class TestLanguage:
    """Test Language model."""

    def test_language_with_proficiency(self):
        """Should create language with proficiency."""
        lang = Language(language="English", proficiency="Native")
        assert lang.language == "English"
        assert lang.proficiency == "Native"
        assert str(lang) == "English (Native)"

    def test_language_without_proficiency(self):
        """Should create language without proficiency."""
        lang = Language(language="Spanish", proficiency="")
        assert lang.language == "Spanish"
        assert str(lang) == "Spanish"

    def test_language_missing_required(self):
        """Should require language field."""
        with pytest.raises(ValidationError):
            Language(proficiency="Fluent")

    def test_language_str_representation(self):
        """Should format string correctly."""
        lang1 = Language(language="French", proficiency="Intermediate")
        assert str(lang1) == "French (Intermediate)"
        
        lang2 = Language(language="German")
        assert str(lang2) == "German"


# ==========================================================================
# TEST Compliance
# ==========================================================================

class TestCompliance:
    """Test Compliance model."""

    def test_valid_compliance(self):
        """Should create valid Compliance."""
        compliance = Compliance(
            kyc_status="Passed",
            dots_status="Verified",
            risk_tier="Low"
        )
        assert compliance.kyc_status == "Passed"
        assert compliance.dots_status == "Verified"
        assert compliance.risk_tier == "Low"

    def test_compliance_with_defaults(self):
        """Should use default empty strings."""
        compliance = Compliance()
        assert compliance.kyc_status == ""
        assert compliance.dots_status == ""
        assert compliance.risk_tier == ""


# ==========================================================================
# TEST ContributorProfile
# ==========================================================================

class TestContributorProfile:
    """Test ContributorProfile model."""

    def test_valid_contributor_profile(self, sample_contributor_profile):
        """Should create valid ContributorProfile."""
        profile = sample_contributor_profile
        assert profile.contributor_email == "test@example.com"
        assert profile.contributor_id == "C123"
        assert len(profile.production_projects) == 3
        assert len(profile.languages) == 2
        assert profile.processing_status == ProcessingStatus.COMPLETED

    def test_contributor_profile_minimal(self):
        """Should create profile with minimal required fields."""
        profile = ContributorProfile(
            contributor_email="min@test.com",
            contributor_id="C999",
            activity_summary=ActivitySummary(),
            location=Location(country="USA"),
            compliance=Compliance()
        )
        assert profile.contributor_email == "min@test.com"
        assert profile.production_projects == []
        assert profile.languages == []

    def test_contributor_profile_invalid_email(self):
        """Should reject invalid email."""
        with pytest.raises(ValidationError):
            ContributorProfile(
                contributor_email="invalid-email",
                contributor_id="C123",
                activity_summary=ActivitySummary(),
                location=Location(country="USA"),
                compliance=Compliance()
            )

    def test_contributor_profile_missing_required_fields(self):
        """Should require essential fields."""
        with pytest.raises(ValidationError):
            ContributorProfile(contributor_email="test@test.com")

    def test_contributor_profile_default_status(self):
        """Should default to PENDING status."""
        profile = ContributorProfile(
            contributor_email="test@test.com",
            contributor_id="C123",
            activity_summary=ActivitySummary(),
            location=Location(country="USA"),
            compliance=Compliance()
        )
        assert profile.processing_status == ProcessingStatus.PENDING

    def test_contributor_profile_with_skills(self):
        """Should store extracted skills."""
        profile = ContributorProfile(
            contributor_email="skilled@test.com",
            contributor_id="C456",
            activity_summary=ActivitySummary(),
            location=Location(country="USA"),
            compliance=Compliance(),
            extracted_skills=["Python", "React", "AWS"]
        )
        assert len(profile.extracted_skills) == 3
        assert "Python" in profile.extracted_skills

    def test_contributor_profile_with_intelligence_summary(self):
        """Should store intelligence summary."""
        summary = "This is a test intelligence summary."
        profile = ContributorProfile(
            contributor_email="intel@test.com",
            contributor_id="C789",
            activity_summary=ActivitySummary(),
            location=Location(country="USA"),
            compliance=Compliance(),
            intelligence_summary=summary
        )
        assert profile.intelligence_summary == summary

    def test_model_dump_json_safe(self, sample_contributor_profile):
        """Should export JSON-safe dictionary."""
        profile = sample_contributor_profile
        data = profile.model_dump_json_safe()
        
        assert "contributor_email" in data
        assert "production_projects" in data
        assert "activity_summary" in data
        
        # Should exclude metadata fields
        assert "created_at" not in data
        assert "updated_at" not in data
        assert "processing_status" not in data

    def test_model_dump_json_safe_includes_none_intelligence(self):
        """Should include intelligence_summary even if None."""
        profile = ContributorProfile(
            contributor_email="test@test.com",
            contributor_id="C123",
            activity_summary=ActivitySummary(),
            location=Location(country="USA"),
            compliance=Compliance(),
            intelligence_summary=None
        )
        data = profile.model_dump_json_safe()
        assert "intelligence_summary" in data


# ==========================================================================
# TEST ContributorRecord
# ==========================================================================

class TestContributorRecord:
    """Test ContributorRecord model."""

    def test_valid_contributor_record(self):
        """Should create valid ContributorRecord."""
        now = datetime.now()
        record = ContributorRecord(
            email="test@example.com",
            contributor_id="C123",
            processed_data={"key": "value"},
            intelligence_summary="Test summary",
            processing_status=ProcessingStatus.COMPLETED,
            created_at=now,
            updated_at=now
        )
        assert record.email == "test@example.com"
        assert record.processing_status == ProcessingStatus.COMPLETED

    def test_contributor_record_with_error(self):
        """Should store error message."""
        now = datetime.now()
        record = ContributorRecord(
            email="fail@test.com",
            contributor_id="C456",
            processed_data={},
            processing_status=ProcessingStatus.FAILED,
            error_message="Processing failed",
            created_at=now,
            updated_at=now
        )
        assert record.processing_status == ProcessingStatus.FAILED
        assert record.error_message == "Processing failed"


# ==========================================================================
# TEST ProcessingStatus Enum
# ==========================================================================

class TestProcessingStatus:
    """Test ProcessingStatus enum."""

    def test_processing_status_values(self):
        """Should have correct enum values."""
        assert ProcessingStatus.PENDING.value == "pending"
        assert ProcessingStatus.PROCESSING.value == "processing"
        assert ProcessingStatus.COMPLETED.value == "completed"
        assert ProcessingStatus.FAILED.value == "failed"

    def test_processing_status_string_comparison(self):
        """Should work with string comparison."""
        status = ProcessingStatus.COMPLETED
        assert status == ProcessingStatus.COMPLETED
        assert status.value == "completed"

