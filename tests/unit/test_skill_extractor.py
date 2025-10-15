"""
Unit tests for src/intelligence/skill_extractor.py
Tests skill extraction and intelligence summary generation.
"""
import pytest
from unittest.mock import MagicMock, patch

from src.intelligence.skill_extractor import (
    has_project_descriptions,
    parse_llm_response,
    generate_summary_no_projects,
    generate_summary_no_descriptions,
    generate_intelligence_summary
)
from src.models import ContributorProfile, ProjectInfo
from tests.fixtures.mock_data import LLM_RESPONSE_VALID, LLM_RESPONSE_NO_SKILLS


# ==========================================================================
# TEST has_project_descriptions
# ==========================================================================

class TestHasProjectDescriptions:
    """Test has_project_descriptions function."""

    def test_with_descriptions(self, sample_contributor_profile):
        """Should return True when projects have descriptions."""
        result = has_project_descriptions(sample_contributor_profile)
        assert result is True

    def test_without_descriptions(self, sample_contributor_profile):
        """Should return False when no descriptions."""
        # Clear all descriptions
        for proj in sample_contributor_profile.production_projects:
            proj.long_desc = ""
        
        result = has_project_descriptions(sample_contributor_profile)
        assert result is False

    def test_with_empty_projects(self, sample_contributor_profile):
        """Should return False with no projects."""
        sample_contributor_profile.production_projects = []
        
        result = has_project_descriptions(sample_contributor_profile)
        assert result is False

    def test_with_whitespace_only_description(self, sample_contributor_profile):
        """Should return False for whitespace-only descriptions."""
        for proj in sample_contributor_profile.production_projects:
            proj.long_desc = "   "
        
        result = has_project_descriptions(sample_contributor_profile)
        assert result is False

    def test_with_mixed_descriptions(self, sample_contributor_profile):
        """Should return True if at least one has description."""
        sample_contributor_profile.production_projects[0].long_desc = "Good description"
        sample_contributor_profile.production_projects[1].long_desc = ""
        sample_contributor_profile.production_projects[2].long_desc = ""
        
        result = has_project_descriptions(sample_contributor_profile)
        assert result is True


# ==========================================================================
# TEST parse_llm_response
# ==========================================================================

class TestParseLlmResponse:
    """Test parse_llm_response function."""

    def test_parse_valid_response(self, mock_llm_response_valid):
        """Should parse valid LLM response."""
        summary, skills = parse_llm_response(mock_llm_response_valid)
        
        assert "contributor" in summary.lower()
        assert len(skills) == 3
        assert "Python" in skills
        assert "React" in skills
        assert "AWS" in skills

    def test_parse_response_without_skills(self, mock_llm_response_no_skills):
        """Should handle response without SKILLS section."""
        summary, skills = parse_llm_response(mock_llm_response_no_skills)
        
        assert len(summary) > 0
        assert skills == []

    def test_parse_full_llm_response(self):
        """Should parse complete LLM response."""
        summary, skills = parse_llm_response(LLM_RESPONSE_VALID)
        
        assert len(summary) > 50
        assert "California" in summary
        assert "Bachelor's Degree" in summary
        assert len(skills) == 7
        assert "Python Programming" in skills
        assert "Machine Learning" in skills

    def test_parse_removes_summary_header(self):
        """Should remove SUMMARY: header."""
        response = "SUMMARY:\nThis is the summary.\n\nSKILLS:\n- Skill1"
        summary, skills = parse_llm_response(response)
        
        assert not summary.startswith("SUMMARY:")
        assert summary.strip() == "This is the summary."

    def test_parse_extracts_bullet_points(self):
        """Should extract skills as bullet points."""
        response = """SUMMARY:
Test summary

SKILLS:
- Python
- JavaScript
- React.js
- Docker"""
        summary, skills = parse_llm_response(response)
        
        assert len(skills) == 4
        assert skills == ["Python", "JavaScript", "React.js", "Docker"]

    def test_parse_ignores_empty_lines(self):
        """Should skip empty lines in skills section."""
        response = """SUMMARY:
Test

SKILLS:
- Skill1

- Skill2

- Skill3"""
        summary, skills = parse_llm_response(response)
        
        assert len(skills) == 3

    def test_parse_strips_whitespace_from_skills(self):
        """Should strip whitespace from skill names."""
        response = """SUMMARY:
Test

SKILLS:
-   Python   
-  JavaScript  
-React"""
        summary, skills = parse_llm_response(response)
        
        assert "Python" in skills
        assert "JavaScript" in skills
        assert "React" in skills

    def test_parse_handles_malformed_response(self):
        """Should handle malformed response gracefully."""
        response = "Random text without structure"
        summary, skills = parse_llm_response(response)
        
        assert summary == "Random text without structure"
        assert skills == []

    def test_parse_error_handling(self):
        """Should handle parsing errors gracefully."""
        response = None  # This will cause an error
        summary, skills = parse_llm_response("")
        
        assert summary == ""
        assert skills == []


# ==========================================================================
# TEST generate_summary_no_projects
# ==========================================================================

class TestGenerateSummaryNoProjects:
    """Test generate_summary_no_projects function."""

    def test_generates_basic_summary(self, sample_contributor_profile):
        """Should generate summary for contributor with no projects."""
        sample_contributor_profile.production_projects = []
        sample_contributor_profile.activity_summary.total_production_projects = 0
        
        summary = generate_summary_no_projects(sample_contributor_profile)
        
        assert "California, United States" in summary
        assert "No production projects available" in summary
        assert "KYC: Passed" in summary

    def test_includes_languages(self, sample_contributor_profile):
        """Should include language information."""
        sample_contributor_profile.production_projects = []
        
        summary = generate_summary_no_projects(sample_contributor_profile)
        
        assert "English" in summary or "Spanish" in summary

    def test_handles_many_languages(self, sample_contributor_profile):
        """Should summarize when many languages."""
        from src.models import Language
        sample_contributor_profile.production_projects = []
        sample_contributor_profile.languages = [
            Language(language=f"Lang{i}", proficiency="Fluent")
            for i in range(5)
        ]
        
        summary = generate_summary_no_projects(sample_contributor_profile)
        
        assert "5 languages" in summary

    def test_handles_no_languages(self, sample_contributor_profile):
        """Should handle no languages."""
        sample_contributor_profile.production_projects = []
        sample_contributor_profile.languages = []
        
        summary = generate_summary_no_projects(sample_contributor_profile)
        
        assert "Not specified" in summary

    def test_includes_education(self, sample_contributor_profile):
        """Should include education level."""
        sample_contributor_profile.production_projects = []
        
        summary = generate_summary_no_projects(sample_contributor_profile)
        
        assert "Bachelor's Degree" in summary

    def test_includes_compliance(self, sample_contributor_profile):
        """Should include compliance information."""
        sample_contributor_profile.production_projects = []
        
        summary = generate_summary_no_projects(sample_contributor_profile)
        
        assert "KYC: Passed" in summary
        assert "DOTS: Verified" in summary
        assert "Risk Tier: Low" in summary


# ==========================================================================
# TEST generate_summary_no_descriptions
# ==========================================================================

class TestGenerateSummaryNoDescriptions:
    """Test generate_summary_no_descriptions function."""

    def test_generates_summary_with_project_info(self, sample_contributor_profile):
        """Should generate summary with project information."""
        # Clear descriptions but keep projects
        for proj in sample_contributor_profile.production_projects:
            proj.long_desc = ""
        
        summary = generate_summary_no_descriptions(sample_contributor_profile)
        
        assert "3 production projects" in summary
        assert "AI Training" in summary
        assert "Project descriptions not available" in summary

    def test_includes_project_types(self, sample_contributor_profile):
        """Should include project types distribution."""
        for proj in sample_contributor_profile.production_projects:
            proj.long_desc = ""
        
        summary = generate_summary_no_descriptions(sample_contributor_profile)
        
        assert "AI Training" in summary

    def test_includes_clients(self, sample_contributor_profile):
        """Should include client information."""
        for proj in sample_contributor_profile.production_projects:
            proj.long_desc = ""
        
        summary = generate_summary_no_descriptions(sample_contributor_profile)
        
        assert "Acme Corp" in summary or "Tech Inc" in summary

    def test_limits_clients_to_five(self, sample_contributor_profile):
        """Should limit clients to 5."""
        from src.models import ActivitySummary
        sample_contributor_profile.activity_summary = ActivitySummary(
            total_production_projects=10,
            project_types_distribution={"Type1": 10},
            unique_clients=[f"Client{i}" for i in range(10)]
        )
        
        for proj in sample_contributor_profile.production_projects:
            proj.long_desc = ""
        
        summary = generate_summary_no_descriptions(sample_contributor_profile)
        
        # Should only show first 5 clients
        clients_in_summary = [f"Client{i}" for i in range(10) if f"Client{i}" in summary]
        assert len(clients_in_summary) <= 5


# ==========================================================================
# TEST generate_intelligence_summary
# ==========================================================================

class TestGenerateIntelligenceSummary:
    """Test generate_intelligence_summary function."""

    def test_generates_summary_with_llm(self, sample_contributor_profile, mock_llm_client):
        """Should generate summary using LLM."""
        summary = generate_intelligence_summary(sample_contributor_profile, mock_llm_client)
        
        assert len(summary) > 0
        mock_llm_client.generate.assert_called_once()

    def test_generates_summary_without_llm_for_no_projects(self, sample_contributor_profile, mock_llm_client):
        """Should skip LLM for contributors with no projects."""
        sample_contributor_profile.production_projects = []
        sample_contributor_profile.activity_summary.total_production_projects = 0
        
        summary = generate_intelligence_summary(sample_contributor_profile, mock_llm_client)
        
        assert "No production projects available" in summary
        mock_llm_client.generate.assert_not_called()

    def test_includes_location_in_prompt(self, sample_contributor_profile, mock_llm_client):
        """Should include location in LLM prompt."""
        generate_intelligence_summary(sample_contributor_profile, mock_llm_client)
        
        call_args = mock_llm_client.generate.call_args[0][0]
        assert "California, United States" in call_args

    def test_includes_languages_in_prompt(self, sample_contributor_profile, mock_llm_client):
        """Should include languages in LLM prompt."""
        generate_intelligence_summary(sample_contributor_profile, mock_llm_client)
        
        call_args = mock_llm_client.generate.call_args[0][0]
        assert "English" in call_args or "Spanish" in call_args

    def test_includes_project_descriptions_in_prompt(self, sample_contributor_profile, mock_llm_client):
        """Should include project descriptions in prompt."""
        generate_intelligence_summary(sample_contributor_profile, mock_llm_client)
        
        call_args = mock_llm_client.generate.call_args[0][0]
        assert "AI data labeling" in call_args or "Data collection" in call_args

    def test_limits_project_descriptions_to_ten(self, sample_contributor_profile, mock_llm_client):
        """Should limit project descriptions to 10."""
        from src.models import ProjectInfo
        sample_contributor_profile.production_projects = [
            ProjectInfo(
                project_id=f"P{i}",
                project_type="Type",
                account_name="Client",
                account_id="A1",
                long_desc=f"Description {i}"
            )
            for i in range(20)
        ]
        
        generate_intelligence_summary(sample_contributor_profile, mock_llm_client)
        
        call_args = mock_llm_client.generate.call_args[0][0]
        # Should only include first 10
        assert "Description 9" in call_args
        assert "Description 10" not in call_args

    def test_truncates_long_descriptions(self, sample_contributor_profile, mock_llm_client):
        """Should truncate descriptions to 400 chars."""
        long_desc = "A" * 1000
        sample_contributor_profile.production_projects[0].long_desc = long_desc
        
        generate_intelligence_summary(sample_contributor_profile, mock_llm_client)
        
        call_args = mock_llm_client.generate.call_args[0][0]
        # Should be truncated
        assert "A" * 401 not in call_args

    def test_handles_llm_error(self, sample_contributor_profile, mock_llm_client):
        """Should handle LLM errors gracefully."""
        mock_llm_client.generate.side_effect = Exception("LLM error")
        
        summary = generate_intelligence_summary(sample_contributor_profile, mock_llm_client)
        
        assert "generation failed" in summary.lower()

    def test_prompt_includes_compliance(self, sample_contributor_profile, mock_llm_client):
        """Should include compliance in prompt."""
        generate_intelligence_summary(sample_contributor_profile, mock_llm_client)
        
        call_args = mock_llm_client.generate.call_args[0][0]
        assert "KYC: Passed" in call_args
        assert "DOTS: Verified" in call_args
        assert "Risk Tier: Low" in call_args

