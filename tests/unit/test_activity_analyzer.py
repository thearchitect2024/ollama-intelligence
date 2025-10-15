"""
Unit tests for src/processors/activity_analyzer.py
Tests activity summary calculation from production projects.
"""
import pytest
from src.processors.activity_analyzer import calculate_activity_summary
from src.models import ProjectInfo, ActivitySummary


class TestCalculateActivitySummary:
    """Test calculate_activity_summary function."""

    def test_calculate_with_multiple_projects(self, sample_projects_list):
        """Should calculate correct activity summary."""
        activity = calculate_activity_summary(sample_projects_list)
        
        assert activity.total_production_projects == 3
        assert activity.project_types_distribution["AI Training"] == 2
        assert activity.project_types_distribution["Data Collection"] == 1
        assert "Acme Corp" in activity.unique_clients
        assert "Tech Inc" in activity.unique_clients
        assert len(activity.unique_clients) == 2

    def test_calculate_with_empty_projects(self):
        """Should handle empty project list."""
        activity = calculate_activity_summary([])
        
        assert activity.total_production_projects == 0
        assert activity.project_types_distribution == {}
        assert activity.unique_clients == []

    def test_calculate_with_single_project(self):
        """Should handle single project."""
        projects = [
            ProjectInfo(
                project_id="P1",
                project_type="AI Training",
                account_name="Solo Client",
                account_id="A1",
                long_desc="Test"
            )
        ]
        activity = calculate_activity_summary(projects)
        
        assert activity.total_production_projects == 1
        assert activity.project_types_distribution["AI Training"] == 1
        assert activity.unique_clients == ["Solo Client"]

    def test_calculate_with_duplicate_clients(self):
        """Should count unique clients only."""
        projects = [
            ProjectInfo(project_id="P1", project_type="Type1", account_name="Client A", account_id="A1"),
            ProjectInfo(project_id="P2", project_type="Type1", account_name="Client A", account_id="A1"),
            ProjectInfo(project_id="P3", project_type="Type2", account_name="Client B", account_id="A2")
        ]
        activity = calculate_activity_summary(projects)
        
        assert len(activity.unique_clients) == 2
        assert "Client A" in activity.unique_clients
        assert "Client B" in activity.unique_clients

    def test_calculate_with_missing_project_type(self):
        """Should handle missing project types."""
        projects = [
            ProjectInfo(project_id="P1", project_type="", account_name="Client", account_id="A1"),
            ProjectInfo(project_id="P2", project_type="", account_name="Client", account_id="A1")
        ]
        activity = calculate_activity_summary(projects)
        
        assert activity.total_production_projects == 2
        assert activity.project_types_distribution["Unknown"] == 2

    def test_calculate_with_empty_account_names(self):
        """Should handle empty account names."""
        projects = [
            ProjectInfo(project_id="P1", project_type="Type1", account_name="", account_id=""),
            ProjectInfo(project_id="P2", project_type="Type1", account_name="Client", account_id="A1")
        ]
        activity = calculate_activity_summary(projects)
        
        # Empty account names should not be included
        assert len(activity.unique_clients) == 1
        assert "Client" in activity.unique_clients

    def test_calculate_project_type_distribution(self):
        """Should correctly count project type distribution."""
        projects = [
            ProjectInfo(project_id="P1", project_type="AI Training", account_name="C1", account_id="A1"),
            ProjectInfo(project_id="P2", project_type="AI Training", account_name="C2", account_id="A2"),
            ProjectInfo(project_id="P3", project_type="AI Training", account_name="C3", account_id="A3"),
            ProjectInfo(project_id="P4", project_type="Data Collection", account_name="C4", account_id="A4"),
            ProjectInfo(project_id="P5", project_type="Content Moderation", account_name="C5", account_id="A5")
        ]
        activity = calculate_activity_summary(projects)
        
        assert activity.project_types_distribution["AI Training"] == 3
        assert activity.project_types_distribution["Data Collection"] == 1
        assert activity.project_types_distribution["Content Moderation"] == 1
        assert len(activity.project_types_distribution) == 3

    def test_calculate_clients_are_sorted(self):
        """Should return sorted unique clients."""
        projects = [
            ProjectInfo(project_id="P1", project_type="T1", account_name="Zebra Corp", account_id="A1"),
            ProjectInfo(project_id="P2", project_type="T1", account_name="Alpha Inc", account_id="A2"),
            ProjectInfo(project_id="P3", project_type="T1", account_name="Beta LLC", account_id="A3")
        ]
        activity = calculate_activity_summary(projects)
        
        # Should be alphabetically sorted
        assert activity.unique_clients == ["Alpha Inc", "Beta LLC", "Zebra Corp"]

    def test_calculate_with_many_projects(self):
        """Should handle large number of projects."""
        projects = [
            ProjectInfo(
                project_id=f"P{i}",
                project_type=f"Type{i % 5}",
                account_name=f"Client{i % 10}",
                account_id=f"A{i}"
            )
            for i in range(100)
        ]
        activity = calculate_activity_summary(projects)
        
        assert activity.total_production_projects == 100
        assert len(activity.project_types_distribution) == 5
        assert len(activity.unique_clients) == 10

    def test_calculate_error_handling_returns_empty_summary(self):
        """Should return empty summary on error."""
        # Pass invalid data that would cause an error
        activity = calculate_activity_summary(None)
        
        assert activity.total_production_projects == 0
        assert activity.project_types_distribution == {}
        assert activity.unique_clients == []

