"""
Activity analysis logic for production projects.
Calculates contributor activity based on project counts (no time tracking).
"""
import logging
from typing import List
from src.models import ActivitySummary, ProjectInfo

logger = logging.getLogger(__name__)


def calculate_activity_summary(projects: List[ProjectInfo]) -> ActivitySummary:
    """
    Calculate activity summary from production projects (count-based only, no time tracking).

    Args:
        projects: List of ProjectInfo objects (production projects only)

    Returns:
        ActivitySummary: Activity metrics
    """
    try:
        total = len(projects)

        # Count by project type
        type_distribution = {}
        for proj in projects:
            ptype = proj.project_type if proj.project_type else "Unknown"
            type_distribution[ptype] = type_distribution.get(ptype, 0) + 1

        # Extract unique clients (sorted)
        unique_clients = sorted(set(
            proj.account_name for proj in projects
            if proj.account_name
        ))

        return ActivitySummary(
            total_production_projects=total,
            project_types_distribution=type_distribution,
            unique_clients=unique_clients
        )

    except Exception as e:
        logger.error(f"Error calculating activity summary: {e}")
        # Return empty summary
        return ActivitySummary(
            total_production_projects=0,
            project_types_distribution={},
            unique_clients=[]
        )


