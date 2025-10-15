"""
Core data processing pipeline for contributor profiles.
Processes CSV input and generates ContributorProfile objects.
"""
import json
import logging
import pandas as pd
from typing import List, Dict, Any

from src.config import Settings
from src.models import (
    ContributorProfile, ActivitySummary, ProjectInfo,
    Location, Language, Compliance,
    ProcessingStatus
)
from src.processors.activity_analyzer import calculate_activity_summary

logger = logging.getLogger(__name__)


def parse_projects_json(projects_str: str) -> List[ProjectInfo]:
    """
    Parse projects_json and return ONLY Production status projects.
    Prioritizes objective.job_description over project_long_description.
    """
    if not projects_str or pd.isna(projects_str):
        return []

    try:
        projects_data = json.loads(projects_str)
        if not isinstance(projects_data, list):
            return []

        production_projects = []

        for proj in projects_data:
            # FILTER: Only Production status
            if proj.get('project_status') != 'Production':
                continue

            # PRIORITY: objective.job_description > project_long_description
            long_desc = ""
            objective = proj.get('objective', {})
            if objective and isinstance(objective, dict) and objective.get('job_description'):
                long_desc = str(objective['job_description']).strip()

            if not long_desc:
                long_desc = str(proj.get('project_long_description', '')).strip()

            # Extract account info
            account = proj.get('account', {})
            if not isinstance(account, dict):
                account = {}

            production_projects.append(ProjectInfo(
                project_id=str(proj.get('project_id', '')),
                project_type=str(proj.get('project_type', '')),
                account_name=str(account.get('account_name', '')),
                account_id=str(account.get('account_id', '')),
                long_desc=long_desc
            ))

        logger.info(f"Parsed {len(production_projects)} production projects from {len(projects_data)} total projects")
        return production_projects

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse projects JSON: {e}")
        return []
    except Exception as e:
        logger.error(f"Error parsing projects: {e}")
        return []


def parse_languages(languages_str: str) -> List[Language]:
    """Parse languages_json column (maps 'level' field to 'proficiency')."""
    if not languages_str or pd.isna(languages_str):
        return []
    try:
        languages_data = json.loads(languages_str)
        if not isinstance(languages_data, list):
            return []

        languages = []
        seen = set()
        for lang_dict in languages_data:
            language = str(lang_dict.get('language', '')).strip()
            # Map 'level' to 'proficiency'
            proficiency = str(lang_dict.get('level', '')).strip()

            if language and language not in seen:
                languages.append(Language(language=language, proficiency=proficiency))
                seen.add(language)

        return languages
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse languages JSON: {e}")
        return []
    except Exception as e:
        logger.error(f"Error parsing languages: {e}")
        return []


def process_contributor(row: pd.Series, settings: Settings) -> ContributorProfile:
    """
    Main processing pipeline for a single contributor with new CSV structure.

    Args:
        row: CSV row containing contributor data
        settings: Application settings

    Returns:
        ContributorProfile: Processed contributor profile
    """
    try:
        # Extract email (PRIMARY KEY)
        email = str(row.get('email', '')).strip().lower()
        if not email:
            raise ValueError("Missing email")

        contributor_id = str(row.get('contributor_id', ''))

        logger.info(f"Processing contributor: {email}")

        # Parse ALL production projects from projects_json
        projects_str = row.get('projects_json', '[]')
        production_projects = parse_projects_json(projects_str)

        # Calculate activity summary from production projects
        activity_summary = calculate_activity_summary(production_projects)

        # Parse languages from languages_json
        languages_str = row.get('languages_json', '[]')
        languages = parse_languages(languages_str)

        # Extract education level
        education_level = str(row.get('highest_education_level_c', ''))

        # Build location
        location = Location(
            country=str(row.get('currently_residing_country_c', '')),
            us_state=str(row.get('currently_residing_us_state_c', '')) if pd.notna(row.get('currently_residing_us_state_c')) else None
        )

        # Build compliance (simplified - 3 fields only)
        compliance = Compliance(
            kyc_status=str(row.get('kyc_status_c', '')),
            dots_status=str(row.get('dots_status_c', '')),
            risk_tier=str(row.get('risk_tier_c', ''))
        )

        # Build complete profile
        profile = ContributorProfile(
            contributor_email=email,
            contributor_id=contributor_id,
            education_level=education_level,
            production_projects=production_projects,
            activity_summary=activity_summary,
            location=location,
            languages=languages,
            compliance=compliance,
            processing_status=ProcessingStatus.COMPLETED
        )

        logger.info(f"Successfully processed: {email} ({len(production_projects)} production projects)")
        return profile

    except Exception as e:
        logger.error(f"Failed to process contributor {row.get('email', 'UNKNOWN')}: {e}")
        raise


def process_csv_batch(csv_path: str, settings: Settings) -> List[ContributorProfile]:
    """
    Process entire CSV file in batch.

    Args:
        csv_path: Path to CSV file
        settings: Application settings

    Returns:
        List[ContributorProfile]: List of processed profiles
    """
    logger.info(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    total_rows = len(df)

    logger.info(f"Processing {total_rows} contributors")

    profiles = []
    successful = 0
    failed = 0

    for index, row in df.iterrows():
        try:
            profile = process_contributor(row, settings)
            profiles.append(profile)
            successful += 1

            if (index + 1) % 10 == 0:
                logger.info(f"Progress: {index + 1}/{total_rows}")

        except Exception as e:
            failed += 1
            logger.error(f"Failed row {index}: {e}")

    logger.info(f"Processing complete: {successful} successful, {failed} failed")
    return profiles
