"""
Skill extraction and intelligence summary generation.
Generates 140-170 word contributor summaries using LLM from ALL production projects.
"""
import logging
from typing import Tuple, List
from src.models import ContributorProfile
from src.intelligence.llm_client import OllamaClient

logger = logging.getLogger(__name__)


def has_project_descriptions(profile: ContributorProfile) -> bool:
    """
    Check if any project has non-empty description.

    Args:
        profile: ContributorProfile to check

    Returns:
        bool: True if at least one project has long_desc with content
    """
    return any(
        proj.long_desc and proj.long_desc.strip()
        for proj in profile.production_projects
    )


def parse_llm_response(llm_output: str) -> Tuple[str, List[str]]:
    """
    Parse structured LLM output into summary text and skills list.

    Expected format:
        SUMMARY:
        [text content]

        SKILLS:
        - Skill 1
        - Skill 2

    Args:
        llm_output: Raw LLM response text

    Returns:
        Tuple of (summary_text, skills_list)
        If parsing fails, returns (full_text, [])
    """
    try:
        # Split on "SKILLS:" marker
        if "SKILLS:" not in llm_output:
            logger.warning("LLM output missing SKILLS section, using full text as summary")
            return llm_output.strip(), []

        parts = llm_output.split("SKILLS:")

        # Extract summary (everything before SKILLS:)
        summary_text = parts[0].replace("SUMMARY:", "").strip()

        # Extract skills (everything after SKILLS:)
        skills_section = parts[1].strip()

        # Parse bullet points into list
        skills_list = []
        for line in skills_section.split('\n'):
            line = line.strip()
            if line.startswith('-'):
                skill = line[1:].strip()  # Remove '-' and whitespace
                if skill:
                    skills_list.append(skill)

        logger.info(f"Parsed LLM response: summary={len(summary_text)} chars, skills={len(skills_list)}")
        return summary_text, skills_list

    except Exception as e:
        logger.error(f"Failed to parse LLM response: {e}, using fallback")
        return llm_output.strip(), []


def generate_summary_no_projects(profile: ContributorProfile) -> str:
    """
    Generate factual summary when NO production projects exist.

    Args:
        profile: ContributorProfile with no projects

    Returns:
        str: Simple factual summary
    """
    location_str = str(profile.location)

    if profile.languages:
        lang_count = len(profile.languages)
        if lang_count <= 3:
            languages_str = ', '.join([lang.language for lang in profile.languages])
        else:
            top_3 = [lang.language for lang in profile.languages[:3]]
            languages_str = f"{lang_count} languages including {', '.join(top_3)}"
    else:
        languages_str = 'Not specified'

    education_str = profile.education_level if profile.education_level else 'Not specified'

    comp = profile.compliance
    compliance_str = f"KYC: {comp.kyc_status}, DOTS: {comp.dots_status}, Risk Tier: {comp.risk_tier}"

    summary = (
        f"This contributor is based in {location_str} and speaks {languages_str}. "
        f"Holds {education_str}. "
        f"No production projects available. "
        f"Compliance: {compliance_str}."
    )

    logger.info(f"Generated summary for contributor with no projects: {profile.contributor_email}")
    return summary


def generate_summary_no_descriptions(profile: ContributorProfile) -> str:
    """
    Generate factual summary when projects exist but ALL descriptions are empty.

    Args:
        profile: ContributorProfile with projects but no descriptions

    Returns:
        str: Factual summary with project info but note about missing descriptions
    """
    location_str = str(profile.location)

    if profile.languages:
        lang_count = len(profile.languages)
        if lang_count <= 3:
            languages_str = ', '.join([lang.language for lang in profile.languages])
        else:
            top_3 = [lang.language for lang in profile.languages[:3]]
            languages_str = f"{lang_count} languages including {', '.join(top_3)}"
    else:
        languages_str = 'Not specified'

    education_str = profile.education_level if profile.education_level else 'Not specified'

    activity = profile.activity_summary
    total_projects = activity.total_production_projects

    # Project types (top 3 by count)
    if activity.project_types_distribution:
        sorted_types = sorted(activity.project_types_distribution.items(),
                             key=lambda x: x[1], reverse=True)[:3]
        types_str = ', '.join([f"{t[0]} ({t[1]})" for t in sorted_types])
    else:
        types_str = "None"

    # Clients (up to 5)
    if activity.unique_clients:
        clients_str = ', '.join(activity.unique_clients[:5])
    else:
        clients_str = "None"

    comp = profile.compliance
    compliance_str = f"KYC: {comp.kyc_status}, DOTS: {comp.dots_status}, Risk Tier: {comp.risk_tier}"

    summary = (
        f"This contributor is based in {location_str} and speaks {languages_str}. "
        f"Holds {education_str}. "
        f"Has worked on {total_projects} production projects across project types including {types_str}. "
        f"Notable clients include: {clients_str}. "
        f"Project descriptions not available - skills cannot be extracted. "
        f"Compliance: {compliance_str}."
    )

    logger.info(f"Generated summary for contributor with no descriptions: {profile.contributor_email}")
    return summary


def generate_intelligence_summary(profile: ContributorProfile, llm_client: OllamaClient) -> str:
    """
    Generate 140-170 word intelligence summary for contributor using ALL production projects.

    Args:
        profile: ContributorProfile object
        llm_client: Ollama LLM client

    Returns:
        str: Intelligence summary
    """
    try:
        # 1. LOCATION
        location_str = str(profile.location)

        # 2. LANGUAGES (summarize if many)
        if profile.languages:
            lang_count = len(profile.languages)
            if lang_count <= 3:
                languages_str = ', '.join([lang.language for lang in profile.languages])
            else:
                top_3 = [lang.language for lang in profile.languages[:3]]
                languages_str = f"{lang_count} languages including {', '.join(top_3)}"
        else:
            languages_str = 'Not specified'

        # 3. EDUCATION
        education_str = profile.education_level if profile.education_level else 'Not specified'

        # 4. ACTIVITY SUMMARY
        activity = profile.activity_summary
        total_projects = activity.total_production_projects

        # Project types (top 3 by count)
        if activity.project_types_distribution:
            sorted_types = sorted(activity.project_types_distribution.items(),
                                 key=lambda x: x[1], reverse=True)[:3]
            types_str = ', '.join([f"{t[0]} ({t[1]})" for t in sorted_types])
            type_count = len(activity.project_types_distribution)
        else:
            types_str = "None"
            type_count = 0

        # Clients (up to 5)
        if activity.unique_clients:
            clients_str = ', '.join(activity.unique_clients[:5])
        else:
            clients_str = "None"

        # 5. PROJECT DESCRIPTIONS FOR SKILL EXTRACTION (limit to 10 for prompt size)
        project_descriptions = []
        for i, proj in enumerate(profile.production_projects[:10], 1):
            desc = proj.long_desc[:400] if proj.long_desc else "No description"
            project_descriptions.append(f"{i}. [{proj.project_type}] {desc}")

        projects_block = '\n'.join(project_descriptions) if project_descriptions else 'No production projects'

        # 6. COMPLIANCE
        comp = profile.compliance
        compliance_str = f"KYC: {comp.kyc_status}, DOTS: {comp.dots_status}, Risk Tier: {comp.risk_tier}"

        # HANDLE EMPTY PROJECTS - Return simple summary without LLM
        if not profile.production_projects:
            summary = (
                f"This contributor is based in {location_str} and speaks {languages_str}. "
                f"Holds {education_str}. "
                f"No production projects available. "
                f"Compliance: {compliance_str}."
            )
            logger.info(f"Generated summary for contributor with no projects: {profile.contributor_email}")
            return summary

        # BUILD PROMPT - NEW FORMAT WITH SKILLS EXTRACTION
        prompt = f"""Write a contributor intelligence summary and extract skills.

Data:
- Location: {location_str}
- Languages: {languages_str}
- Education: {education_str}
- Production Projects: {total_projects}
- Project Types ({type_count}): {types_str}
- Notable Clients: {clients_str}
- Project Descriptions (for skill extraction):
{projects_block}
- Compliance: {compliance_str}

OUTPUT FORMAT:

SUMMARY:
Write a 90-120 word factual paragraph. Include location, languages, education, project counts, types, clients, and compliance. DO NOT describe skills in the summary.

Instructions for summary:
1. Start: "This contributor is based in [location] and speaks [languages]. Holds [education]."
2. Projects: "Has worked on [N] production projects across [X] project types including [types]. Notable clients include: [clients]."
3. End: "Compliance: {compliance_str}"

SKILLS:
Extract 5-10 key technical skills, technologies, or domain expertise from the project descriptions above. List as bullet points starting with "-".

Example:
- Python
- React.js
- Data Analysis
"""

        # GENERATE
        summary = llm_client.generate(prompt)
        logger.info(f"Generated response for {profile.contributor_email} ({len(profile.production_projects)} production projects)")

        return summary

    except Exception as e:
        logger.error(f"Failed to generate summary for {profile.contributor_email}: {e}")
        return f"Intelligence summary generation failed: {str(e)}"
