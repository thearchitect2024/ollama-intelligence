"""
Mock data and test fixtures for testing.
"""
import json


# ==========================================================================
# PROJECTS JSON MOCK DATA
# ==========================================================================

PROJECTS_JSON_PRODUCTION = json.dumps([
    {
        "project_id": "P1",
        "project_status": "Production",
        "project_type": "AI Training",
        "account": {"account_name": "Acme Corp", "account_id": "A001"},
        "objective": {"job_description": "AI model training with Python and TensorFlow"},
        "project_long_description": "Fallback description"
    },
    {
        "project_id": "P2",
        "project_status": "Production",
        "project_type": "Data Collection",
        "account": {"account_name": "Tech Inc", "account_id": "A002"},
        "project_long_description": "Data collection and validation tasks"
    },
    {
        "project_id": "P3",
        "project_status": "Closed",
        "project_type": "Testing",
        "account": {"account_name": "Test Co", "account_id": "A003"}
    }
])

PROJECTS_JSON_NO_DESCRIPTIONS = json.dumps([
    {
        "project_id": "P1",
        "project_status": "Production",
        "project_type": "AI Training",
        "account": {"account_name": "Acme", "account_id": "A1"},
        "objective": {},
        "project_long_description": ""
    }
])

PROJECTS_JSON_EMPTY = "[]"
PROJECTS_JSON_MALFORMED = "{invalid: json}"


# ==========================================================================
# LANGUAGES JSON MOCK DATA
# ==========================================================================

LANGUAGES_JSON_VALID = json.dumps([
    {"language": "English", "level": "Native"},
    {"language": "Spanish", "level": "Fluent"},
    {"language": "French", "level": "Intermediate"}
])

LANGUAGES_JSON_NO_LEVEL = json.dumps([
    {"language": "English"},
    {"language": "Spanish", "level": ""}
])

LANGUAGES_JSON_EMPTY = "[]"


# ==========================================================================
# LLM RESPONSES
# ==========================================================================

LLM_RESPONSE_VALID = """SUMMARY:
This contributor is based in California, United States and speaks English (Native), Spanish (Fluent). Holds Bachelor's Degree. Has worked on 5 production projects across 3 project types including AI Training (3), Data Collection (1), Content Moderation (1). Notable clients include: Acme Corp, Tech Inc, Global Solutions. Compliance: KYC: Passed, DOTS: Verified, Risk Tier: Low.

SKILLS:
- Python Programming
- Machine Learning
- Data Labeling
- TensorFlow
- Content Moderation
- API Integration
- React.js
"""

LLM_RESPONSE_NO_SKILLS = """SUMMARY:
This contributor is based in New York, United States and speaks 2 languages. Has worked on multiple projects.
"""

LLM_RESPONSE_MALFORMED = "This is just random text without proper structure"


# ==========================================================================
# DATABASE MOCK RESPONSES
# ==========================================================================

DB_CONTRIBUTOR_RECORD = {
    'email': 'test@example.com',
    'contributor_id': 'C123',
    'processed_data': {
        'contributor_email': 'test@example.com',
        'contributor_id': 'C123',
        'education_level': "Bachelor's Degree",
        'production_projects': [],
        'activity_summary': {
            'total_production_projects': 0,
            'project_types_distribution': {},
            'unique_clients': []
        },
        'location': {'country': 'USA', 'us_state': None},
        'languages': [],
        'compliance': {'kyc_status': 'Passed', 'dots_status': 'Verified', 'risk_tier': 'Low'},
        'extracted_skills': [],
        'intelligence_summary': None
    },
    'intelligence_summary': None,
    'processing_status': 'completed',
    'error_message': None,
    'created_at': '2025-01-01T00:00:00',
    'updated_at': '2025-01-01T00:00:00',
    'intelligence_extracted_at': None
}

