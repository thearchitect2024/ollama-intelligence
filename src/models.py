"""
Pydantic data models for Contributor Intelligence Platform.
Defines type-safe data structures for all entities.
"""
from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum


# ==========================================================================
# ENUMERATIONS
# ==========================================================================

class ProcessingStatus(str, Enum):
    """Processing status for contributors."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ==========================================================================
# PROJECT MODELS
# ==========================================================================

class ProjectInfo(BaseModel):
    """Simplified project information for production projects only."""
    project_id: str = Field(..., description="Unique project identifier")
    project_type: str = Field(default="", description="Type of project")
    account_name: str = Field(default="", description="Client/Account name")
    account_id: str = Field(default="", description="Client/Account ID")
    long_desc: str = Field(default="", description="Job description or project description for skill extraction")


# ==========================================================================
# ACTIVITY SUMMARY MODEL
# ==========================================================================

class ActivitySummary(BaseModel):
    """Summary of contributor activity based on production projects (count-based, no time tracking)."""
    total_production_projects: int = Field(default=0, ge=0, description="Total number of production projects")
    project_types_distribution: Dict[str, int] = Field(default_factory=dict, description="Count of projects by type")
    unique_clients: List[str] = Field(default_factory=list, description="List of unique client/account names")


# ==========================================================================
# LOCATION & LANGUAGE MODELS
# ==========================================================================

class Location(BaseModel):
    """Contributor location information."""
    country: str = Field(default="", description="Country of residence")
    us_state: Optional[str] = Field(default=None, description="US state (if applicable)")

    def __str__(self) -> str:
        """String representation of location."""
        if self.us_state:
            return f"{self.us_state}, {self.country}"
        return self.country


class Language(BaseModel):
    """Language proficiency information."""
    language: str = Field(..., description="Language name")
    proficiency: str = Field(default="", description="Proficiency level")

    def __str__(self) -> str:
        """String representation of language."""
        if self.proficiency:
            return f"{self.language} ({self.proficiency})"
        return self.language


# ==========================================================================
# COMPLIANCE MODEL
# ==========================================================================

class Compliance(BaseModel):
    """Simplified compliance and verification status."""
    kyc_status: str = Field(default="", description="KYC status")
    dots_status: str = Field(default="", description="DOTS verification status")
    risk_tier: str = Field(default="", description="Risk tier level")


# ==========================================================================
# CONTRIBUTOR PROFILE MODEL
# ==========================================================================

class ContributorProfile(BaseModel):
    """Complete contributor profile with all data."""
    contributor_email: EmailStr = Field(..., description="Contributor email (PRIMARY KEY)")
    contributor_id: str = Field(..., description="Contributor ID for reference")
    education_level: str = Field(default="", description="Highest education level")
    production_projects: List[ProjectInfo] = Field(default_factory=list, description="ALL production status projects")
    activity_summary: ActivitySummary = Field(..., description="Activity metrics from production projects")
    location: Location = Field(..., description="Location information")
    languages: List[Language] = Field(default_factory=list, description="Languages known")
    compliance: Compliance = Field(..., description="Compliance and verification status")
    extracted_skills: List[str] = Field(default_factory=list, description="Extracted skills from project descriptions")
    intelligence_summary: Optional[str] = Field(default=None, description="AI-generated intelligence summary")

    # Metadata
    created_at: Optional[datetime] = Field(default=None, description="Record creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Last update timestamp")
    intelligence_extracted_at: Optional[datetime] = Field(default=None, description="Intelligence extraction timestamp")
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="Processing status")
    error_message: Optional[str] = Field(default=None, description="Error message if processing failed")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
        use_enum_values = True

    def model_dump_json_safe(self) -> dict:
        """Export model as JSON-safe dictionary for database storage."""
        return self.model_dump(
            exclude={'created_at', 'updated_at', 'intelligence_extracted_at', 'processing_status', 'error_message'},
            exclude_none=False  # Changed from True - include intelligence_summary even if None
        )


# ==========================================================================
# DATABASE RECORD MODEL
# ==========================================================================

class ContributorRecord(BaseModel):
    """Database record representation with metadata."""
    email: EmailStr
    contributor_id: str
    processed_data: dict  # JSONB field containing ContributorProfile
    intelligence_summary: Optional[str] = None
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    intelligence_extracted_at: Optional[datetime] = None

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
        use_enum_values = True
