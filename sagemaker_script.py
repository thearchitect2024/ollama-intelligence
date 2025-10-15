#!/usr/bin/env python3
"""
Contributor Intelligence Platform - SageMaker Studio Script
File-Based Database (SQLite) Version
"""

import os
import sys
import sqlite3
import json
import re
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

SQLITE_DB_PATH = '/tmp/contributor_intelligence.db'
CSV_FILE_PATH = '/home/sagemaker-user/contributor_data.csv'  # UPDATE THIS
OLLAMA_MODEL = 'qwen2.5:7b-instruct-q4_0'
OLLAMA_BASE_URL = 'http://localhost:11434'
MAX_CONCURRENT = 10
CHUNK_SIZE = 500

# ============================================================================
# DATA MODELS
# ============================================================================

class ProjectInfo(BaseModel):
    name: str
    description: Optional[str] = None
    hours: float = 0.0
    environment: Optional[str] = None

class ActivitySummary(BaseModel):
    total_hours: float = 0.0
    total_projects: int = 0
    production_projects: int = 0
    avg_hours_per_project: float = 0.0

class ContributorProfile(BaseModel):
    email: str
    contributor_id: str
    country: Optional[str] = None
    us_state: Optional[str] = None
    risk_tier: Optional[str] = None
    kyc_status: Optional[str] = None
    dots_status: Optional[str] = None
    languages: List[str] = Field(default_factory=list)
    qualifications: Optional[str] = None
    projects: List[ProjectInfo] = Field(default_factory=list)
    activity_summary: ActivitySummary = Field(default_factory=ActivitySummary)
    skills: List[str] = Field(default_factory=list)
    intelligence_summary: Optional[str] = None

# ============================================================================
# DATABASE MANAGER
# ============================================================================

class SQLiteDatabaseManager:
    """SQLite database manager."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS contributors (
                    email TEXT PRIMARY KEY,
                    contributor_id TEXT UNIQUE NOT NULL,
                    processed_data TEXT NOT NULL,
                    intelligence_summary TEXT,
                    processing_status TEXT DEFAULT 'pending',
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    intelligence_extracted_at TIMESTAMP
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_status ON contributors(processing_status)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_created ON contributors(created_at)')
            conn.commit()
        logger.info(f"SQLite database initialized: {self.db_path}")
    
    def execute_query(self, query: str, params: tuple = None, fetch_one: bool = False, fetch_all: bool = False):
        """Execute a query and return results."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        try:
            cursor.execute(query, params or ())
            if fetch_one:
                row = cursor.fetchone()
                return dict(row) if row else None
            elif fetch_all:
                return [dict(row) for row in cursor.fetchall()]
            conn.commit()
            return None
        finally:
            cursor.close()
            conn.close()

# ============================================================================
# DATA PROCESSING
# ============================================================================

def parse_projects_json(projects_str: str) -> List[ProjectInfo]:
    """Parse projects JSON string."""
    if not projects_str or pd.isna(projects_str):
        return []
    try:
        projects_data = json.loads(projects_str)
        projects = []
        for p in projects_data:
            if p.get('environment') in ['Production', 'production']:
                projects.append(ProjectInfo(
                    name=p.get('project_name', 'Unknown'),
                    description=p.get('description'),
                    hours=float(p.get('hours', 0)),
                    environment=p.get('environment')
                ))
        return projects
    except:
        return []

def parse_languages(languages_str: str) -> List[str]:
    """Parse languages JSON string."""
    if not languages_str or pd.isna(languages_str):
        return []
    try:
        return json.loads(languages_str) if isinstance(languages_str, str) else []
    except:
        return []

def process_contributor_row(row: pd.Series) -> ContributorProfile:
    """Process a single CSV row into a ContributorProfile."""
    email = str(row.get('email', '')).strip().lower()
    if not email:
        raise ValueError("Missing email")
    
    projects = parse_projects_json(str(row.get('projects_info', row.get('projects_json', ''))))
    languages = parse_languages(str(row.get('languages_known', row.get('languages_json', ''))))
    
    activity = ActivitySummary(
        total_hours=sum(p.hours for p in projects),
        total_projects=len(projects),
        production_projects=len(projects),
        avg_hours_per_project=sum(p.hours for p in projects) / len(projects) if projects else 0
    )
    
    return ContributorProfile(
        email=email,
        contributor_id=str(row.get('contributor_id', email.split('@')[0])),
        country=str(row.get('currently_residing_country__c', '')) if pd.notna(row.get('currently_residing_country__c')) else None,
        us_state=str(row.get('currently_residing_us_state__c', '')) if pd.notna(row.get('currently_residing_us_state__c')) else None,
        risk_tier=str(row.get('risk_tier__c', '')) if pd.notna(row.get('risk_tier__c')) else None,
        kyc_status=str(row.get('kyc_status__c', '')) if pd.notna(row.get('kyc_status__c')) else None,
        dots_status=str(row.get('dots_status__c', '')) if pd.notna(row.get('dots_status__c')) else None,
        languages=languages,
        qualifications=str(row.get('qualifications_summary', '')) if pd.notna(row.get('qualifications_summary')) else None,
        projects=projects,
        activity_summary=activity
    )

# ============================================================================
# DATABASE OPERATIONS
# ============================================================================

def upsert_contributor(db: SQLiteDatabaseManager, profile: ContributorProfile):
    """Insert or update contributor in database."""
    processed_data = json.dumps(profile.dict())
    
    query = '''
        INSERT OR REPLACE INTO contributors 
        (email, contributor_id, processed_data, updated_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
    '''
    
    db.execute_query(query, (profile.email, profile.contributor_id, processed_data))

def get_pending_contributors(db: SQLiteDatabaseManager) -> List[Dict]:
    """Get contributors without intelligence summaries."""
    query = '''
        SELECT email, contributor_id, processed_data 
        FROM contributors 
        WHERE intelligence_summary IS NULL OR intelligence_summary = ''
    '''
    results = db.execute_query(query, fetch_all=True)
    
    for result in results:
        result['processed_data'] = json.loads(result['processed_data'])
    
    return results

def update_intelligence(db: SQLiteDatabaseManager, email: str, summary: str, skills: List[str]):
    """Update contributor with intelligence summary."""
    existing = db.execute_query(
        'SELECT processed_data FROM contributors WHERE email = ?',
        (email,),
        fetch_one=True
    )
    
    if existing:
        data = json.loads(existing['processed_data'])
        data['intelligence_summary'] = summary
        data['skills'] = skills
        
        query = '''
            UPDATE contributors 
            SET intelligence_summary = ?,
                processed_data = ?,
                intelligence_extracted_at = CURRENT_TIMESTAMP,
                processing_status = 'completed'
            WHERE email = ?
        '''
        db.execute_query(query, (summary, json.dumps(data), email))

# ============================================================================
# LLM CLIENT
# ============================================================================

class SimpleLLMClient:
    """Simple async LLM client for Ollama."""
    
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
    
    async def generate_async(self, prompt: str) -> str:
        """Generate text from prompt asynchronously."""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.05,
                            "top_p": 0.9,
                            "num_predict": 320
                        }
                    },
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    result = await response.json()
                    return result.get('response', '')
            except Exception as e:
                logger.error(f"LLM generation error: {e}")
                return ""
    
    async def generate_batch(self, prompts: List[str], max_concurrent: int = 10) -> List[str]:
        """Generate text for multiple prompts concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_limit(prompt: str) -> str:
            async with semaphore:
                return await self.generate_async(prompt)
        
        tasks = [process_with_limit(p) for p in prompts]
        return await asyncio.gather(*tasks, return_exceptions=False)

# ============================================================================
# INTELLIGENCE EXTRACTION
# ============================================================================

def create_prompt(contributor: Dict) -> Optional[str]:
    """Create prompt for LLM."""
    data = contributor['processed_data']
    projects = data.get('projects', [])
    
    if not projects:
        return None
    
    descriptions = [p['description'] for p in projects if p.get('description')]
    
    if not descriptions:
        return None
    
    projects_text = "\n".join([f"- {desc}" for desc in descriptions[:5]])
    
    prompt = f"""Based on the following project descriptions, write a professional 150-word summary of this contributor's work and extract 3-5 key skills.

Project Descriptions:
{projects_text}

Format your response exactly as:
SUMMARY: [150-word professional summary]
SKILLS: [skill1, skill2, skill3]
"""
    return prompt

def parse_llm_response(response: str) -> tuple[str, List[str]]:
    """Parse LLM response into summary and skills."""
    summary = ""
    skills = []
    
    # Extract summary
    summary_match = re.search(r'SUMMARY:\s*(.+?)(?=SKILLS:|$)', response, re.DOTALL | re.IGNORECASE)
    if summary_match:
        summary = summary_match.group(1).strip()
    
    # Extract skills
    skills_match = re.search(r'SKILLS:\s*(.+)', response, re.DOTALL | re.IGNORECASE)
    if skills_match:
        skills_text = skills_match.group(1).strip()
        skills = [s.strip() for s in re.split(r'[,\n]', skills_text) if s.strip()]
        skills = [s.strip('[]"\' ') for s in skills[:5]]
    
    return summary, skills

# ============================================================================
# MAIN PROCESSING
# ============================================================================

async def process_intelligence_extraction(db: SQLiteDatabaseManager, llm_client: SimpleLLMClient):
    """Process intelligence extraction for all pending contributors."""
    pending = get_pending_contributors(db)
    
    if not pending:
        logger.info("No pending contributors")
        return 0, 0
    
    processed = 0
    failed = 0
    start_time = datetime.now()
    
    logger.info(f"Processing {len(pending)} contributors with {MAX_CONCURRENT} concurrent requests")
    
    # Process in batches
    for i in range(0, len(pending), MAX_CONCURRENT):
        batch = pending[i:i+MAX_CONCURRENT]
        
        prompts = []
        valid_contributors = []
        
        for contributor in batch:
            prompt = create_prompt(contributor)
            if prompt:
                prompts.append(prompt)
                valid_contributors.append(contributor)
            else:
                update_intelligence(
                    db,
                    contributor['email'],
                    "Contributor with limited project information available.",
                    []
                )
                processed += 1
        
        if prompts:
            try:
                responses = await llm_client.generate_batch(prompts, MAX_CONCURRENT)
                
                for contributor, response in zip(valid_contributors, responses):
                    try:
                        summary, skills = parse_llm_response(response)
                        if summary:
                            update_intelligence(db, contributor['email'], summary, skills)
                            processed += 1
                        else:
                            failed += 1
                    except Exception as e:
                        logger.error(f"Error processing {contributor['email']}: {e}")
                        failed += 1
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                failed += len(prompts)
        
        # Progress update
        total_done = min(i + MAX_CONCURRENT, len(pending))
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        speed = total_done / elapsed if elapsed > 0 else 0
        remaining = (len(pending) - total_done) / speed if speed > 0 else 0
        
        print(f"\r⚡ Progress: {total_done}/{len(pending)} | "
              f"Speed: {speed:.1f} profiles/min | "
              f"ETA: {remaining:.1f} min", end='')
    
    elapsed = (datetime.now() - start_time).total_seconds() / 60
    print(f"\n\n✅ Extraction Complete!")
    print(f"   Successful: {processed}")
    print(f"   Failed: {failed}")
    print(f"   Time: {elapsed:.1f} minutes")
    print(f"   Speed: {processed/elapsed:.2f} profiles/min")
    
    return processed, failed

def main():
    """Main execution function."""
    print("="*80)
    print("Contributor Intelligence Platform - SageMaker Studio")
    print("="*80)
    
    # Initialize database
    print(f"\n📦 Initializing database: {SQLITE_DB_PATH}")
    db = SQLiteDatabaseManager(SQLITE_DB_PATH)
    
    # Load and process CSV
    print(f"\n📥 Loading CSV: {CSV_FILE_PATH}")
    
    if not os.path.exists(CSV_FILE_PATH):
        print(f"❌ CSV file not found: {CSV_FILE_PATH}")
        print("   Please update CSV_FILE_PATH in the script")
        return
    
    total_processed = 0
    total_failed = 0
    start_time = datetime.now()
    
    for chunk_num, chunk_df in enumerate(pd.read_csv(CSV_FILE_PATH, chunksize=CHUNK_SIZE)):
        print(f"\n📦 Processing chunk {chunk_num + 1} ({len(chunk_df)} rows)...")
        
        for idx, row in tqdm(chunk_df.iterrows(), total=len(chunk_df), desc="Processing"):
            try:
                profile = process_contributor_row(row)
                upsert_contributor(db, profile)
                total_processed += 1
            except Exception as e:
                total_failed += 1
                if total_failed <= 5:
                    logger.error(f"Row {idx} error: {str(e)[:100]}")
    
    elapsed = (datetime.now() - start_time).total_seconds() / 60
    
    print(f"\n✅ CSV Import Complete!")
    print(f"   Processed: {total_processed}")
    print(f"   Failed: {total_failed}")
    print(f"   Time: {elapsed:.1f} minutes")
    
    # Extract intelligence
    print(f"\n🚀 Starting Intelligence Extraction...")
    llm_client = SimpleLLMClient(OLLAMA_BASE_URL, OLLAMA_MODEL)
    
    processed, failed = asyncio.run(process_intelligence_extraction(db, llm_client))
    
    # Show statistics
    stats = db.execute_query(
        '''
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN intelligence_summary IS NOT NULL THEN 1 ELSE 0 END) as with_intelligence,
            SUM(CASE WHEN intelligence_summary IS NULL THEN 1 ELSE 0 END) as pending
        FROM contributors
        ''',
        fetch_one=True
    )
    
    print(f"\n📊 Final Statistics:")
    print(f"   Total Contributors: {stats['total']}")
    print(f"   With Intelligence: {stats['with_intelligence']}")
    print(f"   Pending: {stats['pending']}")
    print(f"   Completion: {stats['with_intelligence']/stats['total']*100:.1f}%")
    
    # Export results
    export_path = '/tmp/contributor_intelligence_results.csv'
    all_contributors = db.execute_query(
        'SELECT email, contributor_id, intelligence_summary, processed_data FROM contributors',
        fetch_all=True
    )
    
    export_data = []
    for c in all_contributors:
        data = json.loads(c['processed_data'])
        export_data.append({
            'email': c['email'],
            'contributor_id': c['contributor_id'],
            'intelligence_summary': c['intelligence_summary'],
            'skills': ', '.join(data.get('skills', [])),
            'total_projects': data['activity_summary']['total_projects'],
            'total_hours': data['activity_summary']['total_hours']
        })
    
    export_df = pd.DataFrame(export_data)
    export_df.to_csv(export_path, index=False)
    
    print(f"\n✅ Results exported to: {export_path}")
    print(f"   Total records: {len(export_df)}")
    
    print("\n" + "="*80)
    print("Processing Complete!")
    print("="*80)

if __name__ == "__main__":
    main()

