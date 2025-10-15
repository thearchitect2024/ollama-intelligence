"""
Contributor Intelligence Platform - Streamlit UI
Production-grade application for contributor profile management and intelligence extraction.
"""
import streamlit as st
import pandas as pd
import logging
import time
from datetime import datetime
import os

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import application modules
from src.config import get_settings
from src.database.connection import DatabaseManager
from src.database.repositories import ContributorRepository
from src.database.migrations import create_schema
from src.processors.data_processor import process_csv_batch
from src.intelligence.llm_client import OllamaClient

# Page configuration
st.set_page_config(
    page_title="Contributor Intelligence Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize settings and database (cached)
@st.cache_resource
def init_application():
    """Initialize application components."""
    try:
        settings = get_settings()
        db_manager = DatabaseManager(settings)
        repository = ContributorRepository(db_manager)

        # Ensure schema exists
        create_schema(db_manager)

        logger.info("Application initialized successfully")
        return settings, db_manager, repository
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        st.error(f"❌ Initialization failed: {e}")
        st.stop()

settings, db_manager, repo = init_application()

# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["📤 Upload & Process", "👤 View Profiles", "🧠 Intelligence Extraction", "🔍 Search"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Contributor Intelligence Platform v2.0**\n\n"
    "Production-grade system with PostgreSQL, "
    "90-day activity tracking, and AI-powered skill extraction."
)

# =============================================================================
# PAGE 1: UPLOAD & PROCESS (Updated for large files and new CSV structure)
# =============================================================================
if page == "📤 Upload & Process":
    st.title("📤 Import Contributors from CSV")
    st.write("Process CSV file with new structure (Production projects only)")

    # Option 1: Small file upload
    st.subheader("Option 1: Upload Small File")
    uploaded_file = st.file_uploader("Upload CSV (< 200MB)", type=['csv'])

    # Option 2: Local file path (for large files)
    st.subheader("Option 2: Large Local File")
    local_path = st.text_input(
        "Enter full file path:",
        placeholder="/Users/.../attachment/Data@10Oct.csv",
        help="For 1.8GB files - will be processed in chunks"
    )

    # Option 3: S3 (placeholder for future)
    st.subheader("Option 3: S3 Location (Coming Soon)")
    s3_path = st.text_input(
        "S3 URI:",
        placeholder="s3://bucket-name/Data@10Oct.csv",
        disabled=True,
        help="S3 support will be added in next version"
    )

    # Process button
    if st.button("🚀 Import CSV Data", type="primary"):
        # Determine source
        if uploaded_file:
            source_type = "uploaded"
            file_source = uploaded_file
        elif local_path:
            source_type = "local"
            file_source = local_path
        else:
            st.error("⚠️ Please provide a file source (upload or local path)")
            st.stop()

        # Chunked processing
        chunksize = 500
        progress_bar = st.progress(0)
        status_text = st.empty()
        start_time = time.time()

        processed_count = 0
        failed_count = 0
        total_estimate = 37894  # Update dynamically if possible

        try:
            # Get chunk iterator
            if source_type == "uploaded":
                chunk_iterator = pd.read_csv(file_source, chunksize=chunksize)
            else:  # local
                chunk_iterator = pd.read_csv(file_source, chunksize=chunksize)

            # Process chunks
            for chunk_num, chunk_df in enumerate(chunk_iterator):
                for idx, row in chunk_df.iterrows():
                    try:
                        from src.processors.data_processor import process_contributor
                        profile = process_contributor(row, settings)
                        repo.upsert_contributor(profile)
                        processed_count += 1
                    except Exception as e:
                        failed_count += 1
                        logger.error(f"Failed to process row {idx}: {e}")

                # Update progress
                progress = min(processed_count / total_estimate, 1.0)
                elapsed = time.time() - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                eta_seconds = (total_estimate - processed_count) / rate if rate > 0 else 0

                progress_bar.progress(progress)
                status_text.text(
                    f"Processed: {processed_count:,} | "
                    f"Failed: {failed_count} | "
                    f"Speed: {rate:.0f} rows/sec | "
                    f"ETA: {eta_seconds/60:.1f} min"
                )

            elapsed_total = time.time() - start_time
            st.success(
                f"✅ Import complete! "
                f"Processed: {processed_count:,} | "
                f"Failed: {failed_count} | "
                f"Time: {elapsed_total/60:.1f} minutes"
            )
            st.info("💡 Go to 'Intelligence Extraction' page to generate summaries")

        except Exception as e:
            st.error(f"❌ Import failed: {e}")
            logger.error(f"Import error: {e}")

# =============================================================================
# PAGE 2: VIEW PROFILES
# =============================================================================
elif page == "👤 View Profiles":
    st.title("👤 View Contributor Profiles")

    emails = repo.get_all_emails()

    if not emails:
        st.warning("⚠️ No contributors found. Please import CSV first.")
    else:
        selected_email = st.selectbox("🔍 Select Contributor Email", emails)

        if selected_email:
            contributor = repo.get_by_email(selected_email)

            if contributor:
                data = contributor['processed_data']

                st.header(f"📧 {selected_email}")
                st.caption(f"ID: {data['contributor_id']}")

                # Education & Activity Summary
                st.subheader("🎓 Education & Activity")
                col1, col2 = st.columns(2)
                col1.metric("Education Level", data.get('education_level', 'N/A'))
                col2.metric("Production Projects", data['activity_summary']['total_production_projects'])

                # Project Types Distribution
                st.subheader("📊 Project Types Distribution")
                type_dist = data['activity_summary']['project_types_distribution']
                if type_dist:
                    df_types = pd.DataFrame([
                        {"Type": k, "Count": v}
                        for k, v in sorted(type_dist.items(), key=lambda x: x[1], reverse=True)
                    ])
                    st.bar_chart(df_types.set_index("Type"))
                else:
                    st.info("No project types available")

                # Clients Worked With
                st.subheader("🏢 Clients Worked With")
                clients = data['activity_summary']['unique_clients']
                if clients:
                    st.write(", ".join(clients))
                else:
                    st.info("No client information available")

                # Languages & Location
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("🌍 Languages")
                    languages = [f"{l['language']} ({l['proficiency']})" if l['proficiency'] else l['language']
                                 for l in data['languages']]
                    st.write(", ".join(languages) if languages else "None")

                with col2:
                    st.subheader("📍 Location")
                    loc = data['location']
                    if loc.get('us_state'):
                        st.write(f"{loc['us_state']}, {loc['country']}")
                    else:
                        st.write(loc['country'])

                # Compliance
                st.subheader("🛡️ Compliance")
                comp = data['compliance']
                col1, col2, col3 = st.columns(3)
                col1.metric("KYC Status", comp['kyc_status'])
                col2.metric("DOTS Status", comp['dots_status'])
                col3.metric("Risk Tier", comp['risk_tier'])

                # ALL Production Projects
                st.subheader("🎯 Production Projects")
                if data['production_projects']:
                    st.write(f"**Total: {len(data['production_projects'])} projects**")

                    # Show in expandable sections
                    for i, proj in enumerate(data['production_projects'], 1):
                        with st.expander(f"{i}. {proj['project_type']} - {proj.get('account_name', 'N/A')}"):
                            st.write(f"**Project ID:** {proj['project_id']}")
                            st.write(f"**Account:** {proj.get('account_name', 'N/A')}")
                            st.write(f"**Type:** {proj['project_type']}")
                            desc = proj.get('long_desc', '')
                            if desc:
                                st.write(f"**Description:** {desc[:500]}...")
                            else:
                                st.write("**Description:** N/A")
                else:
                    st.info("No production projects")

                # Extracted Skills
                extracted_skills = data.get('extracted_skills', [])
                if extracted_skills:
                    st.subheader("🎯 Extracted Skills")
                    # Display skills as tags/chips
                    skills_html = " ".join([f'<span style="background-color: #e0f2fe; padding: 4px 12px; border-radius: 16px; margin: 4px; display: inline-block; font-size: 14px; color: #0369a1;">{skill}</span>' for skill in extracted_skills])
                    st.markdown(skills_html, unsafe_allow_html=True)
                elif data.get('production_projects'):
                    st.info("ℹ️ Skills not extracted (no project descriptions available)")

                # Intelligence Summary (read from JSONB)
                intelligence_summary = data.get('intelligence_summary')
                if intelligence_summary:
                    st.subheader("🧠 Intelligence Summary")
                    st.info(intelligence_summary)
                else:
                    st.warning("⚠️ Intelligence summary not yet generated")

                # Full JSON (for debugging)
                with st.expander("🔍 View Full JSON"):
                    st.json(data)

# =============================================================================
# PAGE 3: INTELLIGENCE EXTRACTION
# =============================================================================
elif page == "🧠 Intelligence Extraction":
    st.title("🧠 Intelligence Extraction")

    stats = repo.get_statistics()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Contributors", stats.get('total', 0) or 0)
    col2.metric("With Intelligence", stats.get('with_intelligence', 0) or 0)
    col3.metric("Remaining", (stats.get('total', 0) or 0) - (stats.get('with_intelligence', 0) or 0))

    st.markdown("---")

    st.warning("⚠️ **Important**: Stay on this page during extraction. Switching tabs will stop the process.")

    if st.button("🚀 Extract Intelligence (Async Batch Processing)", type="primary"):
        try:
            import asyncio
            from src.models import ContributorProfile
            from src.intelligence.skill_extractor import generate_intelligence_summary

            # Start timing
            start_time = time.time()

            # Get contributors
            all_contributors = repo.get_all_contributors()

            # Filter those without intelligence
            contributors_to_process = [
                c for c in all_contributors
                if not c.get('intelligence_summary')
            ]

            if not contributors_to_process:
                st.info("✅ All contributors already have intelligence summaries!")
                st.stop()

            st.info(f"🚀 Processing {len(contributors_to_process)} contributors with async batching ({settings.max_concurrent_llm} concurrent)...")

            # Initialize LLM client
            llm_client = OllamaClient(settings)

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Define async batch processor
            async def process_all_batches():
                processed = 0
                failed = 0
                batch_size = settings.max_concurrent_llm  # 5 concurrent

                # Process in batches
                for batch_start in range(0, len(contributors_to_process), batch_size):
                    batch_end = min(batch_start + batch_size, len(contributors_to_process))
                    batch = contributors_to_process[batch_start:batch_end]

                    # Convert to profiles
                    profiles = []
                    for contrib in batch:
                        try:
                            profile = ContributorProfile(**contrib['processed_data'])
                            profiles.append(profile)
                        except Exception as e:
                            logger.error(f"Failed to parse profile: {e}")
                            failed += 1

                    # Import utility functions for smart skip logic
                    from src.intelligence.skill_extractor import (
                        has_project_descriptions,
                        generate_summary_no_projects,
                        generate_summary_no_descriptions,
                        parse_llm_response
                    )

                    # SMART SKIP LOGIC: Separate profiles by description availability
                    profiles_with_descriptions = []
                    profiles_without_descriptions = []

                    for profile in profiles:
                        # Check if profile has project descriptions
                        if not profile.production_projects:
                            # Case 1: No projects - skip LLM
                            summary = generate_summary_no_projects(profile)
                            profile.extracted_skills = []
                            profiles_without_descriptions.append((profile, summary))

                        elif not has_project_descriptions(profile):
                            # Case 2: Projects but no descriptions - skip LLM
                            summary = generate_summary_no_descriptions(profile)
                            profile.extracted_skills = []
                            profiles_without_descriptions.append((profile, summary))

                        else:
                            # Case 3: Has descriptions - will call LLM
                            profiles_with_descriptions.append(profile)

                    # Build prompts ONLY for profiles WITH descriptions
                    prompts = []
                    for profile in profiles_with_descriptions:
                        # Use the NEW prompt format from skill_extractor
                        from src.intelligence.skill_extractor import generate_intelligence_summary
                        # Note: We'll call generate_intelligence_summary which uses the new format
                        prompts.append(profile)  # Keep profile reference for now

                    # Generate summaries using LLM for profiles WITH descriptions (TRUE BATCH ASYNC)
                    if prompts:
                        # Build all prompts
                        prompt_texts = []
                        for profile in prompts:
                            prompt_text = f"""You MUST output in this EXACT format:

SUMMARY:
[Your 90-120 word paragraph here]

SKILLS:
- Skill 1
- Skill 2
- Skill 3
(List 5-10 skills total)

===== CONTRIBUTOR DATA =====

Location: {str(profile.location)}
Languages: {', '.join([lang.language for lang in profile.languages[:3]]) if profile.languages else 'Not specified'}
Education: {profile.education_level if profile.education_level else 'Not specified'}
Production Projects: {profile.activity_summary.total_production_projects}
Project Types: {', '.join([f"{t[0]} ({t[1]})" for t in sorted(profile.activity_summary.project_types_distribution.items(), key=lambda x: x[1], reverse=True)[:3]]) if profile.activity_summary.project_types_distribution else 'None'}
Notable Clients: {', '.join(profile.activity_summary.unique_clients[:5]) if profile.activity_summary.unique_clients else 'None'}
Compliance: KYC: {profile.compliance.kyc_status}, DOTS: {profile.compliance.dots_status}, Risk Tier: {profile.compliance.risk_tier}

Project Descriptions:
{chr(10).join([f"{i}. [{proj.project_type}] {proj.long_desc[:400] if proj.long_desc else 'No description'}" for i, proj in enumerate(profile.production_projects[:10], 1)])}

===== INSTRUCTIONS =====

SUMMARY: Write 90-120 words covering location, languages, education, projects, clients, compliance. DO NOT describe skills here.

SKILLS: Extract 5-10 TECHNICAL skills from project descriptions. List with "-" prefix.

CRITICAL RULES:
1. ONLY extract skills that are EXPLICITLY MENTIONED or CLEARLY DESCRIBED as actual work tasks
2. Extract the SPECIFIC ACTIVITY or TOOL NAMED in the description
3. DO NOT infer, assume, or generalize - if it's not directly stated, don't extract it

What TO extract:
- Specific work tasks explicitly described (e.g., "data labeling", "search evaluation", "content moderation", "validation", "tagging")
- Tools/software specifically named as used (e.g., "ADAP Tool", "Canva")
- Programming languages or frameworks directly mentioned
- Technical activities clearly stated (e.g., "write instruction commands", "review LLM responses", "audio recording")

What NOT to extract:
- Inferred technical domains (e.g., don't assume "Machine Learning" or "NLP" unless those exact words describe the work)
- Job requirements or qualifications (e.g., "Fluent in English", "Creative Thinker", "Attention to Detail", "3 years experience")
- Soft/management skills (e.g., "Project Management", "Client Interaction", "Communication", "Leadership")
- Marketing/compensation language (e.g., "High-Earning Potential", "Public Contribution", "Flexible Hours")
- System requirements (e.g., "Chrome browser", "Windows", "MacOS")
- General abilities (e.g., "Fast learner", "Detail-oriented", "Adaptable")

CRITICAL: You MUST include both "SUMMARY:" and "SKILLS:" headers."""
                            prompt_texts.append(prompt_text)

                        # Call LLM in TRUE BATCH MODE (concurrent processing)
                        try:
                            llm_responses = await llm_client.generate_batch(prompt_texts, max_concurrent=batch_size)
                        except Exception as e:
                            logger.error(f"Batch LLM generation failed: {e}")
                            llm_responses = [f"Error: {e}"] * len(prompt_texts)

                        # Parse LLM responses and extract skills
                        for profile, llm_response in zip(profiles_with_descriptions, llm_responses):
                            try:
                                if not llm_response.startswith("Error:"):
                                    # Parse to extract summary and skills
                                    summary_text, skills_list = parse_llm_response(llm_response)
                                    profile.extracted_skills = skills_list

                                    # Store in profile.intelligence_summary (will go to JSONB)
                                    if skills_list:
                                        profile.intelligence_summary = f"{summary_text}\n\nSkills: {', '.join(skills_list)}"
                                    else:
                                        profile.intelligence_summary = summary_text
                                else:
                                    profile.intelligence_summary = llm_response
                                    profile.extracted_skills = []

                                # ONLY update JSONB (which includes intelligence_summary + extracted_skills)
                                repo.upsert_contributor(profile)
                                processed += 1

                            except Exception as e:
                                logger.error(f"Failed to process {profile.contributor_email}: {e}")
                                failed += 1

                    # Update database for profiles WITHOUT descriptions (fast path)
                    for profile, summary in profiles_without_descriptions:
                        try:
                            # Store summary in profile object
                            profile.intelligence_summary = summary
                            # profile.extracted_skills already set to [] above

                            # ONLY update JSONB
                            repo.upsert_contributor(profile)
                            processed += 1
                        except Exception as e:
                            logger.error(f"Failed to update DB for {profile.contributor_email}: {e}")
                            failed += 1

                    # Update progress
                    total_processed = batch_end
                    progress = total_processed / len(contributors_to_process)
                    elapsed = time.time() - start_time
                    rate = total_processed / elapsed if elapsed > 0 else 0
                    eta_seconds = (len(contributors_to_process) - total_processed) / rate if rate > 0 else 0

                    progress_bar.progress(progress)
                    status_text.text(
                        f"Processed: {processed}/{len(contributors_to_process)} | "
                        f"Failed: {failed} | "
                        f"Speed: {rate:.1f} profiles/sec | "
                        f"ETA: {eta_seconds/60:.1f} min"
                    )

                # Return counts for outer scope
                return processed, failed

            # Run async processing
            processed, failed = asyncio.run(process_all_batches())

            # Calculate duration
            elapsed_total = time.time() - start_time
            if elapsed_total < 60:
                duration_str = f"{elapsed_total:.1f}s"
            elif elapsed_total < 3600:
                minutes = int(elapsed_total // 60)
                seconds = int(elapsed_total % 60)
                duration_str = f"{minutes}m {seconds}s"
            else:
                hours = int(elapsed_total // 3600)
                minutes = int((elapsed_total % 3600) // 60)
                duration_str = f"{hours}h {minutes}m"

            st.success(
                f"✅ Extraction complete! "
                f"Successful: {processed} | "
                f"Failed: {failed} | "
                f"Time: {duration_str}"
            )

        except Exception as e:
            st.error(f"❌ Extraction failed: {e}")
            logger.error(f"Intelligence extraction error: {e}")

# =============================================================================
# PAGE 4: SEARCH
# =============================================================================
elif page == "🔍 Search":
    st.title("🔍 Search Contributors")

    search_term = st.text_input("Enter email to search")

    if search_term:
        results = repo.search_by_email(search_term)

        st.write(f"**Found {len(results)} result(s)**")

        for result in results:
            with st.expander(f"📧 {result['email']}"):
                data = result['processed_data']
                activity = data['activity_summary']

                col1, col2 = st.columns(2)
                col1.write(f"**Status:** {'✅ Active' if activity['is_active_90d'] else '❌ Inactive'}")
                col2.write(f"**Weekly Hours:** {activity['weekly_hours_avg']:.1f}")

                if result['intelligence_summary']:
                    st.info(result['intelligence_summary'])
