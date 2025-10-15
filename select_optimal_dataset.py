"""
Dataset Selection Script for 4-Hour Processing Window
Selects 3,200 contributors prioritizing those with job_description in production projects.

Priority:
1. Contributors with objective.job_description in production projects (TIER 1)
2. Contributors with project_long_description only in production projects (TIER 2)
3. Contributors with no descriptions (TIER 3/4) - for diversity

Target: 3,200 contributors total
"""
import pandas as pd
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_contributor_descriptions(row):
    """
    Analyze contributor's projects to determine description tier.

    Returns:
        tuple: (tier, has_production_projects, job_desc_count, proj_desc_count)
        tier: 1 (has job_description), 2 (has project_long_description only), 3 (no descriptions)
    """
    try:
        projects_json = row.get('projects_json', '[]')
        if pd.isna(projects_json) or not projects_json:
            return (4, False, 0, 0)  # Tier 4: No projects at all

        projects = json.loads(projects_json)
        if not isinstance(projects, list):
            return (4, False, 0, 0)

        # Filter production projects (case-insensitive)
        production_projects = [p for p in projects if p.get('project_status', '').lower() == 'production']

        if not production_projects:
            return (4, False, 0, 0)  # Tier 4: No production projects

        # Count description types in production projects
        job_desc_count = 0
        proj_desc_count = 0

        for proj in production_projects:
            # Check for job_description in objective
            objective = proj.get('objective', {})
            if isinstance(objective, dict):
                job_desc = objective.get('job_description', '')
                if job_desc and job_desc.strip():
                    job_desc_count += 1

            # Check for project_long_description
            proj_long_desc = proj.get('project_long_description', '')
            if proj_long_desc and proj_long_desc.strip():
                proj_desc_count += 1

        # Determine tier
        if job_desc_count > 0:
            return (1, True, job_desc_count, proj_desc_count)  # TIER 1: Has job_description
        elif proj_desc_count > 0:
            return (2, True, job_desc_count, proj_desc_count)  # TIER 2: Has project_long_description only
        else:
            return (3, True, job_desc_count, proj_desc_count)  # TIER 3: Production projects but no descriptions

    except Exception as e:
        logger.error(f"Error analyzing contributor: {e}")
        return (4, False, 0, 0)


def main():
    """Main dataset selection logic."""
    try:
        # Read CSV
        csv_path = Path("/Users/mathumithamanivasagam/ClaudeProject/OGs/attachment/Data@10Oct_clean.csv")
        logger.info(f"Reading CSV from: {csv_path}")

        df = pd.read_csv(csv_path)
        total_rows = len(df)
        logger.info(f"Loaded {total_rows:,} contributors from CSV")

        # Analyze all contributors
        logger.info("Analyzing contributor description tiers...")
        analysis_results = []

        for idx, row in df.iterrows():
            tier, has_prod, job_count, proj_count = analyze_contributor_descriptions(row)
            analysis_results.append({
                'index': idx,
                'email': row.get('email', ''),
                'contributor_id': row.get('contributor_id', ''),
                'tier': tier,
                'has_production': has_prod,
                'job_desc_count': job_count,
                'proj_desc_count': proj_count
            })

            if (idx + 1) % 5000 == 0:
                logger.info(f"Processed {idx + 1:,} / {total_rows:,} contributors...")

        # Convert to DataFrame for easier filtering
        analysis_df = pd.DataFrame(analysis_results)

        # Count by tier
        tier_counts = analysis_df['tier'].value_counts().sort_index()
        logger.info(f"\n{'='*60}")
        logger.info("TIER DISTRIBUTION IN FULL DATASET:")
        logger.info(f"{'='*60}")
        logger.info(f"TIER 1 (job_description):           {tier_counts.get(1, 0):,} ({tier_counts.get(1, 0)/total_rows*100:.1f}%)")
        logger.info(f"TIER 2 (project_long_description):  {tier_counts.get(2, 0):,} ({tier_counts.get(2, 0)/total_rows*100:.1f}%)")
        logger.info(f"TIER 3 (no descriptions):           {tier_counts.get(3, 0):,} ({tier_counts.get(3, 0)/total_rows*100:.1f}%)")
        logger.info(f"TIER 4 (no production projects):    {tier_counts.get(4, 0):,} ({tier_counts.get(4, 0)/total_rows*100:.1f}%)")
        logger.info(f"{'='*60}\n")

        # Select optimal 6,000 contributors
        TARGET_SIZE = 6000

        # Get tier groups
        tier1 = analysis_df[analysis_df['tier'] == 1]
        tier2 = analysis_df[analysis_df['tier'] == 2]
        tier3 = analysis_df[analysis_df['tier'] == 3]
        tier4 = analysis_df[analysis_df['tier'] == 4]

        # Selection strategy - Majority from TIER 1 & 2, only ~20 from TIER 3 & 4
        logger.info(f"Selecting optimal {TARGET_SIZE:,} contributors...")

        selected_indices = []

        # Priority 1: Take TIER 3 & 4 first (small sample for diversity - ~20 total)
        tier3_target = min(len(tier3), 10)  # Up to 10 from TIER 3
        if tier3_target > 0:
            selected_tier3 = tier3.sample(n=tier3_target, random_state=42)
            selected_indices.extend(selected_tier3['index'].tolist())
            logger.info(f"Selected {tier3_target:,} from TIER 3 (no descriptions)")

        tier4_target = min(len(tier4), 10)  # Up to 10 from TIER 4
        if tier4_target > 0:
            selected_tier4 = tier4.sample(n=tier4_target, random_state=42)
            selected_indices.extend(selected_tier4['index'].tolist())
            logger.info(f"Selected {tier4_target:,} from TIER 4 (no production projects)")

        remaining = TARGET_SIZE - len(selected_indices)

        # Priority 2: Fill with TIER 1 (job_description) - take as many as possible
        tier1_target = min(len(tier1), remaining)
        if tier1_target > 0:
            selected_tier1 = tier1.sample(n=tier1_target, random_state=42)
            selected_indices.extend(selected_tier1['index'].tolist())
            logger.info(f"Selected {tier1_target:,} from TIER 1 (job_description)")

        remaining = TARGET_SIZE - len(selected_indices)

        # Priority 3: Fill remaining with TIER 2 (project_long_description)
        if remaining > 0:
            tier2_target = min(len(tier2), remaining)
            if tier2_target > 0:
                selected_tier2 = tier2.sample(n=tier2_target, random_state=42)
                selected_indices.extend(selected_tier2['index'].tolist())
                logger.info(f"Selected {tier2_target:,} from TIER 2 (project_long_description)")

        # Extract selected rows from original dataframe
        selected_df = df.iloc[selected_indices].copy()

        logger.info(f"\n{'='*60}")
        logger.info(f"FINAL SELECTION: {len(selected_df):,} contributors")
        logger.info(f"{'='*60}")

        # Verify composition
        selected_analysis = analysis_df[analysis_df['index'].isin(selected_indices)]
        selected_tier_counts = selected_analysis['tier'].value_counts().sort_index()

        logger.info(f"TIER 1 (job_description):           {selected_tier_counts.get(1, 0):,} ({selected_tier_counts.get(1, 0)/len(selected_df)*100:.1f}%)")
        logger.info(f"TIER 2 (project_long_description):  {selected_tier_counts.get(2, 0):,} ({selected_tier_counts.get(2, 0)/len(selected_df)*100:.1f}%)")
        logger.info(f"TIER 3 (no descriptions):           {selected_tier_counts.get(3, 0):,} ({selected_tier_counts.get(3, 0)/len(selected_df)*100:.1f}%)")
        logger.info(f"TIER 4 (no production projects):    {selected_tier_counts.get(4, 0):,} ({selected_tier_counts.get(4, 0)/len(selected_df)*100:.1f}%)")
        logger.info(f"{'='*60}\n")

        # Save to new CSV
        output_path = Path(f"/Users/mathumithamanivasagam/ClaudeProject/OGs/attachment/dataset_{TARGET_SIZE}_optimal.csv")
        selected_df.to_csv(output_path, index=False)
        logger.info(f"✓ Saved optimal dataset to: {output_path}")

        # Save analysis report
        report_path = Path(f"/Users/mathumithamanivasagam/ClaudeProject/OGs/attachment/dataset_{TARGET_SIZE}_analysis.json")
        report = {
            "total_contributors": len(selected_df),
            "tier_distribution": {
                "tier1_job_description": int(selected_tier_counts.get(1, 0)),
                "tier2_project_long_description": int(selected_tier_counts.get(2, 0)),
                "tier3_no_descriptions": int(selected_tier_counts.get(3, 0)),
                "tier4_no_production": int(selected_tier_counts.get(4, 0))
            },
            "estimated_processing_time": {
                "rate_per_minute": 14.3,
                "estimated_minutes": round(TARGET_SIZE / 14.3, 1),
                "estimated_hours": round(TARGET_SIZE / 14.3 / 60, 2)
            },
            "source_file": str(csv_path),
            "output_file": str(output_path),
            "selection_strategy": "Priority: TIER 1 (job_description) > TIER 2 (project_long_description) > TIER 3 (no descriptions) > TIER 4 (no production)"
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"✓ Saved analysis report to: {report_path}")

        logger.info(f"\n{'='*60}")
        logger.info("PROCESSING ESTIMATES:")
        logger.info(f"{'='*60}")
        logger.info(f"Contributors: {len(selected_df):,}")
        logger.info(f"Rate: 14.3 profiles/minute")
        logger.info(f"Estimated time: {TARGET_SIZE/14.3/60:.2f} hours (~{int(TARGET_SIZE/14.3)} minutes)")
        logger.info(f"{'='*60}\n")

        logger.info("✓ Dataset selection complete!")
        logger.info(f"Next step: Use 'dataset_{TARGET_SIZE}_optimal.csv' for processing")

    except Exception as e:
        logger.error(f"Dataset selection failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
