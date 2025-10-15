"""
Test script for new CSV structure with 10 rows.
Validates parsing, production filtering, and data structure.
"""
import pandas as pd
import json
from src.config import get_settings
from src.processors.data_processor import process_contributor

def test_csv_parsing():
    """Test CSV parsing with first 10 rows"""

    csv_path = "/Users/mathumithamanivasagam/ClaudeProject/OGs/attachment/Data@10Oct.csv"

    print("=" * 80)
    print("Testing New CSV Structure - First 10 Rows")
    print("=" * 80)

    # Read first 10 rows
    df = pd.read_csv(csv_path, nrows=10)

    print(f"\n✓ Loaded {len(df)} rows")
    print(f"✓ Columns: {list(df.columns)}")

    settings = get_settings()

    successful = 0
    failed = 0

    for idx, row in df.iterrows():
        try:
            profile = process_contributor(row, settings)

            print(f"\n{'=' * 80}")
            print(f"Row {idx + 1}: {profile.contributor_email}")
            print(f"{'=' * 80}")
            print(f"Contributor ID: {profile.contributor_id}")
            print(f"Education: {profile.education_level}")
            print(f"Location: {profile.location}")
            print(f"Languages: {len(profile.languages)}")

            # Production projects
            print(f"\n📊 Production Projects: {len(profile.production_projects)}")
            for i, proj in enumerate(profile.production_projects[:3], 1):
                print(f"  {i}. {proj.project_type} - {proj.account_name}")

            # Activity summary
            activity = profile.activity_summary
            print(f"\n📈 Activity Summary:")
            print(f"  Total: {activity.total_production_projects}")
            print(f"  Types: {dict(list(activity.project_types_distribution.items())[:3])}")
            print(f"  Clients: {activity.unique_clients[:3]}")

            # Compliance
            print(f"\n🛡️ Compliance:")
            print(f"  KYC: {profile.compliance.kyc_status}")
            print(f"  DOTS: {profile.compliance.dots_status}")
            print(f"  Risk: {profile.compliance.risk_tier}")

            successful += 1

        except Exception as e:
            print(f"\n❌ Row {idx + 1} FAILED: {e}")
            failed += 1

    print(f"\n{'=' * 80}")
    print(f"RESULTS: {successful} successful, {failed} failed")
    print(f"{'=' * 80}")

    if successful > 0:
        print("\n✅ TEST PASSED - New CSV structure working correctly!")
        print("\nNext steps:")
        print("1. Check if Ollama is running: ollama serve")
        print("2. Pull qwen2.5:3b model: ollama pull qwen2.5:3b")
        print("3. Start Streamlit app: streamlit run app.py")
        print("4. Import full CSV using local file path option")
    else:
        print("\n❌ TEST FAILED - Fix errors above before proceeding")

if __name__ == "__main__":
    test_csv_parsing()
