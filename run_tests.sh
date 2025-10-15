#!/bin/bash
# Run all tests with coverage
cd "/Users/ashaik/Documents/Github/Cursor-summaryData/dev branch/SummaryData"
python3 -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html --tb=short --maxfail=5 2>&1 | tee test_results.txt
echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
tail -30 test_results.txt

