#!/bin/bash
# Zeely AI Avatar Pipeline — Quick Start
# ========================================

set -e

echo "🔧 Checking environment..."

# Check Python
python3 --version 2>/dev/null || { echo "❌ Python 3 required"; exit 1; }

# Check API keys
if [ -z "$FAL_KEY" ]; then
    echo "❌ FAL_KEY not set. Get yours at https://fal.ai/dashboard/keys"
    echo "   Run: export FAL_KEY='your-key-here'"
    exit 1
fi

if [ -z "$REPLICATE_API_TOKEN" ]; then
    echo "❌ REPLICATE_API_TOKEN not set. Get yours at https://replicate.com/account/api-tokens"
    echo "   Run: export REPLICATE_API_TOKEN='your-token-here'"
    exit 1
fi

echo "✓ API keys detected"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt --quiet

# Check input files
USER_COUNT=$(find input/users -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.webp" \) 2>/dev/null | wc -l)
OUTFIT_COUNT=$(find input/outfits -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.webp" \) 2>/dev/null | wc -l)

echo "📸 Found $USER_COUNT user photo(s) in input/users/"
echo "👔 Found $OUTFIT_COUNT outfit reference(s) in input/outfits/"

if [ "$USER_COUNT" -eq 0 ]; then
    echo "❌ No user photos found. Place photos in input/users/"
    exit 1
fi

if [ "$OUTFIT_COUNT" -eq 0 ]; then
    echo "❌ No outfit references found. Place garment images in input/outfits/"
    exit 1
fi

# Run pipeline
echo ""
echo "🚀 Starting pipeline..."
echo "================================"
python3 pipeline.py "$@"
echo ""
echo "✅ Done! Check the output/ directory for results."
