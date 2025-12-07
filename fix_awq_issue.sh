#!/bin/bash
# Fix AutoAWQ compatibility issue for fine-tuning
# 
# Problem: PEFT tries to import AutoAWQ code even when using base models,
# causing import errors due to AutoAWQ deprecation.
#
# Solution: Temporarily uninstall autoawq for fine-tuning, reinstall for inference.

set -e

echo "=========================================="
echo "Fixing AutoAWQ Compatibility Issue"
echo "=========================================="
echo ""
echo "Issue: PEFT tries to use AutoAWQ code even for base models,"
echo "       causing import errors with deprecated AutoAWQ."
echo ""
echo "Solution: Uninstall autoawq for fine-tuning"
echo "          (You can reinstall it later for inference)"
echo ""

# Check if autoawq is installed
if pip show autoawq > /dev/null 2>&1; then
    echo "Found autoawq installed. Uninstalling..."
    pip uninstall autoawq -y
    echo "✅ autoawq uninstalled"
    echo ""
    echo "You can now run fine-tuning:"
    echo "  ./run_qwen32b_awq.sh"
    echo ""
    echo "To reinstall autoawq later for inference:"
    echo "  pip install autoawq>=0.1.8"
else
    echo "autoawq is not installed - no action needed."
    echo "You should be able to run fine-tuning now."
fi

echo ""

