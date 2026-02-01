#!/bin/bash
# Monitor longtests and compile reports when complete

CACHE_DIR="MultistateModelsTests/cache/longtest_logs"
REPORTS_DIR="MultistateModelsTests/reports"
EXPECTED_TESTS=11

echo "Monitoring longtest progress..."
echo "Expected tests: $EXPECTED_TESTS"
echo ""

while true; do
    # Count completed tests by checking for "Test Summary" in logs
    COMPLETED=$(grep -l "Test Summary" "$CACHE_DIR"/longtest_*.log 2>/dev/null | wc -l | tr -d ' ')
    PASSED=$(grep -l "Test Summary.*Pass" "$CACHE_DIR"/longtest_*.log 2>/dev/null | wc -l | tr -d ' ')
    
    echo "[$(date +%H:%M:%S)] Completed: $COMPLETED/$EXPECTED_TESTS"
    
    if [ "$COMPLETED" -ge "$EXPECTED_TESTS" ]; then
        echo ""
        echo "All tests completed!"
        echo "Passed: $PASSED / $EXPECTED_TESTS"
        echo ""
        
        # Show any failures
        for log in "$CACHE_DIR"/longtest_*.log; do
            if grep -q "Test Summary" "$log" && ! grep -q "Pass" "$log"; then
                echo "FAILED: $(basename $log)"
            fi
        done
        
        echo ""
        echo "Compiling reports..."
        cd "$REPORTS_DIR"
        quarto render 03_long_tests.qmd
        echo ""
        echo "Report compilation complete!"
        break
    fi
    
    sleep 60
done
