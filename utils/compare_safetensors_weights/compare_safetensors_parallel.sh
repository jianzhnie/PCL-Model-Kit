#!/bin/bash

# Check if arguments are provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <dir1> <dir2> <output_dir> [max_jobs]"
    echo "Example: $0 /path/to/folder1 /path/to/folder2 ./logs 4"
    exit 1
fi

DIR1="$1"
DIR2="$2"
OUTPUT_DIR="$3"
MAX_JOBS="${4:-4}" # Default to 4 parallel jobs if not specified

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/compare_safetensors_single.py"

echo "Starting parallel comparison..."
echo "Source Dir: $DIR1"
echo "Target Dir: $DIR2"
echo "Output Dir: $OUTPUT_DIR"
echo "Max Jobs:   $MAX_JOBS"
echo "----------------------------------------"

# Loop from 1 to 61
for i in {1..61}; do
    # Construct filenames based on the user provided patterns
    # Folder 1: model-1-of-61.safetensors
    FILE1="${DIR1}/model-${i}-of-61.safetensors"

    # Folder 2: model-00001-of-000061.safetensors (zero padded to 5 digits)
    PADDED_I=$(printf "%05d" "$i")
    FILE2="${DIR2}/model-${PADDED_I}-of-000061.safetensors"

    OUTPUT_FILE="${OUTPUT_DIR}/compare_result_${i}.txt"

    # Check if input files exist before running
    if [ ! -f "$FILE1" ]; then
        echo "Warning: Source file not found: $FILE1" > "$OUTPUT_FILE"
        continue
    fi

    if [ ! -f "$FILE2" ]; then
        echo "Warning: Target file not found: $FILE2" > "$OUTPUT_FILE"
        continue
    fi

    echo "Running comparison for index $i (Background)..."

    # Run python script in background
    python3 "$PYTHON_SCRIPT" --source "$FILE1" --target "$FILE2" > "$OUTPUT_FILE" 2>&1 &

    # Job control: Wait if we have reached MAX_JOBS
    while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 1
    done
done

# Wait for all background jobs to finish
wait

echo "----------------------------------------"
echo "All comparisons completed. Results saved in $OUTPUT_DIR"
