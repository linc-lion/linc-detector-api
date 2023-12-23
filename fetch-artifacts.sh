#!/bin/bash

JSON_FILE="src/artifact/artifacts.json"

# Check if the JSON file doesn't exist
if [ ! -f "$JSON_FILE" ]; then
    echo "JSON file 'artifacts.json' not found."
    exit 1  # Exit with a non-zero status to indicate an error
fi

# Read the JSON file
JSON=$(cat "$JSON_FILE")

# Extract values using jq
MODEL_NAME=$(echo "$JSON" | jq -r '.name')
MODEL_VERSION=$(echo "$JSON" | jq -r '.version')

MODEL_PATH="s3://linc-model-artifact/$MODEL_NAME/$MODEL_VERSION/model.pth"

LOCAL_DIR="artifacts/$MODEL_NAME/$MODEL_VERSION/"
echo "Downloading artifact from $MODEL_PATH to $LOCAL_DIR"

aws s3 cp $MODEL_PATH "$LOCAL_DIR"
