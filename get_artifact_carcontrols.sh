#!/bin/bash
REPO_OWNER="SEAME-pt"
REPO_NAME="HotWheels-Cluster"
ARTIFACT_NAME="aarch64-car-controls"

# GitHub Personal Access Token (replace with your token)
GITHUB_TOKEN=""

# Function to check repository existence
check_repository() {
    local response=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
        "https://api.github.com/repos/$REPO_OWNER/$REPO_NAME")
    
    if echo "$response" | grep -q "Not Found"; then
        echo "Error: Repository $REPO_OWNER/$REPO_NAME does not exist or is not accessible."
        exit 1
    fi
}

# Function to get latest workflow run ID
get_latest_run_id() {
    local runs=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
        "https://api.github.com/repos/$REPO_OWNER/$REPO_NAME/actions/runs?per_page=1")
    
    # Extract the latest run ID
    echo "$runs" | jq -r '.workflow_runs[0].id // empty'
    #echo "$runs" | jq -r '.workflow_runs[] | select(.name == "aarch64-car-controls") | .id'
}

# Function to fetch artifacts for a specific run ID
fetch_artifacts() {
    local run_id="$1"
    
    # Fetch artifacts for the specific run
    local artifacts=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
        "https://api.github.com/repos/$REPO_OWNER/$REPO_NAME/actions/runs/$run_id/artifacts")
    
    # Find the specific artifact URL
    local artifact_url=$(echo "$artifacts" | \
        jq -r ".artifacts[] | select(.name == \"$ARTIFACT_NAME\").archive_download_url // empty")
    
    # Check if artifact URL was found
    if [ -z "$artifact_url" ]; then
        echo "No artifact found with name: $ARTIFACT_NAME"
        exit 1
    fi
    
    echo "$artifact_url"
}

# Function to download and extract the artifact
download_and_extract_artifact() {
    local artifact_url="$1"
    
    echo "Downloading artifact from URL: $artifact_url"
    
    # Download the artifact as a zip file
    local zip_file="/tmp/$ARTIFACT_NAME.zip"
    wget -q --header="Authorization: token $GITHUB_TOKEN" "$artifact_url" -O "$zip_file"
    
    # Check download success
    if [ $? -eq 0 ]; then
        echo "Artifact downloaded successfully."
        
        # Check if the directory already exists
        if [ -f "/home/tpereira/teste/aarch64-car-controls" ]; then
            echo "Removing existing aarch64-car-controls file in /home/tpereira/teste..."
            rm -f /home/tpereira/teste/aarch64-car-controls  # using -f to suppress prompt
        fi
        
        # Unzip the downloaded artifact into the target directory
        unzip -qo "$zip_file" -d /home/tpereira/teste
        
        # Clean up the zip file
        rm "$zip_file"
        
        echo "Artifact extracted to /home/tpereira/teste/"
    else
        echo "Failed to download artifact"
        exit 1
    fi
}



# Main script execution
main() {
    # Check repository existence
    check_repository
    
    # Get latest run ID
    local latest_run=$(get_latest_run_id)
    echo "Latest run: "
    echo $latest_run
    
    # Check if we got a valid run ID
    if [ -z "$latest_run" ]; then
        echo "No workflow runs found in the repository."
        exit 1
    fi
    
    # Fetch and download the artifact
    local artifact_url=$(fetch_artifacts "$latest_run")
    download_and_extract_artifact "$artifact_url"
}

# Run the main function
main