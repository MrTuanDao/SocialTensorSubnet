#!/bin/bash

# Save the current HEAD hash
current_head=$(git rev-parse HEAD)

# Pull the latest changes from the repository
git stash
git pull -f

# Get the new HEAD hash
new_head=$(git rev-parse HEAD)

# Check if the new HEAD is different from the current HEAD
if [ "$current_head" != "$new_head" ]; then
    # The HEAD has changed, meaning there's a new version
    echo "New version detected, restarting the validator."
    pm2 restart validator_nicheimage
else
    # No new version, no action needed
    echo "No new version detected, no restart needed."
fi

# No need for a loop, as PM2 will execute this script every hour
