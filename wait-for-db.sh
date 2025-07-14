#!/bin/bash

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
    echo "Warning: DATABASE_URL is not set. Skipping database check and proceeding with deployment."
    exit 0
fi

# Parse host and port from DATABASE_URL
# Example: postgresql://user:password@host:port/dbname
host=$(echo $DATABASE_URL | grep -oP '(?<=@)[^:/]+' || echo "")
port=$(echo $DATABASE_URL | grep -oP ':[0-9]+(?=/)' | cut -d':' -f2 || echo "")

# Fallback if initial parsing fails
if [ -z "$host" ] || [ -z "$port" ]; then
    # Alternative parsing for non-standard formats
    host=$(echo $DATABASE_URL | grep -oP '(?<=@)[^/]+(?=:[0-9]+/)' || echo "")
    port=$(echo $DATABASE_URL | grep -oP ':[0-9]+(?=/)' | cut -d':' -f2 || echo "")
fi

if [ -z "$host" ] || [ -z "$port" ]; then
    echo "Warning: Could not parse host or port from DATABASE_URL: $DATABASE_URL. Skipping database check."
    exit 0
fi

echo "Waiting for database at $host:$port..."

# Retry connection up to 30 times with 2-second intervals
for i in {1..30}; do
    nc -z $host $port
    if [ $? -eq 0 ]; then
        echo "Database is ready!"
        exit 0
    fi
    echo "Database not ready, retrying in 2 seconds..."
    sleep 2
done

echo "Error: Database not available after 60 seconds."
exit 1
