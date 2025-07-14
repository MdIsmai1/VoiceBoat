#!/bin/bash
set -e

if [ -z "$DATABASE_URL" ]; then
    echo "Warning: DATABASE_URL not set. Skipping database check."
    exit 0
fi

# Parse DATABASE_URL to extract host and port
# Handle both postgres:// and postgresql:// schemes
url="$DATABASE_URL"
host=$(echo "$url" | awk -F'[/:@]' '{print $(NF-2)}')
port=$(echo "$url" | awk -F'[/:@]' '{print $(NF-1)}')

if [ -z "$host" ] || [ -z "$port" ]; then
    echo "Error: Could not parse host or port from DATABASE_URL: $url"
    exit 1
fi

max_attempts=12
attempt=1

echo "Waiting for database at $host:$port..."
while [ $attempt -le $max_attempts ]; do
    if nc -z "$host" "$port"; then
        echo "Database is ready!"
        exit 0
    fi
    echo "Database not ready, retrying in 5 seconds (attempt $attempt/$max_attempts)..."
    sleep 5
    attempt=$((attempt + 1))
done

echo "Error: Database not available after $max_attempts attempts"
exit 1
