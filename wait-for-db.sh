#!/bin/bash

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
    echo "Warning: DATABASE_URL is not set. Skipping database check and proceeding with deployment."
    exit 0
fi

# Parse host and port from DATABASE_URL
# Example: postgresql://user:password@host:port/dbname or postgresql://user:password@host/dbname
host=$(echo $DATABASE_URL | grep -oP '(?<=@)[a-zA-Z0-9.-]+(?=[:/])' || echo "")
port=$(echo $DATABASE_URL | grep -oP ':[0-9]+(?=/)' | cut -d':' -f2 || echo "5432")

# Fallback for Render-specific URLs (e.g., dpg-xxx.oregon-postgres.render.com)
if [ -z "$host" ]; then
    host=$(echo $DATABASE_URL | grep -oP '(?<=@)[a-zA-Z0-9.-]+(?=/)' || echo "")
fi

if [ -z "$host" ]; then
    echo "Warning: Could not parse host from DATABASE_URL: $DATABASE_URL. Skipping database check."
    exit 0
fi

if [ -z "$port" ]; then
    port="5432"
    echo "Warning: Could not parse port from DATABASE_URL: $DATABASE_URL. Using default port 5432."
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
