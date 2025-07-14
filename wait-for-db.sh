#!/bin/bash
# wait-for-db.sh
set -e

if [ -z "$DATABASE_URL" ]; then
    echo "Error: DATABASE_URL not set"
    exit 1
fi

host=$(echo $DATABASE_URL | sed -E 's/postgres:\/\/[^:]+:[^@]+@([^:\/]+).*/\1/')
port=$(echo $DATABASE_URL | sed -E 's/postgres:\/\/[^:]+:[^@]+@[^:]+:([0-9]+).*/\1/')

echo "Waiting for database at $host:$port..."
until nc -z $host $port; do
    echo "Database not ready, retrying in 5 seconds..."
    sleep 5
done
echo "Database is ready!"
