#!/bin/bash

if [ -z "$DATABASE_URL" ]; then
    echo "Error: DATABASE_URL is not set."
    exit 1
fi

host=$(echo $DATABASE_URL | grep -oP '(?<=@)[^/]+' | cut -d':' -f1)
port=$(echo $DATABASE_URL | grep -oP '(?<=:)[0-9]+(?=/)')

if [ -z "$host" ] || [ -z "$port" ]; then
    echo "Error: Could not parse host or port from DATABASE_URL."
    exit 1
fi

echo "Waiting for database at $host:$port..."

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
