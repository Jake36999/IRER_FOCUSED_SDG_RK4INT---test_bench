# Redis Setup and Usage

This document provides an overview of the Redis setup and its usage within the IRER V14 architecture.

## Overview

Redis is utilized as the "Nervous System" of the IRER V14 architecture, facilitating communication between the orchestrator and worker components. It serves as a task queue where job manifests are pushed by the orchestrator and subsequently processed by the workers.

## Installation

To set up Redis, follow these steps:

1. **Install Redis**: You can download and install Redis from the official website or use a package manager like `apt` for Ubuntu or `brew` for macOS.

   For Ubuntu:
   ```
   sudo apt update
   sudo apt install redis-server
   ```

   For macOS:
   ```
   brew install redis
   ```

2. **Start Redis Server**: After installation, start the Redis server using the following command:
   ```
   redis-server
   ```

3. **Verify Installation**: You can verify that Redis is running by using the Redis CLI:
   ```
   redis-cli ping
   ```
   You should receive a response of `PONG`.

## Configuration

Ensure that the Redis server is configured to allow connections from the orchestrator and worker components. You may need to adjust the `redis.conf` file to set the appropriate bind address and port.

## Usage

- **Job Queue**: The orchestrator will push job manifests to a Redis queue. Workers will pop these jobs for processing.
- **Status Checking**: The stateless API will query Redis for the status of jobs, ensuring that it does not directly interact with simulation files.

## Best Practices

- Monitor Redis performance and memory usage to ensure optimal operation.
- Implement appropriate error handling in the orchestrator and worker components to manage Redis connection issues.

## Conclusion

Integrating Redis into the IRER V14 architecture enhances the system's stability and scalability, allowing for efficient task management and communication between components.