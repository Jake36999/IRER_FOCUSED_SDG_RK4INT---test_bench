# Orchestrator Module

The Orchestrator module is responsible for managing tasks and job manifests within the IRER V14 architecture. It interfaces with Redis to push and manage job queues, enabling a decoupled and efficient microservices architecture.

## Setup Instructions

1. **Install Dependencies**: Ensure you have the required Python packages installed. You can do this by running:
   ```
   pip install -r requirements.txt
   ```

2. **Redis Setup**: Make sure you have Redis installed and running. The orchestrator will communicate with Redis to manage job queues.

3. **Configuration**: Update any necessary configuration settings in the `orchestrator.py` file to match your environment.

## Usage Guidelines

- The orchestrator can be started by running the `orchestrator.py` script. This will initialize the connection to Redis and begin managing job manifests.
- Ensure that workers are properly configured to listen for job manifests pushed by the orchestrator.
- Monitor the Redis queue to track job statuses and worker activity.

## Contributing

If you would like to contribute to the Orchestrator module, please follow the project's contribution guidelines and ensure that your changes are well-documented.