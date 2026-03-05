# API Module Documentation

## Overview
The API module provides a stateless FastAPI application that serves as a read-only viewer for job statuses from the Redis task queue. It is designed to facilitate easy access to the current state of jobs without directly interacting with simulation files.

## Setup Instructions
1. **Install Dependencies**: Navigate to the `api` directory and install the required Python packages listed in `requirements.txt`:
   ```
   pip install -r requirements.txt
   ```

2. **Run the FastAPI Application**: Start the FastAPI application using the following command:
   ```
   uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
   ```

3. **Access the API**: Once the application is running, you can access the API at `http://localhost:8000`. The API provides endpoints to query the status of jobs in the Redis queue.

## Usage Guidelines
- The API is designed to be stateless and only retrieves information from Redis.
- Ensure that the Redis server is running and accessible before starting the FastAPI application.
- Refer to the API documentation for details on available endpoints and their usage.

## Contributing
If you wish to contribute to the API module, please follow the project's contribution guidelines and ensure that your changes are well-documented.