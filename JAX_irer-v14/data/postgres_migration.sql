-- SQL commands to migrate from SQLite to PostgreSQL for the Knowledge Extraction Log (KEL)

-- Create the new table structure in PostgreSQL
CREATE TABLE kel (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(255) NOT NULL,
    result JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert data from the old SQLite database
INSERT INTO kel (job_id, result, created_at, updated_at)
SELECT job_id, result, created_at, updated_at FROM old_kel_table; 

-- Drop the old table if necessary
DROP TABLE IF EXISTS old_kel_table; 

-- Add any necessary indexes
CREATE INDEX idx_kel_job_id ON kel (job_id);