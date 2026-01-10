-- TimescaleDB Initialization Script for PolyglotLink

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create IoT metrics table
CREATE TABLE IF NOT EXISTS iot_metrics (
    time TIMESTAMPTZ NOT NULL,
    device_id TEXT NOT NULL,
    metric TEXT NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Convert to hypertable
SELECT create_hypertable('iot_metrics', 'time', if_not_exists => TRUE);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_metrics_device_id ON iot_metrics (device_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_metric ON iot_metrics (metric, time DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_device_metric ON iot_metrics (device_id, metric, time DESC);

-- Create continuous aggregate for hourly rollups
CREATE MATERIALIZED VIEW IF NOT EXISTS iot_metrics_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    device_id,
    metric,
    AVG(value) AS avg_value,
    MIN(value) AS min_value,
    MAX(value) AS max_value,
    COUNT(*) AS sample_count
FROM iot_metrics
GROUP BY bucket, device_id, metric
WITH NO DATA;

-- Add retention policy (keep raw data for 7 days)
SELECT add_retention_policy('iot_metrics', INTERVAL '7 days', if_not_exists => TRUE);

-- Add compression policy (compress after 1 day)
ALTER TABLE iot_metrics SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'device_id,metric'
);
SELECT add_compression_policy('iot_metrics', INTERVAL '1 day', if_not_exists => TRUE);

-- Refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy('iot_metrics_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Create table for device registry
CREATE TABLE IF NOT EXISTS devices (
    device_id TEXT PRIMARY KEY,
    device_type TEXT,
    name TEXT,
    location TEXT,
    tags TEXT[],
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create table for schema cache
CREATE TABLE IF NOT EXISTS schema_cache (
    schema_signature TEXT PRIMARY KEY,
    field_mappings JSONB NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    source TEXT NOT NULL,
    hit_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_devices_updated_at
    BEFORE UPDATE ON devices
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_schema_cache_updated_at
    BEFORE UPDATE ON schema_cache
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
