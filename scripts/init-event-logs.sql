-- Mnemodrome Event Log PostgreSQL Schema & Functions
-- Run against the event_logs database as event_log_user
--
-- Schema: event_logs
-- Tables: sessions, events
-- Functions: prune_old_events, prune_by_size

-- Create schema if not exists
CREATE SCHEMA IF NOT EXISTS event_logs;

-- Sessions table
CREATE TABLE IF NOT EXISTS event_logs.sessions (
    session_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    last_active_at TIMESTAMPTZ NOT NULL,
    status TEXT DEFAULT 'active',
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_sessions_last_active
    ON event_logs.sessions (last_active_at DESC);

-- Events table
CREATE TABLE IF NOT EXISTS event_logs.events (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    event_type TEXT NOT NULL,
    session_id TEXT NOT NULL REFERENCES event_logs.sessions(session_id),
    actor TEXT,
    action TEXT NOT NULL,
    target TEXT,
    context JSONB,
    result TEXT,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_events_session_id
    ON event_logs.events (session_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp
    ON event_logs.events (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_events_event_type
    ON event_logs.events (event_type);
CREATE INDEX IF NOT EXISTS idx_events_action
    ON event_logs.events (action);

-- Phase 2D: Age-based retention function
CREATE OR REPLACE FUNCTION event_logs.prune_old_events(retention_days integer)
RETURNS integer
LANGUAGE plpgsql
AS $$
DECLARE
    deleted_count integer;
BEGIN
    DELETE FROM event_logs.events
    WHERE timestamp < (NOW() - (retention_days || ' days')::interval);
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$;

-- Phase 2D: Size-based retention function
CREATE OR REPLACE FUNCTION event_logs.prune_by_size(max_size_mb integer)
RETURNS integer
LANGUAGE plpgsql
AS $$
DECLARE
    current_size_mb numeric;
    deleted_count integer := 0;
    batch_size integer := 100;
    batch_deleted integer;
BEGIN
    SELECT (pg_total_relation_size('event_logs.events') +
            pg_total_relation_size('event_logs.sessions')) / (1024.0 * 1024.0)
    INTO current_size_mb;

    WHILE current_size_mb > max_size_mb LOOP
        DELETE FROM event_logs.events
        WHERE id IN (
            SELECT id FROM event_logs.events
            ORDER BY timestamp ASC
            LIMIT batch_size
        );
        GET DIAGNOSTICS batch_deleted = ROW_COUNT;
        deleted_count := deleted_count + batch_deleted;

        IF batch_deleted = 0 THEN
            EXIT;
        END IF;

        SELECT (pg_total_relation_size('event_logs.events') +
                pg_total_relation_size('event_logs.sessions')) / (1024.0 * 1024.0)
        INTO current_size_mb;
    END LOOP;

    RETURN deleted_count;
END;
$$;
