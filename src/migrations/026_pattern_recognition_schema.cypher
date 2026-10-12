// Feature 026: Cross-Session Pattern Recognition - Neo4j Schema Extension
// Idempotent migration - safe to re-run
// Run via: cat migrations/026_pattern_recognition_schema.cypher | cypher-shell -u neo4j -p $NEO4J_PASSWORD

// Unique constraint on Pattern UUID
CREATE CONSTRAINT pattern_uuid IF NOT EXISTS
FOR (p:Pattern) REQUIRE p.uuid IS UNIQUE;

// Index on pattern_type for fast filtering by problem category
CREATE INDEX pattern_type IF NOT EXISTS
FOR (p:Pattern) ON (p.pattern_type);

// Index on success_rate for finding high-success patterns
CREATE INDEX pattern_success_rate IF NOT EXISTS
FOR (p:Pattern) ON (p.success_rate);

// Index on occurrence_count for ranking by frequency
CREATE INDEX pattern_occurrence_count IF NOT EXISTS
FOR (p:Pattern) ON (p.occurrence_count);

// Index on created_at for temporal queries and recent pattern lookups
CREATE INDEX pattern_created_at IF NOT EXISTS
FOR (p:Pattern) ON (p.created_at);

// Index on archived for excluding archived patterns efficiently
CREATE INDEX pattern_archived IF NOT EXISTS
FOR (p:Pattern) ON (p.archived);

// Index on latest_occurrence_date for recency-based ranking
CREATE INDEX pattern_latest_occurrence IF NOT EXISTS
FOR (p:Pattern) ON (p.latest_occurrence_date);

// Verification: Show all node types and counts
MATCH (n) RETURN labels(n) AS label, COUNT(n) AS count ORDER BY count DESC;
