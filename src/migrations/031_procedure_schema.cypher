// Feature 031: Procedural Memory Layer - Neo4j Schema Extension
// Idempotent migration - safe to re-run
// Run via: cat migrations/031_procedure_schema.cypher | cypher-shell -u neo4j -p $NEO4J_PASSWORD

// Unique constraint on Procedure UUID
CREATE CONSTRAINT procedure_uuid IF NOT EXISTS
FOR (p:Procedure) REQUIRE p.uuid IS UNIQUE;

// Index on problem_type for filtering queries
CREATE INDEX procedure_problem_type IF NOT EXISTS
FOR (p:Procedure) ON (p.problem_type);

// Index on outcome for filtering queries
CREATE INDEX procedure_outcome IF NOT EXISTS
FOR (p:Procedure) ON (p.outcome);

// Index on created_at for temporal queries
CREATE INDEX procedure_created IF NOT EXISTS
FOR (p:Procedure) ON (p.created_at);

// Verification: Show all node types and counts
MATCH (n) RETURN labels(n) AS label, COUNT(n) AS count ORDER BY count DESC;
