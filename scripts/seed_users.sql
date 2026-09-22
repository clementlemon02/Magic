-- Demo personas for Aurelia Financial. Applied after schema.sql on first boot.
-- Document and transaction fixtures are the ingestion workstream's (§8); this is
-- only the identities POST /query resolves a caller against.
--
-- Idempotent, so re-running against a live database is safe.

INSERT INTO roles (name, clearance_level) VALUES
    ('support',     0),
    ('engineering', 0),
    ('compliance',  1)
ON CONFLICT (name) DO NOTHING;

-- Alex asks the questions in Scenarios 1-3: standard clearance, support department.
-- Marcus is the compliance officer of Scenarios 4-5, and the only persona who can
-- read an explanation back out via GET /audit/{request_id}.
INSERT INTO users (name, email, role_id, dept) VALUES
    ('Alex Tan',   'alex.tan@aurelia.example',   (SELECT id FROM roles WHERE name = 'support'),    'support'),
    ('Marcus Lim', 'marcus.lim@aurelia.example', (SELECT id FROM roles WHERE name = 'compliance'), 'compliance')
ON CONFLICT (email) DO NOTHING;
