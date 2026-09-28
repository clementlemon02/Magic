-- Demo personas for Aurelia Financial. Applied after schema.sql on first boot.
-- Document and transaction fixtures are the ingestion workstream's (§8); this is
-- only the identities POST /query resolves a caller against.
--
-- Idempotent, so re-running it against a live database is safe. Grants are NOT
-- here: scripts/seed_demo.py derives them from each connector's own check_access,
-- for every user in this table. Add a persona here, re-run seed_demo, done.

INSERT INTO roles (name, clearance_level) VALUES
    ('support',     0),
    ('engineering', 0),
    ('operations',  0),
    ('compliance',  1)
ON CONFLICT (name) DO NOTHING;

-- Ids 1 and 2 are load-bearing: the scenarios, the evals and the integration tests
-- all address Alex and Marcus by id. Append below them, never renumber.
--
-- Alex asks the questions in Scenarios 1-3: standard clearance, support department.
-- Marcus is the compliance officer of Scenarios 4-5, and the only persona who can
-- read an explanation back out via GET /audit/{request_id}.
--
-- The rest exist so the access-gap report has a population to count. It ranks by
-- DISTINCT ASKERS, so with two personas its headline number can never exceed two.
-- Four departments, and six people who will hit the same restricted AML material:
-- Alex, Priya and Daniel from support, Wei and Sofia from engineering, and Raj in
-- operations, whose access is the narrowest of anyone's.
--
-- Hannah is a second compliance officer on purpose. She clears everything Marcus
-- does EXCEPT the private #compliance-alerts channel, whose Slack ACL is a list of
-- member ids rather than a group — so the two officers differ, and the demo can
-- show a per-channel membership grant rather than a role granting everything.
-- Ids are ASSIGNED, not left to the sequence. ON CONFLICT DO NOTHING still consumes
-- a sequence value for every row it skips, so re-running this file would otherwise
-- hand out different ids on every machine — and the scenarios, evals and integration
-- tests all address Alex and Marcus by id. Append below, never renumber.
--
-- Alex asks the questions in Scenarios 1-3: standard clearance, support department.
-- Marcus is the compliance officer of Scenarios 4-5, and the only persona who can
-- read an explanation back out via GET /audit/{request_id}.
--
-- The rest exist so the access-gap report has a population to count. It ranks by
-- DISTINCT ASKERS, so with two personas its headline number can never exceed two.
-- Four departments, and six people who hit the same restricted AML material: Alex,
-- Priya and Daniel in support, Wei and Sofia in engineering, and Raj in operations,
-- whose access is the narrowest of anyone's.
--
-- Hannah is a second compliance officer on purpose. She clears everything Marcus
-- does EXCEPT the private #compliance-alerts channel, whose Slack ACL is a list of
-- member ids rather than a group — so the two officers differ, and the demo can show
-- a per-channel membership grant instead of a role granting everything.
--
-- CAREFUL: that channel's mock ACL is member_ids [2, 9, 42]. Ids 9 and 42 are
-- deliberately orphaned — no persona holds them. Give either to a new user and they
-- silently gain restricted compliance content, which is precisely the stale-membership
-- oversharing this project exists to surface. Pick 11 and up.
-- Every persona's password is the literal word `demo`. A DEMO CREDENTIAL, written
-- here on purpose so a fresh clone can sign in; the hashes are scrypt with a fixed
-- per-user salt so this file produces the same rows on every machine. Nothing here
-- is a secret worth protecting, and nothing here should exist outside a demo.
INSERT INTO users (id, name, email, role_id, dept, password_hash) VALUES
    (1, 'Alex Tan',    'alex.tan@aurelia.example',    (SELECT id FROM roles WHERE name = 'support'),     'support', 'scrypt$AQEBAQEBAQEBAQEBAQEBAQ$TAJzOrP73_JZxAgVUaMP5igEyau3BlxrXVNGZFf6RPU'),
    (2, 'Marcus Lim',  'marcus.lim@aurelia.example',  (SELECT id FROM roles WHERE name = 'compliance'),  'compliance', 'scrypt$AgICAgICAgICAgICAgICAg$RzsFfPxjAsdLyxYwi2mBsyG8WvDkblt4FhiLwJ6hJ4Q'),
    (3, 'Priya Nair',  'priya.nair@aurelia.example',  (SELECT id FROM roles WHERE name = 'support'),     'support', 'scrypt$AwMDAwMDAwMDAwMDAwMDAw$dFC9iNTCKloRG3UZaZ4av4CpdGqhWJ1YxTt-QJfQ4TQ'),
    (4, 'Daniel Ong',  'daniel.ong@aurelia.example',  (SELECT id FROM roles WHERE name = 'support'),     'support', 'scrypt$BAQEBAQEBAQEBAQEBAQEBA$B9Mjan8Kceo-pgtkukkU3UlQNDy0wcmeBKhH7MupKkM'),
    (5, 'Wei Lin',     'wei.lin@aurelia.example',     (SELECT id FROM roles WHERE name = 'engineering'), 'engineering', 'scrypt$BQUFBQUFBQUFBQUFBQUFBQ$0hbNT9iBmDlcOsRUJcokEZ26iFEhsfq-EaKAm9uhhc4'),
    (6, 'Sofia Reyes', 'sofia.reyes@aurelia.example', (SELECT id FROM roles WHERE name = 'engineering'), 'engineering', 'scrypt$BgYGBgYGBgYGBgYGBgYGBg$Qj3QmA0bndUWrlUmRNKnplRZwpCCCvUxLdS5pF2sNC0'),
    (7, 'Hannah Koh',  'hannah.koh@aurelia.example',  (SELECT id FROM roles WHERE name = 'compliance'),  'compliance', 'scrypt$BwcHBwcHBwcHBwcHBwcHBw$-VCQPThh15fsvVGyQg1K-JbB7s4eglWMJR8gK3ort3Q'),
    (8, 'Raj Menon',   'raj.menon@aurelia.example',   (SELECT id FROM roles WHERE name = 'operations'),  'operations', 'scrypt$CAgICAgICAgICAgICAgICA$giVXvvP1MyLoexFwXPBRjUa87y9C76ryz_LWCjat3ds')
-- Fill in a hash for a row seeded before the column existed, and leave alone any
-- password someone has since set. DO NOTHING would skip existing personas entirely,
-- which is how eight identities ended up unable to sign in.
ON CONFLICT (id) DO UPDATE
    SET password_hash = EXCLUDED.password_hash
    WHERE users.password_hash IS NULL;

-- Hand the sequence back past BOTH the assigned ids and the orphaned Slack member id
-- 9, so the next auto-assigned user is 11 and cannot inherit a private channel by
-- number. GREATEST keeps a database that already has higher ids intact.
SELECT setval('users_id_seq', GREATEST((SELECT max(id) FROM users), 10), true);
