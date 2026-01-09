CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    condition TEXT NOT NULL,
    prompt TEXT NOT NULL,
    candidate_code TEXT NOT NULL,
    passed INTEGER NOT NULL,
    test_log TEXT NOT NULL,
    created_at TEXT NOT NULL,
    code_hash TEXT NOT NULL,
    run_id INTEGER,
    prompt_hash TEXT,
    retrieval_used INTEGER DEFAULT 0,
    memory_sources TEXT,
    memory_top_score REAL
);

CREATE TABLE IF NOT EXISTS failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    condition TEXT NOT NULL,
    prompt TEXT NOT NULL,
    candidate_code TEXT NOT NULL,
    passed INTEGER NOT NULL,
    test_log TEXT NOT NULL,
    created_at TEXT NOT NULL,
    code_hash TEXT NOT NULL,
    run_id INTEGER,
    prompt_hash TEXT,
    retrieval_used INTEGER DEFAULT 0,
    memory_sources TEXT,
    memory_top_score REAL
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    mode TEXT NOT NULL,
    backend TEXT NOT NULL,
    model TEXT NOT NULL,
    adapter_name TEXT,
    memory_enabled INTEGER NOT NULL,
    semantic_enabled INTEGER NOT NULL,
    learning_enabled INTEGER NOT NULL,
    k INTEGER NOT NULL,
    max_tokens INTEGER NOT NULL,
    temperature REAL NOT NULL,
    top_p REAL NOT NULL,
    top_k INTEGER NOT NULL,
    notes TEXT,
    config_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS schema_meta (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    schema_version INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);

INSERT OR IGNORE INTO schema_meta (singleton, schema_version, updated_at)
VALUES (1, 2, datetime('now'));

CREATE TABLE IF NOT EXISTS semantic_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL,
    rule_text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    origin_episode_ids TEXT NOT NULL,
    evidence_count INTEGER NOT NULL,
    eval_snapshot TEXT,
    active INTEGER NOT NULL,
    superseded_by INTEGER,
    last_verified_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS procedures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern TEXT NOT NULL,
    recipe_text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    origin_episode_ids TEXT NOT NULL,
    evidence_count INTEGER NOT NULL,
    eval_snapshot TEXT,
    active INTEGER NOT NULL,
    superseded_by INTEGER,
    last_verified_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS adapters (
    name TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    notes TEXT,
    active INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS promotions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    previous_adapter_name TEXT,
    candidate_adapter_name TEXT,
    decision TEXT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    heldout_size INTEGER NOT NULL,
    k INTEGER NOT NULL,
    backend TEXT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS train_progress (
    task_id TEXT PRIMARY KEY,
    first_seen_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS regression_tasks (
    task_id TEXT PRIMARY KEY,
    rank INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS policies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    parent_policy_name TEXT,
    policy_json TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS active_policy (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    policy_name TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS policy_promotions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    previous_policy_name TEXT,
    candidate_policy_name TEXT,
    decision TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS verification_cache (
    key TEXT PRIMARY KEY,
    passed INTEGER NOT NULL,
    log TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    prompt, candidate_code, test_log, content='episodes', content_rowid='id'
);
CREATE VIRTUAL TABLE IF NOT EXISTS rules_fts USING fts5(
    rule_text, content='semantic_rules', content_rowid='id'
);
CREATE VIRTUAL TABLE IF NOT EXISTS procedures_fts USING fts5(
    recipe_text, content='procedures', content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
    INSERT INTO episodes_fts(rowid, prompt, candidate_code, test_log)
    VALUES (new.id, new.prompt, new.candidate_code, new.test_log);
END;

CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, prompt, candidate_code, test_log)
    VALUES ('delete', old.id, old.prompt, old.candidate_code, old.test_log);
END;

CREATE TRIGGER IF NOT EXISTS rules_ai AFTER INSERT ON semantic_rules BEGIN
    INSERT INTO rules_fts(rowid, rule_text) VALUES (new.id, new.rule_text);
END;

CREATE TRIGGER IF NOT EXISTS rules_ad AFTER DELETE ON semantic_rules BEGIN
    INSERT INTO rules_fts(rules_fts, rowid, rule_text) VALUES ('delete', old.id, old.rule_text);
END;

CREATE TRIGGER IF NOT EXISTS procedures_ai AFTER INSERT ON procedures BEGIN
    INSERT INTO procedures_fts(rowid, recipe_text) VALUES (new.id, new.recipe_text);
END;

CREATE TRIGGER IF NOT EXISTS procedures_ad AFTER DELETE ON procedures BEGIN
    INSERT INTO procedures_fts(procedures_fts, rowid, recipe_text) VALUES ('delete', old.id, old.recipe_text);
END;
