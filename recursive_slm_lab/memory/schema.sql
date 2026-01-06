CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    condition TEXT NOT NULL,
    prompt TEXT NOT NULL,
    candidate_code TEXT NOT NULL,
    passed INTEGER NOT NULL,
    test_log TEXT NOT NULL,
    created_at TEXT NOT NULL,
    code_hash TEXT NOT NULL
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
    code_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS semantic_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE NOT NULL,
    rule_text TEXT NOT NULL,
    evidence_count INTEGER NOT NULL,
    last_verified_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS procedures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern TEXT UNIQUE NOT NULL,
    recipe_text TEXT NOT NULL,
    evidence_count INTEGER NOT NULL,
    last_verified_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS adapters (
    name TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    notes TEXT,
    active INTEGER NOT NULL
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
