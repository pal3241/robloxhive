from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any


class CognitiveMemory:
    """Unified long-term memory for a single high-intelligence bot.

    Stores episodic, semantic, procedural, social, role, UI, map, failure,
    strategy, and action-outcome memories in one lightweight SQLite database.
    Retrieval is intentionally simple/fast: lexical overlap + recency +
    confidence + success weighting.
    """

    KINDS = {
        "episodic",
        "semantic",
        "procedural",
        "social",
        "team",
        "enemy",
        "role",
        "ui",
        "map",
        "strategy",
        "failure",
        "goal",
        "action",
        "research",
    }

    def __init__(self, path: str | Path = "data/memory/robloxhive.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    game_id INTEGER NOT NULL DEFAULT 0,
                    kind TEXT NOT NULL,
                    key TEXT,
                    text TEXT NOT NULL,
                    data_json TEXT NOT NULL DEFAULT '{}',
                    confidence REAL NOT NULL DEFAULT 0.5,
                    success REAL,
                    importance REAL NOT NULL DEFAULT 0.5,
                    hits INTEGER NOT NULL DEFAULT 0,
                    last_hit REAL
                );
                CREATE INDEX IF NOT EXISTS idx_mem_game_kind
                  ON memories(game_id, kind, ts DESC);
                CREATE INDEX IF NOT EXISTS idx_mem_key
                  ON memories(game_id, kind, key);

                CREATE TABLE IF NOT EXISTS relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    game_id INTEGER NOT NULL DEFAULT 0,
                    subject TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    object TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    data_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_rel_subject
                  ON relations(game_id, subject, relation);

                CREATE TABLE IF NOT EXISTS kv (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )

    @staticmethod
    def _tokens(text: str) -> set[str]:
        token = []
        out: set[str] = set()
        for ch in text.lower():
            if ch.isalnum() or ch == "_":
                token.append(ch)
            elif token:
                value = "".join(token)
                if len(value) >= 2:
                    out.add(value)
                token.clear()
        if token:
            value = "".join(token)
            if len(value) >= 2:
                out.add(value)
        return out

    def remember(
        self,
        kind: str,
        text: str,
        *,
        game_id: int = 0,
        key: str | None = None,
        data: dict[str, Any] | None = None,
        confidence: float = 0.5,
        success: bool | float | None = None,
        importance: float = 0.5,
    ) -> int:
        kind = kind.strip().lower()
        if kind not in self.KINDS:
            raise ValueError(f"Unsupported memory kind: {kind}")
        now = time.time()
        success_value = None if success is None else float(success)
        with self._connect() as db:
            cur = db.execute(
                """
                INSERT INTO memories
                  (ts, game_id, kind, key, text, data_json, confidence, success, importance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    int(game_id or 0),
                    kind,
                    key,
                    text.strip(),
                    json.dumps(data or {}, ensure_ascii=False),
                    max(0.0, min(float(confidence), 1.0)),
                    success_value,
                    max(0.0, min(float(importance), 1.0)),
                ),
            )
            return int(cur.lastrowid)

    def upsert_fact(
        self,
        kind: str,
        key: str,
        text: str,
        *,
        game_id: int = 0,
        data: dict[str, Any] | None = None,
        confidence: float = 0.7,
        importance: float = 0.6,
    ) -> int:
        kind = kind.strip().lower()
        with self._connect() as db:
            row = db.execute(
                """
                SELECT id, confidence FROM memories
                WHERE game_id=? AND kind=? AND key=?
                ORDER BY ts DESC LIMIT 1
                """,
                (int(game_id or 0), kind, key),
            ).fetchone()
            if row is not None:
                new_conf = max(float(row["confidence"]), float(confidence))
                db.execute(
                    """
                    UPDATE memories
                    SET ts=?, text=?, data_json=?, confidence=?, importance=?
                    WHERE id=?
                    """,
                    (
                        time.time(),
                        text.strip(),
                        json.dumps(data or {}, ensure_ascii=False),
                        new_conf,
                        max(0.0, min(float(importance), 1.0)),
                        int(row["id"]),
                    ),
                )
                return int(row["id"])
        return self.remember(
            kind,
            text,
            game_id=game_id,
            key=key,
            data=data,
            confidence=confidence,
            importance=importance,
        )

    def relate(
        self,
        subject: str,
        relation: str,
        object_: str,
        *,
        game_id: int = 0,
        confidence: float = 0.6,
        data: dict[str, Any] | None = None,
    ) -> int:
        with self._connect() as db:
            cur = db.execute(
                """
                INSERT INTO relations
                  (ts, game_id, subject, relation, object, confidence, data_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(),
                    int(game_id or 0),
                    subject.strip(),
                    relation.strip(),
                    object_.strip(),
                    max(0.0, min(float(confidence), 1.0)),
                    json.dumps(data or {}, ensure_ascii=False),
                ),
            )
            return int(cur.lastrowid)

    def retrieve(
        self,
        query: str,
        *,
        game_id: int = 0,
        kinds: list[str] | None = None,
        limit: int = 16,
    ) -> list[dict[str, Any]]:
        query_tokens = self._tokens(query)
        params: list[Any] = [int(game_id or 0)]
        sql = "SELECT * FROM memories WHERE game_id IN (0, ?)"
        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            sql += f" AND kind IN ({placeholders})"
            params.extend(k.strip().lower() for k in kinds)
        sql += " ORDER BY ts DESC LIMIT 500"

        now = time.time()
        scored: list[tuple[float, sqlite3.Row]] = []
        with self._connect() as db:
            rows = db.execute(sql, params).fetchall()
            for row in rows:
                text_tokens = self._tokens(str(row["text"]))
                key_tokens = self._tokens(str(row["key"] or ""))
                overlap = len(query_tokens & (text_tokens | key_tokens))
                lexical = overlap / max(1.0, math.sqrt(len(query_tokens) or 1))
                age_hours = max(0.0, (now - float(row["ts"])) / 3600.0)
                recency = 1.0 / (1.0 + age_hours / 24.0)
                confidence = float(row["confidence"])
                importance = float(row["importance"])
                success = row["success"]
                success_bonus = 0.15 if success is not None and float(success) >= 0.5 else 0.0
                score = lexical * 1.8 + confidence * 0.7 + importance * 0.55 + recency * 0.35 + success_bonus
                if query_tokens and overlap == 0:
                    score *= 0.35
                scored.append((score, row))

            scored.sort(key=lambda item: item[0], reverse=True)
            selected = scored[: max(1, min(int(limit), 64))]
            ids = [int(row["id"]) for _, row in selected]
            if ids:
                db.executemany(
                    "UPDATE memories SET hits=hits+1, last_hit=? WHERE id=?",
                    [(now, memory_id) for memory_id in ids],
                )

        return [
            {
                "id": int(row["id"]),
                "kind": row["kind"],
                "key": row["key"],
                "text": row["text"],
                "data": json.loads(row["data_json"] or "{}"),
                "confidence": float(row["confidence"]),
                "success": row["success"],
                "importance": float(row["importance"]),
                "score": round(score, 4),
                "ts": float(row["ts"]),
            }
            for score, row in selected
        ]

    def relations(
        self,
        *,
        game_id: int = 0,
        subject: str | None = None,
        relation: str | None = None,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM relations WHERE game_id IN (0, ?)"
        params: list[Any] = [int(game_id or 0)]
        if subject:
            sql += " AND lower(subject)=lower(?)"
            params.append(subject)
        if relation:
            sql += " AND lower(relation)=lower(?)"
            params.append(relation)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(max(1, min(int(limit), 200)))
        with self._connect() as db:
            rows = db.execute(sql, params).fetchall()
        return [
            {
                "subject": row["subject"],
                "relation": row["relation"],
                "object": row["object"],
                "confidence": float(row["confidence"]),
                "data": json.loads(row["data_json"] or "{}"),
                "ts": float(row["ts"]),
            }
            for row in rows
        ]

    def set_state(self, key: str, value: Any) -> None:
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO kv(key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                  value_json=excluded.value_json,
                  updated_at=excluded.updated_at
                """,
                (key, json.dumps(value, ensure_ascii=False), time.time()),
            )

    def get_state(self, key: str, default: Any = None) -> Any:
        with self._connect() as db:
            row = db.execute("SELECT value_json FROM kv WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value_json"])
        except json.JSONDecodeError:
            return default

    def stats(self) -> dict[str, Any]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT kind, COUNT(*) AS count FROM memories GROUP BY kind ORDER BY kind"
            ).fetchall()
            relation_count = db.execute("SELECT COUNT(*) AS c FROM relations").fetchone()["c"]
        return {
            "path": str(self.path),
            "total": sum(int(row["count"]) for row in rows),
            "by_kind": {row["kind"]: int(row["count"]) for row in rows},
            "relations": int(relation_count),
        }
