from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4

from robloxhive.brain.memory import GameMemory
from robloxhive.brain.research import InternetResearcher


@dataclass(slots=True)
class LearningJob:
    id: str
    game_id: int
    game_name: str
    objective: str
    status: str = "queued"
    stage: str = "queued"
    completed: int = 0
    total: int = 1
    error: str | None = None
    research_file: str | None = None
    created_at: str = ""
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LearningManager:
    def __init__(
        self,
        memory: GameMemory,
        researcher: InternetResearcher | None = None,
        workers: int = 2,
    ) -> None:
        self.memory = memory
        self.researcher = researcher or InternetResearcher()
        self._jobs: dict[str, LearningJob] = {}
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="robloxhive-learn")

    def start(self, game_id: int, game_name: str, objective: str) -> LearningJob:
        job = LearningJob(
            id=uuid4().hex[:12],
            game_id=game_id,
            game_name=game_name.strip(),
            objective=objective.strip() or "learn from beginner to completion and collect tips and tricks",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._jobs[job.id] = job
        self._executor.submit(self._run, job.id)
        return job

    def _progress(self, job_id: str, stage: str, completed: int, total: int) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.stage = stage
            job.completed = completed
            job.total = max(total, 1)

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            game_id = job.game_id
            game_name = job.game_name
            objective = job.objective

        try:
            bundle = self.researcher.research(
                game_name=game_name,
                objective=objective,
                progress=lambda stage, completed, total: self._progress(
                    job_id, stage, completed, total
                ),
            )
            path = self.memory.save_research(game_id, game_name, bundle)
            with self._lock:
                job = self._jobs[job_id]
                job.status = "complete"
                job.stage = "saved"
                job.completed = job.total
                job.research_file = str(path)
                job.finished_at = datetime.now(timezone.utc).isoformat()
        except Exception as exc:
            with self._lock:
                job = self._jobs[job_id]
                job.status = "failed"
                job.stage = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.finished_at = datetime.now(timezone.utc).isoformat()

    def get(self, job_id: str) -> LearningJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [job.to_dict() for job in reversed(list(self._jobs.values()))]
