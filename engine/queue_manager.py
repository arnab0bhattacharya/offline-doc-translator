import os
import time
import uuid
import threading
import queue
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable, Dict, List, Union

from engine.core import TranslationMode
from engine.errors import TranslatorError
from engine.run_job import execute_translation


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TranslationJob:
    id: str
    input_path: str
    output_path: str
    direction: str
    mode: TranslationMode
    model_name: str
    glossary: Dict[str, str]
    status: JobStatus
    progress: float
    progress_message: str
    error: Optional[TranslatorError] = None
    error_message: Optional[str] = None
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    review_log_path: str = ""
    include_source_text: bool = False


class TranslationQueue:
    def __init__(
        self,
        on_job_update: Optional[Callable[[TranslationJob], None]] = None,
        on_log: Optional[Callable[[str, str], None]] = None
    ):
        """
        on_job_update called whenever a job's status/progress changes.
        on_log(job_id, message) called for telemetry log messages.
        """
        self.on_job_update = on_job_update
        self.on_log = on_log
        self._jobs: List[TranslationJob] = []
        self._lock = threading.Lock()
        self._queue = queue.Queue()
        
        self._shutdown_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def add_job(
        self,
        input_path: str,
        output_path: str,
        direction: str,
        mode: Union[TranslationMode, str],
        model_name: str,
        glossary: Dict[str, str],
        include_source_text: bool = False,
    ) -> str:
        """Creates a TranslationJob and adds to queue. Returns job ID."""
        job_id = str(uuid.uuid4())
        review_log_path = f"{output_path}.needs_review.log"
        mode_enum = mode if isinstance(mode, TranslationMode) else TranslationMode(mode)
        
        job = TranslationJob(
            id=job_id,
            input_path=input_path,
            output_path=output_path,
            direction=direction,
            mode=mode_enum,
            model_name=model_name,
            glossary=glossary,
            status=JobStatus.QUEUED,
            progress=0.0,
            progress_message="Queued",
            error=None,
            error_message=None,
            created_at=time.time(),
            started_at=None,
            completed_at=None,
            review_log_path=review_log_path,
            include_source_text=include_source_text,
        )

        
        with self._lock:
            self._jobs.append(job)
        
        self._trigger_update(job)
        self._queue.put(job)
        
        return job_id

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued job. Cannot cancel a running job (just removes from queue)."""
        with self._lock:
            for job in self._jobs:
                if job.id == job_id:
                    if job.status == JobStatus.QUEUED:
                        job.status = JobStatus.CANCELLED
                        self._trigger_update(job)
                        return True
                    return False
        return False

    def get_all_jobs(self) -> List[TranslationJob]:
        """Returns all jobs (queued, running, completed, failed) in order."""
        with self._lock:
            return list(self._jobs)

    def get_job(self, job_id: str) -> Optional[TranslationJob]:
        """Get a specific job by ID."""
        with self._lock:
            for job in self._jobs:
                if job.id == job_id:
                    return job
        return None

    def clear_completed(self) -> None:
        """Remove completed and failed jobs from the list."""
        with self._lock:
            self._jobs = [
                job for job in self._jobs
                if job.status not in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
            ]

    def shutdown(self) -> None:
        """Gracefully stop the worker thread."""
        self._shutdown_event.set()
        # Send a sentinel to unblock queue.get() immediately
        self._queue.put(None)
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

    def _trigger_update(self, job: TranslationJob) -> None:
        if self.on_job_update:
            try:
                self.on_job_update(job)
            except Exception:
                pass

    def _log(self, job_id: str, message: str) -> None:
        if self.on_log:
            try:
                self.on_log(job_id, message)
            except Exception:
                pass

    def _worker_loop(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
                
            if job is None:
                # Sentinel received
                break
                
            if job.status == JobStatus.CANCELLED:
                self._queue.task_done()
                continue
                
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            job.progress_message = "Starting..."
            self._trigger_update(job)
            
            try:
                self._process_job(job)
                job.status = JobStatus.COMPLETED
                job.progress = 100.0
                job.progress_message = "Completed"
            except TranslatorError as te:
                job.status = JobStatus.FAILED
                job.error = te
                job.error_message = te.title
                job.progress_message = f"Failed: {te.title}"
            except Exception as e:
                job.status = JobStatus.FAILED
                job.error = None
                job.error_message = str(e)
                job.progress_message = f"Failed: {str(e)}"
            finally:
                job.completed_at = time.time()
                self._trigger_update(job)
                self._queue.task_done()

    def _process_job(self, job: TranslationJob) -> None:
        def progress_cb(current: int, total: int, msg: str) -> None:
            pct = (current / max(1, total)) * 100.0 if total > 0 else 0.0
            job.progress = pct
            job.progress_message = msg
            self._trigger_update(job)

        def log_cb(msg: str) -> None:
            self._log(job.id, msg)

        execute_translation(
            input_path=job.input_path,
            output_path=job.output_path,
            direction=job.direction,
            mode=job.mode,
            model_name=job.model_name,
            glossary=job.glossary,
            progress_cb=progress_cb,
            log_cb=log_cb,
            include_source_text=job.include_source_text,
        )

