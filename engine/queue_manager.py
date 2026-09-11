import os
import time
import uuid
import threading
import queue
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List, Union, Any

from engine.core import TranslationMode
from engine.cache import CachePolicy
from engine.errors import ErrorCode, TranslatorError
from engine.run_job import execute_translation


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def format_eta(eta_seconds: Optional[float]) -> str:
    """Formats estimated time remaining in seconds into a human-readable string."""
    if eta_seconds is None or eta_seconds <= 0:
        return ""
    remaining = int(round(eta_seconds))
    if remaining < 120:
        return f"~{remaining}s left"
    elif remaining < 3600:
        return f"~{int(round(remaining / 60))}m left"
    else:
        hrs = remaining // 3600
        mins = int(round((remaining % 3600) / 60))
        if mins == 60:
            hrs += 1
            mins = 0
        if mins > 0:
            return f"~{hrs}h {mins}m left"
        return f"~{hrs}h left"


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
    cancel_event: threading.Event = field(default_factory=threading.Event)
    cache_policy: CachePolicy = CachePolicy.ENCRYPTED_PERSISTENT
    result: Optional[Dict[str, Any]] = None
    eta_seconds: Optional[float] = None

    @property
    def eta_str(self) -> str:
        if self.eta_seconds is not None:
            return format_eta(self.eta_seconds)
        if self.status == JobStatus.RUNNING and self.started_at and self.progress > 0 and self.progress < 100.0:
            elapsed = max(0.0, time.time() - self.started_at)
            rem = elapsed * (100.0 - self.progress) / self.progress
            return format_eta(rem)
        return ""


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
        cache_policy: Union[CachePolicy, str] = CachePolicy.ENCRYPTED_PERSISTENT,
    ) -> str:
        """Creates a TranslationJob and adds to queue. Returns job ID."""
        job_id = str(uuid.uuid4())
        review_log_path = f"{output_path}.needs_review.log"
        mode_enum = mode if isinstance(mode, TranslationMode) else TranslationMode(mode)
        if not isinstance(cache_policy, CachePolicy):
            try:
                cache_policy_enum = CachePolicy(cache_policy)
            except (ValueError, TypeError):
                cache_policy_enum = CachePolicy.ENCRYPTED_PERSISTENT
        else:
            cache_policy_enum = cache_policy
        
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
            cache_policy=cache_policy_enum,
        )

        
        with self._lock:
            self._jobs.append(job)
        
        self._trigger_update(job)
        self._queue.put(job)
        
        return job_id

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued or running job."""
        with self._lock:
            for job in self._jobs:
                if job.id == job_id:
                    if job.status == JobStatus.QUEUED:
                        job.status = JobStatus.CANCELLED
                        job.progress_message = "Cancelled"
                        self._trigger_update(job)
                        return True
                    elif job.status == JobStatus.RUNNING:
                        job.cancel_event.set()
                        job.progress_message = "Cancelling..."
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
                if job.cancel_event.is_set():
                    job.status = JobStatus.CANCELLED
                    job.progress_message = "Cancelled"
                    job.error_message = "Cancelled by user"
                    job.eta_seconds = None
                else:
                    job.status = JobStatus.COMPLETED
                    job.progress = 100.0
                    job.progress_message = "Completed"
                    job.eta_seconds = 0.0
            except TranslatorError as te:
                job.eta_seconds = None
                if te.code == ErrorCode.E09 or job.cancel_event.is_set():
                    job.status = JobStatus.CANCELLED
                    job.error = te
                    job.error_message = te.title
                    job.progress_message = "Cancelled"
                else:
                    job.status = JobStatus.FAILED
                    job.error = te
                    job.error_message = te.title
                    job.progress_message = f"Failed: {te.title}"
            except Exception as e:
                job.eta_seconds = None
                if job.cancel_event.is_set():
                    job.status = JobStatus.CANCELLED
                    job.error = TranslatorError(ErrorCode.E09, detail=str(e))
                    job.error_message = "Cancelled by user"
                    job.progress_message = "Cancelled"
                else:
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
            if job.started_at and pct > 0:
                elapsed = max(0.0, time.time() - job.started_at)
                if pct < 100.0:
                    job.eta_seconds = elapsed * (100.0 - pct) / pct
                else:
                    job.eta_seconds = 0.0
            else:
                job.eta_seconds = None
            self._trigger_update(job)

        def log_cb(msg: str) -> None:
            self._log(job.id, msg)

        stats = execute_translation(
            input_path=job.input_path,
            output_path=job.output_path,
            direction=job.direction,
            mode=job.mode,
            model_name=job.model_name,
            glossary=job.glossary,
            progress_cb=progress_cb,
            log_cb=log_cb,
            include_source_text=job.include_source_text,
            cancel_event=job.cancel_event,
            cache_policy=job.cache_policy,
        )
        job.result = stats
        return stats

