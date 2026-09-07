import os
import time
import uuid
import threading
import queue
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable, Dict, List, Union

from engine.errors import TranslatorError, ErrorCode
from engine.core import TranslationEngine, TranslationMode
from engine.preflight import run_preflight, run_nmt_preflight

from formats.pptx_handler import PPTXHandler
from formats.xlsx_handler import XLSXHandler
from formats.docx_handler import DOCXHandler
from formats.pdf_handler import PDFHandler


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
    error: Optional[str]
    created_at: float
    started_at: Optional[float]
    completed_at: Optional[float]
    review_log_path: str


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
        glossary: Dict[str, str]
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
            created_at=time.time(),
            started_at=None,
            completed_at=None,
            review_log_path=review_log_path
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
            except Exception as e:
                job.status = JobStatus.FAILED
                job.error = str(e)
                job.progress_message = f"Failed: {str(e)}"
            finally:
                job.completed_at = time.time()
                self._trigger_update(job)
                self._queue.task_done()

    def _process_job(self, job: TranslationJob) -> None:
        llm_available = job.mode == TranslationMode.PURE_LLM
        
        # 1. Preflight
        run_preflight(
            model_name=job.model_name,
            input_path=job.input_path,
            output_path=job.output_path,
            check_model=(job.mode == TranslationMode.PURE_LLM),
            require_ollama=(job.mode == TranslationMode.PURE_LLM)
        )
        if job.mode == TranslationMode.FAST_NMT:
            run_nmt_preflight(job.direction)

        # 2. Setup Engine
        engine = TranslationEngine(
            model_name=job.model_name,
            mode=job.mode,
            glossary=job.glossary,
            cache_file=os.path.join(os.path.dirname(os.path.abspath(job.output_path)), ".translation_cache.json"),
            allow_llm=llm_available
        )
        engine.load_cache(job.direction)

        # 3. Create format handler
        ext = os.path.splitext(job.input_path)[1].lower()
        if ext == ".pptx":
            handler = PPTXHandler(engine)
        elif ext == ".xlsx":
            handler = XLSXHandler(engine)
        elif ext == ".docx":
            handler = DOCXHandler(engine)
        elif ext == ".pdf":
            handler = PDFHandler(engine)
        else:
            raise TranslatorError(ErrorCode.E04, detail=f"Unsupported extension '{ext}'")

        # 4. Progress/Log Callbacks
        def progress_cb(current: int, total: int, msg: str) -> None:
            pct = (current / max(1, total)) * 100.0 if total > 0 else 0.0
            job.progress = pct
            job.progress_message = msg
            self._trigger_update(job)

        def log_cb(msg: str) -> None:
            self._log(job.id, msg)

        # Clear old review log if it exists
        if os.path.exists(job.review_log_path):
            try:
                os.remove(job.review_log_path)
            except OSError:
                pass

        # 5. Translate
        # For mock injection during testing, we can check if it's monkey-patched, 
        # but Python handles monkeypatching transparently.
        handler.translate(
            input_path=job.input_path,
            output_path=job.output_path,
            direction=job.direction,
            review_log_path=job.review_log_path,
            progress_cb=progress_cb,
            log_cb=log_cb
        )
