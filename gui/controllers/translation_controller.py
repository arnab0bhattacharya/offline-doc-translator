"""
gui/controllers/translation_controller.py
=========================================
Bridges UI events with the background TranslationQueue.
"""

import os
import time
from collections.abc import Callable
from typing import Any

from engine.cache import CachePolicy
from engine.core import TranslationMode
from engine.queue_manager import JobStatus, TranslationJob, TranslationQueue


class TranslationController:
    """
    Manages translation queue interactions, job submissions, and callback routing.
    """

    def __init__(
        self,
        on_job_update: Callable[[TranslationJob], None] | None = None,
        on_log: Callable[[str, str], None] | None = None,
    ):
        self.on_job_update = on_job_update
        self.on_log = on_log
        self._current_batch_ids: set[str] = set()
        self._batch_start_time: float | None = None

        self.queue = TranslationQueue(
            on_job_update=self._handle_job_update,
            on_log=self._handle_log,
        )

    def _handle_job_update(self, job: TranslationJob) -> None:
        if self.on_job_update:
            try:
                self.on_job_update(job)
            except Exception:
                pass

    def _handle_log(self, job_id: str, message: str) -> None:
        if self.on_log:
            try:
                self.on_log(job_id, message)
            except Exception:
                pass

    def add_job(
        self,
        input_path: str,
        output_path: str,
        direction: str,
        mode: TranslationMode | str,
        model_name: str,
        glossary: dict[str, str],
        include_source_text: bool = False,
        cache_policy: CachePolicy | str = CachePolicy.ENCRYPTED_PERSISTENT,
    ) -> str:
        """Enqueues a single translation job."""
        return self.queue.add_job(
            input_path=input_path,
            output_path=output_path,
            direction=direction,
            mode=mode,
            model_name=model_name,
            glossary=glossary,
            include_source_text=include_source_text,
            cache_policy=cache_policy,
        )

    def start_batch(
        self,
        input_files: list[str],
        direction: str,
        mode: TranslationMode | str,
        model_name: str,
        glossary: dict[str, str],
        include_source_text: bool = False,
        cache_policy: CachePolicy | str = CachePolicy.ENCRYPTED_PERSISTENT,
    ) -> list[tuple[str, str, str]]:
        """
        Calculates output paths and adds multiple documents to the queue.
        Returns list of (job_id, input_path, output_path).
        """
        if not self._current_batch_ids or self.is_batch_complete():
            self._current_batch_ids = set()
            self._batch_start_time = time.time()

        dispatched: list[tuple[str, str, str]] = []
        for input_path in input_files:
            if not os.path.exists(input_path):
                continue
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_{direction}{ext}"

            job_id = self.add_job(
                input_path=input_path,
                output_path=output_path,
                direction=direction,
                mode=mode,
                model_name=model_name,
                glossary=glossary,
                include_source_text=include_source_text,
                cache_policy=cache_policy,
            )
            dispatched.append((job_id, input_path, output_path))
            self._current_batch_ids.add(job_id)
        return dispatched

    def get_current_batch_ids(self) -> set[str]:
        """Returns the set of job IDs in the current active batch."""
        return set(self._current_batch_ids)

    def is_batch_complete(self, batch_ids: set[str] | None = None) -> bool:
        """Returns True if all jobs in the batch have reached a terminal status."""
        ids = batch_ids if batch_ids is not None else self._current_batch_ids
        if not ids:
            return False
        terminal = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
        for jid in ids:
            job = self.get_job(jid)
            if job is None or job.status not in terminal:
                return False
        return True

    def get_batch_summary(self, batch_ids: set[str] | None = None) -> dict[str, Any]:
        """
        Calculates aggregate statistics across all jobs in the given or current batch.
        """
        ids = batch_ids if batch_ids is not None else self._current_batch_ids
        jobs = [self.get_job(jid) for jid in ids]
        valid_jobs = [j for j in jobs if j is not None]

        completed = [j for j in valid_jobs if j.status == JobStatus.COMPLETED]
        failed = [j for j in valid_jobs if j.status == JobStatus.FAILED]
        cancelled = [j for j in valid_jobs if j.status == JobStatus.CANCELLED]

        total_translated = sum((j.result or {}).get("translated", 0) for j in completed)
        total_reverted = sum((j.result or {}).get("reverted", 0) for j in completed)
        total_skipped = sum((j.result or {}).get("skipped", 0) for j in completed)
        total_chunks = sum((j.result or {}).get("total", 0) for j in completed)

        start_times = [j.started_at for j in valid_jobs if j.started_at is not None]
        end_times = [j.completed_at for j in valid_jobs if j.completed_at is not None]

        if start_times and end_times:
            elapsed = max(0.0, max(end_times) - min(start_times))
        elif self._batch_start_time is not None:
            elapsed = max(0.0, time.time() - self._batch_start_time)
        else:
            elapsed = 0.0

        secs = max(0, round(elapsed))
        m, s = divmod(secs, 60)
        formatted_time = f"{m}m {s:02d}s" if m > 0 else f"{s}s"

        review_logs = [
            j.review_log_path
            for j in completed
            if j.review_log_path and os.path.exists(j.review_log_path) and os.path.getsize(j.review_log_path) > 0
        ]

        output_paths = [j.output_path for j in completed if j.output_path and os.path.exists(j.output_path)]
        common_dir = ""
        if output_paths:
            try:
                if len(output_paths) == 1:
                    common_dir = os.path.dirname(os.path.abspath(output_paths[0]))
                else:
                    cd = os.path.commonpath([os.path.abspath(p) for p in output_paths])
                    common_dir = os.path.dirname(cd) if os.path.isfile(cd) else cd
            except Exception:
                common_dir = os.path.dirname(os.path.abspath(output_paths[0]))

        files_line = f"Files: {len(completed)} completed, {len(failed)} failed"
        if cancelled:
            files_line += f", {len(cancelled)} cancelled"

        chunk_parts = [f"{total_translated} translated"]
        if total_reverted > 0:
            chunk_parts.append(f"{total_reverted} reverted")
        if total_skipped > 0:
            chunk_parts.append(f"{total_skipped} skipped")
        chunks_line = f"Chunks: {', '.join(chunk_parts)}"

        rev_line = (
            f"Review logs: {len(review_logs)} file{'s' if len(review_logs) != 1 else ''} {'have' if len(review_logs) != 1 else 'has'} items to review"
            if review_logs
            else "Review logs: None (clean translation)"
        )

        formatted_text = f"📊 Batch Summary\n{files_line}\n{chunks_line}\nTotal time: {formatted_time}\n{rev_line}"

        return {
            "total_files": len(valid_jobs),
            "completed_count": len(completed),
            "failed_count": len(failed),
            "cancelled_count": len(cancelled),
            "translated_chunks": total_translated,
            "reverted_chunks": total_reverted,
            "skipped_chunks": total_skipped,
            "total_chunks": total_chunks,
            "elapsed_seconds": elapsed,
            "formatted_time": formatted_time,
            "review_log_paths": review_logs,
            "output_paths": output_paths,
            "common_dir": common_dir,
            "formatted_text": formatted_text,
            "jobs": valid_jobs,
        }

    def cancel_job(self, job_id: str) -> bool:
        """Cancels a queued or running job."""
        return self.queue.cancel_job(job_id)

    def clear_completed(self) -> None:
        """Removes completed and failed jobs from queue history."""
        self.queue.clear_completed()

    def get_job(self, job_id: str) -> TranslationJob | None:
        """Returns a job by ID."""
        return self.queue.get_job(job_id)

    def get_all_jobs(self) -> list[TranslationJob]:
        """Returns all jobs in order."""
        return self.queue.get_all_jobs()

    def shutdown(self) -> None:
        """Stops worker thread gracefully."""
        self.queue.shutdown()
