"""
gui/controllers/translation_controller.py
=========================================
Bridges UI events with the background TranslationQueue.
"""

import os
from collections.abc import Callable

from engine.cache import CachePolicy
from engine.core import TranslationMode
from engine.queue_manager import TranslationJob, TranslationQueue


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
        return dispatched

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
