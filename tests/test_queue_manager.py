import os
import sys
import time
import pytest
from unittest.mock import patch, MagicMock

from engine.queue_manager import TranslationQueue, JobStatus, TranslationJob
from engine.errors import ErrorCode, TranslatorError
from engine.core import TranslationMode
from engine.cache import CachePolicy

@pytest.fixture
def queue_mgr():
    qm = TranslationQueue()
    yield qm
    qm.shutdown()

@patch('engine.queue_manager.execute_translation')
def test_job_creation_and_processing(mock_execute, queue_mgr):
    # Setup mocks
    def fake_execute(input_path, output_path, direction, mode, model_name, glossary, progress_cb=None, log_cb=None, **kwargs):
        if progress_cb:
            progress_cb(1, 2, "Halfway")
            time.sleep(0.1)
            progress_cb(2, 2, "Done")
        return {"total": 2, "translated": 2, "reverted": 0, "skipped": 0}

    mock_execute.side_effect = fake_execute


    
    updates = []
    def on_update(job: TranslationJob):
        updates.append((job.status, job.progress))
        
    queue_mgr.on_job_update = on_update
    
    job_id = queue_mgr.add_job(
        input_path="test.docx",
        output_path="test_en.docx",
        direction="ja2en",
        mode="fast_nmt",
        model_name="test-model",
        glossary={}
    )
    
    assert job_id is not None
    job = queue_mgr.get_job(job_id)
    assert job is not None
    assert job.status in [JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.COMPLETED]
    
    # Wait for completion
    timeout = time.time() + 2.0
    while job.status not in (JobStatus.COMPLETED, JobStatus.FAILED) and time.time() < timeout:
        time.sleep(0.05)
        
    assert job.status == JobStatus.COMPLETED
    assert job.progress == 100.0
    assert job.progress_message == "Completed"
    assert job.result == {"total": 2, "translated": 2, "reverted": 0, "skipped": 0}
    
    # Verify updates were sent
    assert len(updates) > 0
    statuses = [u[0] for u in updates]
    assert JobStatus.QUEUED in statuses
    assert JobStatus.RUNNING in statuses
    assert JobStatus.COMPLETED in statuses

def test_cancel_queued_job(queue_mgr):
    # Suspend the worker temporarily by adding a slow job
    with patch('engine.queue_manager.TranslationQueue._process_job') as mock_process:
        def slow_process(job):
            time.sleep(0.5)
        mock_process.side_effect = slow_process
        
        # Add first job which will block the worker
        queue_mgr.add_job("test1.docx", "out1.docx", "ja2en", "fast_nmt", "test", {})
        
        # Add second job which will stay in QUEUED state
        job_id = queue_mgr.add_job("test2.docx", "out2.docx", "ja2en", "fast_nmt", "test", {})
        job = queue_mgr.get_job(job_id)
        
        assert job.status == JobStatus.QUEUED
        
        # Cancel the second job
        success = queue_mgr.cancel_job(job_id)
        assert success is True
        assert job.status == JobStatus.CANCELLED


def test_cancel_running_job(queue_mgr):
    started_evt = time.time()
    with patch('engine.queue_manager.execute_translation') as mock_exec:
        def fake_exec(*args, cancel_event=None, **kwargs):
            # Wait until cancelled or timed out
            if cancel_event:
                cancelled = cancel_event.wait(timeout=2.0)
                if cancelled:
                    raise TranslatorError(ErrorCode.E09, detail="Cancelled by user")
            return {"total": 0, "translated": 0, "reverted": 0, "skipped": 0}

        mock_exec.side_effect = fake_exec

        job_id = queue_mgr.add_job("running.docx", "out_running.docx", "ja2en", "fast_nmt", "test", {})
        job = queue_mgr.get_job(job_id)

        # Wait until the job enters RUNNING
        timeout = time.time() + 2.0
        while job.status != JobStatus.RUNNING and time.time() < timeout:
            time.sleep(0.05)
        assert job.status == JobStatus.RUNNING

        # Cancel while running
        success = queue_mgr.cancel_job(job_id)
        assert success is True
        assert job.cancel_event.is_set()

        # Wait for worker to transition job to CANCELLED
        timeout = time.time() + 2.0
        while job.status != JobStatus.CANCELLED and time.time() < timeout:
            time.sleep(0.05)

        assert job.status == JobStatus.CANCELLED
        assert job.progress_message == "Cancelled"
        assert job.error is not None
        assert job.error.code == ErrorCode.E09


def test_cancel_running_job_cooperative_e09(queue_mgr):
    with patch('engine.queue_manager.execute_translation') as mock_exec:
        mock_exec.side_effect = TranslatorError(ErrorCode.E09, detail="Translation cancelled by user.")

        job_id = queue_mgr.add_job("coop.docx", "out_coop.docx", "ja2en", "fast_nmt", "test", {})
        job = queue_mgr.get_job(job_id)

        timeout = time.time() + 2.0
        while job.status != JobStatus.CANCELLED and time.time() < timeout:
            time.sleep(0.05)

        assert job.status == JobStatus.CANCELLED
        assert job.progress_message == "Cancelled"
        assert job.error_message == "Translation Cancelled"


def test_sequential_processing(queue_mgr):
    with patch('engine.queue_manager.TranslationQueue._process_job') as mock_process:
        processed_order = []
        def track_process(job):
            processed_order.append(job.id)
            time.sleep(0.1)
            
        mock_process.side_effect = track_process
        
        id1 = queue_mgr.add_job("1.docx", "o1.docx", "ja2en", "fast_nmt", "test", {})
        id2 = queue_mgr.add_job("2.docx", "o2.docx", "ja2en", "fast_nmt", "test", {})
        
        # Wait for both to complete
        timeout = time.time() + 2.0
        while len(processed_order) < 2 and time.time() < timeout:
            time.sleep(0.05)
            
        assert processed_order == [id1, id2]
        
def test_clear_completed(queue_mgr):
    with patch('engine.queue_manager.TranslationQueue._process_job'):
        id1 = queue_mgr.add_job("1.docx", "o1.docx", "ja2en", "fast_nmt", "test", {})
        id2 = queue_mgr.add_job("2.docx", "o2.docx", "ja2en", "fast_nmt", "test", {})
        
        timeout = time.time() + 2.0
        while len(queue_mgr.get_all_jobs()) < 2:
            time.sleep(0.05)
            
        # wait until completed
        while any(j.status not in (JobStatus.COMPLETED, JobStatus.FAILED) for j in queue_mgr.get_all_jobs()) and time.time() < timeout:
            time.sleep(0.05)
            
        queue_mgr.clear_completed()
        assert len(queue_mgr.get_all_jobs()) == 0

def test_shutdown_stops_worker():
    qm = TranslationQueue()
    assert qm._worker_thread.is_alive()
    qm.shutdown()
    
    # Allow time for thread to exit
    time.sleep(0.1)
    assert not qm._worker_thread.is_alive()


@patch('engine.queue_manager.execute_translation')
def test_job_failure_with_translator_error(mock_execute, queue_mgr):
    err = TranslatorError(ErrorCode.E01, detail="Ollama connection refused")
    mock_execute.side_effect = err

    job_id = queue_mgr.add_job(
        input_path="test_fail.docx",
        output_path="test_fail_out.docx",
        direction="ja2en",
        mode=TranslationMode.PURE_LLM,
        model_name="gemma4:e2b-it-qat",
        glossary={}
    )
    job = queue_mgr.get_job(job_id)
    assert job is not None

    timeout = time.time() + 2.0
    while job.status not in (JobStatus.COMPLETED, JobStatus.FAILED) and time.time() < timeout:
        time.sleep(0.05)

    assert job.status == JobStatus.FAILED
    assert isinstance(job.error, TranslatorError)
    assert job.error.code == ErrorCode.E01
    assert job.error.detail == "Ollama connection refused"
    assert job.error_message == err.title
    assert job.progress_message == f"Failed: {err.title}"
    assert "E01" in job.error.format_user_dialog()


@patch('engine.queue_manager.execute_translation')
def test_job_failure_with_generic_exception(mock_execute, queue_mgr):
    mock_execute.side_effect = RuntimeError("Unexpected disk error")

    job_id = queue_mgr.add_job(
        input_path="test_crash.docx",
        output_path="test_crash_out.docx",
        direction="ja2en",
        mode=TranslationMode.FAST_NMT,
        model_name="argos",
        glossary={}
    )
    job = queue_mgr.get_job(job_id)
    assert job is not None

    timeout = time.time() + 2.0
    while job.status not in (JobStatus.COMPLETED, JobStatus.FAILED) and time.time() < timeout:
        time.sleep(0.05)

    assert job.status == JobStatus.FAILED
    assert job.error is None
    assert job.error_message == "Unexpected disk error"
    assert job.progress_message == "Failed: Unexpected disk error"


def test_gui_widget_update_failed_structured_error():
    from gui.app import TranslatorApp
    app = MagicMock()
    st_lbl = MagicMock()
    app._job_widgets = {
        "job-1": {
            "progress": MagicMock(),
            "status_label": st_lbl,
            "cancel_btn": MagicMock(),
        }
    }
    te = TranslatorError(ErrorCode.E02, detail="Model gemma missing")
    job = TranslationJob(
        id="job-1",
        input_path="sample.docx",
        output_path="sample_out.docx",
        direction="ja2en",
        mode=TranslationMode.PURE_LLM,
        model_name="gemma",
        glossary={},
        status=JobStatus.FAILED,
        progress=25.0,
        progress_message="Failed",
        error=te,
        error_message=te.title,
    )

    TranslatorApp._update_job_widget(app, job)

    assert any(call.kwargs.get("cursor") == "hand2" for call in st_lbl.configure.call_args_list)
    assert st_lbl.bind.called
    click_seq, callback = st_lbl.bind.call_args[0]
    assert click_seq == "<Button-1>"

    with patch('gui.app.messagebox.showerror') as mock_showerror:
        callback(None)
        assert mock_showerror.called
        title_arg, msg_arg = mock_showerror.call_args[0]
        assert "E02" in title_arg
        assert te.user_message in msg_arg


def test_gui_widget_update_failed_generic_error():
    from gui.app import TranslatorApp
    app = MagicMock()
    st_lbl = MagicMock()
    app._job_widgets = {
        "job-2": {
            "progress": MagicMock(),
            "status_label": st_lbl,
            "cancel_btn": MagicMock(),
        }
    }
    job = TranslationJob(
        id="job-2",
        input_path="sample.docx",
        output_path="sample_out.docx",
        direction="ja2en",
        mode=TranslationMode.PURE_LLM,
        model_name="gemma",
        glossary={},
        status=JobStatus.FAILED,
        progress=10.0,
        progress_message="Failed",
        error=None,
        error_message="Unknown system crash",
    )

    TranslatorApp._update_job_widget(app, job)

    assert any(call.kwargs.get("cursor") == "hand2" for call in st_lbl.configure.call_args_list)
    assert st_lbl.bind.called
    click_seq, callback = st_lbl.bind.call_args[0]
    assert click_seq == "<Button-1>"

    with patch('gui.app.messagebox.showerror') as mock_showerror:
        callback(None)
        assert mock_showerror.called
        title_arg, msg_arg = mock_showerror.call_args[0]
        assert title_arg == "Translation Error"
        assert msg_arg == "Unknown system crash"


def test_job_cache_policy_forwarding(queue_mgr):
    with patch('engine.queue_manager.execute_translation') as mock_exec:
        mock_exec.return_value = {"total": 0, "translated": 0, "reverted": 0, "skipped": 0}

        # 1. Default job has CachePolicy.ENCRYPTED_PERSISTENT
        jid1 = queue_mgr.add_job("f1.docx", "o1.docx", "ja2en", "fast_nmt", "model", {})
        job1 = queue_mgr.get_job(jid1)
        assert job1.cache_policy == CachePolicy.ENCRYPTED_PERSISTENT

        # 2. Custom job with string/enum cache policy
        jid2 = queue_mgr.add_job(
            "f2.docx", "o2.docx", "ja2en", "fast_nmt", "model", {},
            cache_policy="memory_only"
        )
        job2 = queue_mgr.get_job(jid2)
        assert job2.cache_policy == CachePolicy.MEMORY_ONLY

        # Wait for job2 to finish processing
        timeout = time.time() + 2.0
        while (job1.status not in (JobStatus.COMPLETED, JobStatus.FAILED) or
               job2.status not in (JobStatus.COMPLETED, JobStatus.FAILED)) and time.time() < timeout:
            time.sleep(0.05)

        # Assert execute_translation was called with each job's cache_policy
        assert mock_exec.call_count >= 2
        calls = mock_exec.call_args_list
        assert any(c.kwargs.get("cache_policy") == CachePolicy.ENCRYPTED_PERSISTENT for c in calls)
        assert any(c.kwargs.get("cache_policy") == CachePolicy.MEMORY_ONLY for c in calls)


def test_job_row_update_job_completed_with_stats():
    from gui.widgets.job_row import JobRow
    st_lbl = MagicMock()
    progress_bar = MagicMock()
    cancel_btn = MagicMock()

    job = TranslationJob(
        id="job-stats-1",
        input_path="sample.docx",
        output_path="sample_out.docx",
        direction="ja2en",
        mode=TranslationMode.FAST_NMT,
        model_name="argos",
        glossary={},
        status=JobStatus.COMPLETED,
        progress=100.0,
        progress_message="Completed",
        started_at=100.0,
        completed_at=165.0,  # 65s -> 01:05
        result={"total": 10, "translated": 8, "reverted": 2, "skipped": 1},
    )

    row = MagicMock(spec=JobRow)
    row.st_label = st_lbl
    row.progress_bar = progress_bar
    row.cancel_button = cancel_btn

    JobRow.update_job(row, job)

    assert progress_bar.set.called
    progress_bar.set.assert_called_with(1.0)
    assert cancel_btn.configure.called
    cancel_btn.configure.assert_called_with(state="disabled")

    assert st_lbl.configure.called
    text_val = st_lbl.configure.call_args.kwargs.get("text", "")
    assert "✅ Done (01:05)" in text_val
    assert "8 translated" in text_val
    assert "⚠ 2 reverted" in text_val
    assert "1 skipped" in text_val


def test_job_row_update_job_completed_clean_stats():
    from gui.widgets.job_row import JobRow
    st_lbl = MagicMock()
    progress_bar = MagicMock()
    cancel_btn = MagicMock()

    job = TranslationJob(
        id="job-stats-clean",
        input_path="sample.pptx",
        output_path="sample_out.pptx",
        direction="ja2en",
        mode=TranslationMode.FAST_NMT,
        model_name="argos",
        glossary={},
        status=JobStatus.COMPLETED,
        progress=100.0,
        progress_message="Completed",
        started_at=100.0,
        completed_at=115.0,  # 15s -> 00:15
        result={"total": 5, "translated": 5, "reverted": 0, "skipped": 0},
    )

    row = MagicMock(spec=JobRow)
    row.st_label = st_lbl
    row.progress_bar = progress_bar
    row.cancel_button = cancel_btn

    JobRow.update_job(row, job)

    text_val = st_lbl.configure.call_args.kwargs.get("text", "")
    assert text_val == "✅ Done (00:15) — 5 translated"


def test_job_row_update_job_completed_without_result():
    from gui.widgets.job_row import JobRow
    st_lbl = MagicMock()
    progress_bar = MagicMock()
    cancel_btn = MagicMock()

    job = TranslationJob(
        id="job-stats-none",
        input_path="sample.xlsx",
        output_path="sample_out.xlsx",
        direction="ja2en",
        mode=TranslationMode.FAST_NMT,
        model_name="argos",
        glossary={},
        status=JobStatus.COMPLETED,
        progress=100.0,
        progress_message="Completed",
        started_at=100.0,
        completed_at=105.0,  # 5s -> 00:05
        result=None,
    )

    row = MagicMock(spec=JobRow)
    row.st_label = st_lbl
    row.progress_bar = progress_bar
    row.cancel_button = cancel_btn

    JobRow.update_job(row, job)

    text_val = st_lbl.configure.call_args.kwargs.get("text", "")
    assert text_val == "✅ Done (00:05)"


def test_gui_widget_update_completed_stats():
    from gui.app import TranslatorApp
    app = MagicMock()
    st_lbl = MagicMock()
    app._job_widgets = {
        "job-done-1": {
            "progress": MagicMock(),
            "status_label": st_lbl,
            "cancel_btn": MagicMock(),
        }
    }
    job = TranslationJob(
        id="job-done-1",
        input_path="sample.docx",
        output_path="sample_out.docx",
        direction="ja2en",
        mode=TranslationMode.PURE_LLM,
        model_name="gemma",
        glossary={},
        status=JobStatus.COMPLETED,
        progress=100.0,
        progress_message="Completed",
        started_at=50.0,
        completed_at=70.0,
        result={"total": 12, "translated": 11, "reverted": 1, "skipped": 0},
    )

    TranslatorApp._update_job_widget(app, job)

    assert st_lbl.configure.called
    text_val = st_lbl.configure.call_args.kwargs.get("text", "")
    assert "✅ Done (00:20)" in text_val
    assert "11 translated" in text_val
    assert "⚠ 1 reverted" in text_val


def test_format_eta():
    from engine.queue_manager import format_eta
    assert format_eta(None) == ""
    assert format_eta(0) == ""
    assert format_eta(-10) == ""
    assert format_eta(15) == "~15s left"
    assert format_eta(60) == "~60s left"
    assert format_eta(119) == "~119s left"
    assert format_eta(120) == "~2m left"
    assert format_eta(180) == "~3m left"
    assert format_eta(3599) == "~60m left"
    assert format_eta(3600) == "~1h left"
    assert format_eta(3900) == "~1h 5m left"
    assert format_eta(7200) == "~2h left"


def test_job_eta_str_property():
    job = TranslationJob(
        id="job-eta-1",
        input_path="sample.docx",
        output_path="sample_out.docx",
        direction="ja2en",
        mode=TranslationMode.FAST_NMT,
        model_name="argos",
        glossary={},
        status=JobStatus.RUNNING,
        progress=50.0,
        progress_message="Translating",
        started_at=100.0,
        eta_seconds=45.0,
    )
    assert job.eta_str == "~45s left"

    # Test dynamic fallback calculation when eta_seconds is None
    job.eta_seconds = None
    with patch("time.time", return_value=120.0):  # elapsed = 20s, at 50% -> remaining = 20s
        assert job.eta_str == "~20s left"

    # Completed job has empty eta_str
    job.status = JobStatus.COMPLETED
    assert job.eta_str == ""


def test_queue_manager_progress_cb_calculates_eta(queue_mgr):
    with patch('engine.queue_manager.execute_translation') as mock_execute:
        def fake_execute(*args, progress_cb=None, **kwargs):
            if progress_cb:
                progress_cb(1, 4, "Chunk 1")
                # Wait briefly so elapsed > 0
                time.sleep(0.05)
                progress_cb(2, 4, "Chunk 2")
            return {"total": 4, "translated": 4, "reverted": 0, "skipped": 0}

        mock_execute.side_effect = fake_execute

        job_id = queue_mgr.add_job("doc.docx", "doc_out.docx", "ja2en", "fast_nmt", "test", {})
        job = queue_mgr.get_job(job_id)

        timeout = time.time() + 2.0
        while job.status != JobStatus.COMPLETED and time.time() < timeout:
            time.sleep(0.02)

        assert job.status == JobStatus.COMPLETED
        # On completion, eta_seconds is reset to 0.0
        assert job.eta_seconds == 0.0


def test_job_row_update_job_running_with_eta():
    from gui.widgets.job_row import JobRow
    st_lbl = MagicMock()
    progress_bar = MagicMock()
    cancel_btn = MagicMock()

    job = TranslationJob(
        id="job-running-eta",
        input_path="sample.docx",
        output_path="sample_out.docx",
        direction="ja2en",
        mode=TranslationMode.FAST_NMT,
        model_name="argos",
        glossary={},
        status=JobStatus.RUNNING,
        progress=45.0,
        progress_message="Translating text node 9/20...",
        started_at=100.0,
        eta_seconds=35.0,
    )

    row = MagicMock(spec=JobRow)
    row.st_label = st_lbl
    row.progress_bar = progress_bar
    row.cancel_button = cancel_btn

    JobRow.update_job(row, job)

    assert progress_bar.set.called
    assert cancel_btn.configure.called
    cancel_btn.configure.assert_called_with(state="normal")

    assert st_lbl.configure.called
    text_val = st_lbl.configure.call_args.kwargs.get("text", "")
    assert "🔄 45%" in text_val
    assert "(~35s left)" in text_val
    assert "Translating text" in text_val


def test_job_row_update_job_running_without_eta():
    from gui.widgets.job_row import JobRow
    st_lbl = MagicMock()
    progress_bar = MagicMock()
    cancel_btn = MagicMock()

    job = TranslationJob(
        id="job-running-no-eta",
        input_path="sample.docx",
        output_path="sample_out.docx",
        direction="ja2en",
        mode=TranslationMode.FAST_NMT,
        model_name="argos",
        glossary={},
        status=JobStatus.RUNNING,
        progress=0.0,
        progress_message="Starting...",
        started_at=100.0,
        eta_seconds=None,
    )

    row = MagicMock(spec=JobRow)
    row.st_label = st_lbl
    row.progress_bar = progress_bar
    row.cancel_button = cancel_btn

    JobRow.update_job(row, job)

    text_val = st_lbl.configure.call_args.kwargs.get("text", "")
    assert text_val == "🔄 0% Starting..."


def test_gui_widget_update_running_eta():
    from gui.app import TranslatorApp
    app = MagicMock()
    st_lbl = MagicMock()
    app._job_widgets = {
        "job-run-1": {
            "progress": MagicMock(),
            "status_label": st_lbl,
            "cancel_btn": MagicMock(),
        }
    }
    job = TranslationJob(
        id="job-run-1",
        input_path="sample.docx",
        output_path="sample_out.docx",
        direction="ja2en",
        mode=TranslationMode.PURE_LLM,
        model_name="gemma",
        glossary={},
        status=JobStatus.RUNNING,
        progress=60.0,
        progress_message="Translating...",
        started_at=100.0,
        eta_seconds=150.0,
    )

    TranslatorApp._update_job_widget(app, job)

    assert st_lbl.configure.called
    text_val = st_lbl.configure.call_args.kwargs.get("text", "")
    assert "🔄 60%" in text_val
    assert "(~2m left)" in text_val


def test_job_row_completed_micro_actions_without_review_log(tmp_path):
    from gui.widgets.job_row import JobRow
    st_lbl = MagicMock()
    progress_bar = MagicMock()
    cancel_btn = MagicMock()
    open_file_btn = MagicMock()
    open_folder_btn = MagicMock()
    review_btn = MagicMock()

    out_file = str(tmp_path / "result.docx")
    # File exists on disk
    with open(out_file, "w") as f:
        f.write("test")

    job = TranslationJob(
        id="job-micro-1",
        input_path="sample.docx",
        output_path=out_file,
        direction="ja2en",
        mode=TranslationMode.FAST_NMT,
        model_name="argos",
        glossary={},
        status=JobStatus.COMPLETED,
        progress=100.0,
        progress_message="Completed",
        started_at=100.0,
        completed_at=110.0,
        review_log_path=str(tmp_path / "result.docx.needs_review.log"),
    )

    row = MagicMock(spec=JobRow)
    row.st_label = st_lbl
    row.progress_bar = progress_bar
    row.cancel_button = cancel_btn
    row.open_file_button = open_file_btn
    row.open_folder_button = open_folder_btn
    row.review_button = review_btn
    row._output_path = ""
    row._review_log_path = ""

    JobRow.update_job(row, job)

    assert cancel_btn.pack_forget.called
    assert open_file_btn.pack.called
    assert open_folder_btn.pack.called
    # Review log does not exist on disk, so review button should be hidden
    assert review_btn.pack_forget.called


def test_job_row_completed_micro_actions_with_review_log(tmp_path):
    from gui.widgets.job_row import JobRow
    st_lbl = MagicMock()
    progress_bar = MagicMock()
    cancel_btn = MagicMock()
    open_file_btn = MagicMock()
    open_folder_btn = MagicMock()
    review_btn = MagicMock()

    out_file = str(tmp_path / "result.docx")
    with open(out_file, "w") as f:
        f.write("content")

    rev_file = str(tmp_path / "result.docx.needs_review.log")
    with open(rev_file, "w") as f:
        f.write("Item needing review")

    job = TranslationJob(
        id="job-micro-rev",
        input_path="sample.docx",
        output_path=out_file,
        direction="ja2en",
        mode=TranslationMode.FAST_NMT,
        model_name="argos",
        glossary={},
        status=JobStatus.COMPLETED,
        progress=100.0,
        progress_message="Completed",
        started_at=100.0,
        completed_at=110.0,
        review_log_path=rev_file,
    )

    row = MagicMock(spec=JobRow)
    row.st_label = st_lbl
    row.progress_bar = progress_bar
    row.cancel_button = cancel_btn
    row.open_file_button = open_file_btn
    row.open_folder_button = open_folder_btn
    row.review_button = review_btn
    row._output_path = ""
    row._review_log_path = ""

    JobRow.update_job(row, job)

    assert cancel_btn.pack_forget.called
    assert open_file_btn.pack.called
    assert open_folder_btn.pack.called
    # Review log exists and has content, so review button is packed
    assert review_btn.pack.called


def test_job_row_open_action_handlers(tmp_path):
    from gui.widgets.job_row import JobRow

    out_file = str(tmp_path / "sample_out.docx")
    with open(out_file, "w") as f:
        f.write("sample")

    rev_file = str(tmp_path / "sample_out.docx.needs_review.log")
    with open(rev_file, "w") as f:
        f.write("review items")

    row = MagicMock(spec=JobRow)
    row._output_path = out_file
    row._review_log_path = rev_file

    with patch("os.startfile", create=True) as mock_startfile:
        with patch("subprocess.Popen") as mock_popen:
            # 1. Open output file
            JobRow._handle_open_file(row)
            if sys.platform == "win32":
                mock_startfile.assert_called_with(out_file)
            else:
                assert mock_popen.called

            # 2. Open containing folder (selects file on Windows)
            JobRow._handle_open_folder(row)
            if sys.platform == "win32":
                mock_popen.assert_called_with(["explorer", "/select,", os.path.abspath(out_file)])
            else:
                assert mock_popen.called

            # 3. Open review log
            JobRow._handle_open_review(row)
            if sys.platform == "win32":
                mock_startfile.assert_called_with(rev_file)
            else:
                assert mock_popen.called


def test_documents_view_multi_job_post_action_summary(tmp_path):
    from gui.views.documents_view import DocumentsView
    from gui.widgets.job_row import JobRow

    view = MagicMock(spec=DocumentsView)
    view._completed_outputs = []
    view._completed_review_logs = []
    view._last_output_path = None
    view._last_review_log = None
    view.open_file_btn = MagicMock()
    view.open_dir_btn = MagicMock()
    view.review_btn = MagicMock()
    view._update_post_action_buttons = lambda: DocumentsView._update_post_action_buttons(view)

    out1 = str(tmp_path / "doc1.docx")
    with open(out1, "w") as f: f.write("1")
    out2 = str(tmp_path / "doc2.docx")
    with open(out2, "w") as f: f.write("2")

    job1 = TranslationJob(
        id="j1", input_path="d1.docx", output_path=out1, direction="ja2en",
        mode=TranslationMode.FAST_NMT, model_name="m", glossary={},
        status=JobStatus.COMPLETED, progress=100.0, progress_message="Done",
        review_log_path="",
    )
    job2 = TranslationJob(
        id="j2", input_path="d2.docx", output_path=out2, direction="ja2en",
        mode=TranslationMode.FAST_NMT, model_name="m", glossary={},
        status=JobStatus.COMPLETED, progress=100.0, progress_message="Done",
        review_log_path=str(tmp_path / "rev.log"),
    )
    with open(str(tmp_path / "rev.log"), "w") as f: f.write("review log")

    # 1. Complete job 1: single output summary
    view._job_widgets = {job1.id: MagicMock(spec=JobRow), job2.id: MagicMock(spec=JobRow)}
    DocumentsView.update_job(view, job1)
    assert view._completed_outputs == [out1]
    assert view.open_file_btn.configure.called
    assert view.open_file_btn.configure.call_args.kwargs["text"] == "📄  Open Output File"
    assert view.open_file_btn.configure.call_args.kwargs["state"] == "normal"
    assert view.open_dir_btn.configure.call_args.kwargs["text"] == "📁  Open Folder"

    # 2. Complete job 2: multi-output batch summary
    DocumentsView.update_job(view, job2)
    assert len(view._completed_outputs) == 2
    assert view.open_file_btn.configure.call_args.kwargs["text"] == "📄  Open All Outputs (2)"
    assert view.open_dir_btn.configure.call_args.kwargs["text"] == "📁  Open Output Folders"
    assert view.review_btn.configure.call_args.kwargs["text"] == "⚠  Review Log"
    assert view.review_btn.configure.call_args.kwargs["state"] == "normal"

    # 3. Test open_last_output opens all completed outputs
    with patch("os.startfile", create=True) as mock_startfile:
        with patch("subprocess.Popen") as mock_popen:
            DocumentsView.open_last_output(view)
            if sys.platform == "win32":
                assert mock_startfile.call_count == 2
            else:
                assert mock_popen.call_count == 2




