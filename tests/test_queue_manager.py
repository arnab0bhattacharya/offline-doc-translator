import os
import time
import pytest
from unittest.mock import patch, MagicMock

from engine.queue_manager import TranslationQueue, JobStatus, TranslationJob

@pytest.fixture
def queue_mgr():
    qm = TranslationQueue()
    yield qm
    qm.shutdown()

@patch('engine.queue_manager.run_preflight')
@patch('engine.queue_manager.TranslationEngine')
@patch('engine.queue_manager.DOCXHandler')
def test_job_creation_and_processing(mock_handler, mock_engine, mock_preflight, queue_mgr):
    # Setup mocks

    mock_instance = mock_handler.return_value
    
    # We will simulate a translation that takes a small amount of time and triggers progress
    def fake_translate(input_path, output_path, direction, review_log_path, progress_cb, log_cb):
        progress_cb(1, 2, "Halfway")
        time.sleep(0.1)
        progress_cb(2, 2, "Done")
        return {"total": 2, "translated": 2, "reverted": 0, "skipped": 0}
        
    mock_instance.translate.side_effect = fake_translate
    
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
