"""
tests/test_logging.py
=====================
Unit tests for the structured logging subsystem:
- TranslationLogEvent formatting and serialization
- TranslationLogger event publishing and subscription
- Legacy callback bridging (as_log_cb and add_legacy_callback)
- TranslationEngine structured log emission
- execute_translation logger integration
"""

import unittest

from engine.core import TranslationEngine, TranslationMode
from engine.logging import (
    TranslationLogEvent,
    TranslationLogger,
    get_logger,
    set_default_logger,
)


class TestTranslationLogEvent(unittest.TestCase):
    """Verifies TranslationLogEvent dataclass, serialization, and formatters."""

    def test_event_to_dict_and_json(self):
        event = TranslationLogEvent(
            level="info",
            category="cache",
            message="Cache Hit",
            location="slide1.p0",
            elapsed=0.0,
            details={"key": "val"},
        )
        d = event.to_dict()
        self.assertEqual(d["level"], "info")
        self.assertEqual(d["category"], "cache")
        self.assertEqual(d["location"], "slide1.p0")
        self.assertEqual(d["details"], {"key": "val"})

        j = event.to_json()
        self.assertIn('"category": "cache"', j)
        self.assertIn('"location": "slide1.p0"', j)

    def test_to_cli_string_cache(self):
        event = TranslationLogEvent(
            level="info",
            category="cache",
            message='"A" => "B"',
            location="doc1",
            elapsed=0.0,
        )
        s = event.to_cli_string()
        self.assertIn("[⚡ Cache]", s)
        self.assertIn("doc1:", s)
        self.assertIn('"A" => "B"', s)

    def test_to_cli_string_translation(self):
        event = TranslationLogEvent(
            level="info",
            category="translation",
            message='"Hello" => "Bonjour"',
            location="p1",
            elapsed=0.45,
        )
        s = event.to_cli_string()
        self.assertIn("[✓ Done in 0.45s]", s)
        self.assertIn("p1:", s)

    def test_to_cli_string_error(self):
        event = TranslationLogEvent(
            level="error",
            category="translation",
            message="Model timeout",
            location="chunk_9",
        )
        s = event.to_cli_string()
        self.assertIn("[-] Error", s)
        self.assertIn("chunk_9:", s)

    def test_to_cli_string_warning(self):
        event = TranslationLogEvent(
            level="warning",
            category="translation",
            message="Placeholder dropped",
            location="row_4",
        )
        s = event.to_cli_string()
        self.assertIn("[⚠ Warning]", s)
        self.assertIn("row_4:", s)

    def test_to_gui_string(self):
        event = TranslationLogEvent(
            level="info",
            category="system",
            message="Ready",
        )
        self.assertEqual(event.to_gui_string(), "[*] Ready")


class TestTranslationLogger(unittest.TestCase):
    """Verifies TranslationLogger event dispatcher and subscriber management."""

    def test_subscribe_and_emit(self):
        logger = TranslationLogger(name="test")
        events = []
        logger.subscribe(events.append)

        logger.info("Service started", category="system")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].level, "info")
        self.assertEqual(events[0].category, "system")
        self.assertEqual(events[0].message, "Service started")

    def test_unsubscribe(self):
        logger = TranslationLogger(name="test")
        events = []
        listener = events.append
        logger.subscribe(listener)
        logger.info("One")
        self.assertEqual(len(events), 1)

        logger.unsubscribe(listener)
        logger.info("Two")
        self.assertEqual(len(events), 1)

    def test_subscriber_exception_isolation(self):
        logger = TranslationLogger(name="test")
        broken_calls = []
        working_calls = []

        def broken_listener(event):
            broken_calls.append(event)
            raise ValueError("Listener crash")

        def working_listener(event):
            working_calls.append(event)

        logger.subscribe(broken_listener)
        logger.subscribe(working_listener)

        # Emitting should not raise, and working_listener should still be invoked
        logger.info("Resilient event")
        self.assertEqual(len(broken_calls), 1)
        self.assertEqual(len(working_calls), 1)

    def test_convenience_methods(self):
        logger = TranslationLogger(name="test")
        events = []
        logger.subscribe(events.append)

        logger.warning("Warn msg", category="translation")
        logger.error("Err msg", category="backend")
        logger.debug("Debug msg")

        self.assertEqual(events[0].level, "warning")
        self.assertEqual(events[1].level, "error")
        self.assertEqual(events[2].level, "debug")

    def test_add_legacy_callback(self):
        logger = TranslationLogger(name="test")
        strings = []
        logger.add_legacy_callback(strings.append)

        logger.info("Cache hit text", category="cache", location="loc1")
        self.assertEqual(len(strings), 1)
        self.assertIn("[⚡ Cache]", strings[0])
        self.assertIn("loc1:", strings[0])

    def test_as_log_cb(self):
        logger = TranslationLogger(name="test")
        events = []
        logger.subscribe(events.append)

        cb = logger.as_log_cb(category="system")
        cb("  [⚡ Cache] loc: 'A' => 'B'")
        cb("  [!] Warning: low disk")
        cb("  [-] Error in backend")
        cb("  [*] Normal status message")

        self.assertEqual(len(events), 4)
        self.assertEqual(events[0].category, "cache")
        self.assertEqual(events[1].level, "warning")
        self.assertEqual(events[2].level, "error")
        self.assertEqual(events[3].level, "info")

    def test_global_logger_getter_and_setter(self):
        custom = TranslationLogger(name="custom_singleton")
        set_default_logger(custom)
        self.assertIs(get_logger(), custom)


class TestEngineStructuredLoggingIntegration(unittest.TestCase):
    """Verifies that TranslationEngine emits structured events to TranslationLogger."""

    def test_engine_emits_structured_events_on_translate(self):
        class MockNMT:
            name = "nmt"

            def is_ready(self, direction):
                return True

            def translate(self, text, direction, **kwargs):
                return f"[NMT: {text}]", 0.05

        test_logger = TranslationLogger(name="engine_test")
        events = []
        test_logger.subscribe(events.append)

        engine = TranslationEngine(
            mode=TranslationMode.FAST_NMT,
            backend=MockNMT(),
            logger=test_logger,
        )

        # 1. Translation emission
        res = engine.translate_chunk("これはテストです。", "ja2en", location_id="chunk_1")
        self.assertTrue(res.was_translated)
        trans_events = [e for e in events if e.category == "translation"]
        self.assertGreaterEqual(len(trans_events), 1)
        self.assertEqual(trans_events[-1].location, "chunk_1")
        self.assertEqual(trans_events[-1].level, "info")
        self.assertGreater(trans_events[-1].elapsed, 0.0)

        # 2. Cache hit emission
        events.clear()
        res2 = engine.translate_chunk("これはテストです。", "ja2en", location_id="chunk_2")
        self.assertEqual(res2.source_backend, "cache")
        cache_events = [e for e in events if e.category == "cache"]
        self.assertGreaterEqual(len(cache_events), 1)
        self.assertEqual(cache_events[-1].location, "chunk_2")
        self.assertEqual(cache_events[-1].elapsed, 0.0)

    def test_engine_emits_warning_on_dropped_placeholder(self):
        class DroppingBackend:
            name = "nmt"

            def is_ready(self, direction):
                return True

            def translate(self, text, direction, **kwargs):
                return "Dropped placeholder output", 0.02

        test_logger = TranslationLogger(name="drop_test")
        events = []
        test_logger.subscribe(events.append)

        engine = TranslationEngine(
            mode=TranslationMode.FAST_NMT,
            backend=DroppingBackend(),
            logger=test_logger,
        )

        res = engine.translate_chunk("Total: 500 items", "en2ja", location_id="num_chunk")
        self.assertTrue(res.was_reverted)
        warn_events = [e for e in events if e.level == "warning"]
        self.assertGreaterEqual(len(warn_events), 1)
        self.assertEqual(warn_events[-1].location, "num_chunk")


if __name__ == "__main__":
    unittest.main()
