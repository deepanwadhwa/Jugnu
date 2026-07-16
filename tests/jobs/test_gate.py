#!/usr/bin/env python3
"""tests/jobs/test_gate.py — J1.13 resource gate tests.

Uses fake_serve to verify that:
1. interactive_active=true → job pauses with reason "interactive_chat"
2. Flipping to false → job resumes
3. min_free_gb above actual → pause with "low_disk"
4. Thread budget is reported from /internal/v1/status
"""

import json
import os
import sys
import tempfile
import time
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'dist'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import samosa_jobs
import fake_serve


class TestGateInterlock(unittest.TestCase):
    """Gate pauses when serve reports interactive_active."""

    def setUp(self):
        self.server, self.port = fake_serve.start_server(0)
        self.serve_url = f'http://127.0.0.1:{self.port}'

    def tearDown(self):
        self.server.shutdown()

    def _job(self, **overrides):
        job = {
            'job_id': 'test-gate',
            'input': {'folder': '/tmp/nonexistent'},
            'instruction': 'Extract.',
            'output_schema': {'type': 'object', 'properties': {'x': {'type': 'string'}}},
            'resources': {
                'max_attempts': 1,
                'run_on_battery': True,  # Don't test battery in CI
                'min_free_gb': 0,
            },
        }
        job.update(overrides)
        validated, _ = samosa_jobs.validate_job(job)
        return validated

    def test_interactive_active_pauses(self):
        """When interactive_active=true, gate_check returns False."""
        fake_serve.set_status(interactive_active=True)
        ok, reason = samosa_jobs.gate_check(self._job(), self.serve_url)
        self.assertFalse(ok)
        self.assertEqual(reason, 'interactive_chat')

    def test_interactive_cleared_proceeds(self):
        """When interactive_active=false and no recent timestamp, gate_check returns True."""
        fake_serve.set_status(interactive_active=False, last_interactive_ts=None)
        ok, reason = samosa_jobs.gate_check(self._job(), self.serve_url)
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_recent_interactive_pauses(self):
        """Gate pauses if last_interactive_ts is within 60s."""
        # Create an ISO timestamp less than 60s ago
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        recent = now.strftime('%Y-%m-%dT%H:%M:%SZ')
        fake_serve.set_status(interactive_active=False, last_interactive_ts=recent)
        ok, reason = samosa_jobs.gate_check(self._job(), self.serve_url)
        self.assertFalse(ok)
        self.assertEqual(reason, 'interactive_chat')

    def test_old_interactive_proceeds(self):
        """Gate clears if last_interactive_ts is >60s ago."""
        import datetime
        old = (datetime.datetime.now(datetime.timezone.utc) -
               datetime.timedelta(seconds=120)).strftime('%Y-%m-%dT%H:%M:%SZ')
        fake_serve.set_status(interactive_active=False, last_interactive_ts=old)
        ok, reason = samosa_jobs.gate_check(self._job(), self.serve_url)
        self.assertTrue(ok)

    def test_low_disk_pauses(self):
        """min_free_gb above actual free space triggers low_disk."""
        job = self._job(resources={'min_free_gb': 999999, 'run_on_battery': True, 'max_attempts': 1})
        fake_serve.set_status(interactive_active=False, last_interactive_ts=None)
        ok, reason = samosa_jobs.gate_check(job, self.serve_url)
        self.assertFalse(ok)
        self.assertEqual(reason, 'low_disk')

    def test_thread_budget_in_status(self):
        """Thread count from /internal/v1/status is accessible."""
        fake_serve.set_status(threads=4)
        status = samosa_jobs.get_serve_status(self.serve_url)
        self.assertIsNotNone(status)
        self.assertEqual(status['threads'], 4)


class TestGateNoServe(unittest.TestCase):
    """Gate when serve is unreachable — should still run (not gate on connection error)."""

    def test_serve_down_proceeds(self):
        """If serve is unreachable, gate proceeds (no connection = no interactive chat)."""
        job = {
            'job_id': 'test-gate-down',
            'input': {'folder': '/tmp/nonexistent'},
            'instruction': 'Extract.',
            'output_schema': {'type': 'object', 'properties': {'x': {'type': 'string'}}},
            'resources': {'min_free_gb': 0, 'run_on_battery': True, 'max_attempts': 1},
        }
        validated, _ = samosa_jobs.validate_job(job)
        ok, reason = samosa_jobs.gate_check(validated, 'http://127.0.0.1:19999')
        self.assertTrue(ok)


if __name__ == '__main__':
    unittest.main()
