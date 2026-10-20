"""serve/test_multi.py - several conversations at once (READY ... multi=<n>): StrataEngine's per-request routing over a
scripted engine (serve/fake_multi_engine.py), and the service's FIFO with n places.

    python -m unittest serve.test_multi -v
"""
from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.server import FairLock, StrataEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class Engine(unittest.TestCase):
    def setUp(self):
        self.engine = StrataEngine(str(ROOT / "serve/fake_multi_engine.py"), ["3"])

    def tearDown(self):
        self.engine.close()

    def test_ready_says_how_many(self):
        self.assertEqual(self.engine.multi, 3)

    def test_concurrent_requests_each_get_their_own_tokens(self):
        out = {}

        def one(k):
            ids = list(range(k * 100, k * 100 + 40))
            toks = [t for t in self.engine.generate(ids, 40, {}, threading.Event()) if t is not None]
            out[k] = (ids, toks)

        ths = [threading.Thread(target=one, args=(k,)) for k in range(3)]
        [t.start() for t in ths]
        [t.join(30) for t in ths]
        for k, (ids, toks) in out.items():
            self.assertEqual(toks, [i + 1000 for i in ids], k)
        self.assertEqual(len(out), 3)

    def test_a_request_reads_only_its_own_figures(self):
        """--parallel: each request's timings, Monitor history row and /v1/status come from ITS DONE line.

        They used to read engine.last, which every concurrent request overwrote (and which held the PREVIOUS
        request's line while a request was still running): a 5-token prompt could get a 200-token prompt's prompt_ms
        and report its prompt processing speed.  last_request() keeps them per thread, keyed by the request's rid."""
        a = list(range(200))
        gen = self.engine.generate(a, 200, {}, threading.Event())
        got = []
        for _ in range(5):                            # A is running: its DONE has not arrived
            t = next(gen)
            if t is not None:
                got.append(t)
        self.assertIsNone(self.engine.last_request())     # ... so it has no figures yet (not B's, not an old one's)
        out = {}

        def other():
            ids = list(range(1000, 1005))
            out["toks"] = len([t for t in self.engine.generate(ids, 5, {}, threading.Event()) if t is not None])
            out["last"] = dict(self.engine.last_request())

        th = threading.Thread(target=other)
        th.start()
        th.join(30)
        self.assertEqual(out["toks"], 5)
        self.assertEqual(out["last"]["prompt_tokens"], 5)     # B's own: the prompt it sent
        for _ in gen:
            pass
        mine = self.engine.last_request()
        self.assertEqual(mine["prompt_tokens"], len(a))        # A's own, after its DONE
        self.assertEqual(mine["generated"], len(a))            # (the fake engine answers one token per prompt token)

    def test_a_consumer_that_stops_ends_only_its_request(self):
        out = {}

        def long_one():
            ids = list(range(500, 600))
            out["long"] = [t for t in self.engine.generate(ids, 100, {}, threading.Event()) if t is not None]

        th = threading.Thread(target=long_one)
        th.start()
        gen = self.engine.generate(list(range(10)), 50, {}, threading.Event())
        got = []
        for t in gen:
            if t is not None:
                got.append(t)
            if len(got) == 5:
                break
        gen.close()                                   # STOP <rid>, drained to its DONE
        self.assertEqual(self.engine.last.get("finish"), "cancel")
        th.join(30)
        self.assertEqual(out["long"], [i + 1000 for i in range(500, 600)])
        self.assertEqual(got, [1000, 1001, 1002, 1003, 1004])
        self.assertEqual(self.engine._routes, {})

    def test_an_engine_error_reaches_its_request(self):
        # a fourth request on a 3-state engine is refused for that request alone
        evs = [threading.Event() for _ in range(3)]
        gens = [self.engine.generate(list(range(200)), 200, {}, evs[i]) for i in range(3)]
        for g in gens:
            next(g)                                   # their GEN lines are out
        with self.assertRaises(ValueError) as cm:
            list(self.engine.generate([1, 2], 2, {}, threading.Event()))
        self.assertIn("busy", str(cm.exception))
        for g in gens:
            g.close()


class Places(unittest.TestCase):
    def test_n_requests_hold_n_places_and_a_whole_waits_for_all(self):
        lock = FairLock(3)
        for _ in range(3):
            self.assertTrue(lock.acquire(units=1, timeout=1))
        self.assertFalse(lock.acquire(units=1, blocking=False))
        got = []
        th = threading.Thread(target=lambda: (lock.acquire(), got.append("all"), lock.release()))
        th.start()
        while lock.waiting() < 1:
            time.sleep(0.001)
        lock.release(units=1)
        time.sleep(0.05)
        self.assertEqual(got, [])                     # the exclusive waiter needs every place
        lock.release(units=1)
        lock.release(units=1)
        th.join(5)
        self.assertEqual(got, ["all"])
        self.assertFalse(lock.locked())

    def test_a_waiter_for_all_holds_back_later_ones(self):
        lock = FairLock(2)
        lock.acquire(units=1)
        order = []
        a = threading.Thread(target=lambda: (lock.acquire(), order.append("all"), lock.release()))
        a.start()
        while lock.waiting() < 1:
            time.sleep(0.001)
        b = threading.Thread(target=lambda: (lock.acquire(units=1), order.append("one"), lock.release(units=1)))
        b.start()
        while lock.waiting() < 2:
            time.sleep(0.001)
        time.sleep(0.05)
        self.assertEqual(order, [])                   # a free place, but the first in line wants both
        lock.release(units=1)
        a.join(5)
        b.join(5)
        self.assertEqual(order, ["all", "one"])


if __name__ == "__main__":
    unittest.main()
