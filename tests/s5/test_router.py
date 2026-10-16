import copy
import json
import math
import unittest

from successors.frem_s5.router import Episode, Router, mixture_nll, nll, solve


class RouterTests(unittest.TestCase):
    def test_student_t_known_density(self):
        self.assertAlmostEqual(nll(0, 0, 1, 3), -math.log(2 / (math.pi * math.sqrt(3))))

    def test_exact_mixture_density(self):
        self.assertAlmostEqual(mixture_nll([-math.log(0.2), -math.log(0.8)], [0.5, 0.5]), -math.log(0.5))

    def test_extreme_losses_stable(self):
        self.assertTrue(math.isfinite(mixture_nll([10000, 10001], [0.5, 0.5])))

    def test_invalid_distributions_rejected(self):
        for args in ((0, 0, 0, 5), (0, 0, 1, 2), (math.nan, 0, 1, 5)):
            with self.assertRaises(ValueError):
                nll(*args)

    def test_strict_maturity(self):
        r = Router("recent")
        r.weights(10, (0,))
        r.observe(Episode(10, 20, (0,), (0, 5, 5, 5)))
        self.assertEqual(r.weights(20, (0,))[0], [0.25] * 4)
        self.assertGreater(r.weights(21, (0,))[0][0], 0.25)

    def test_future_outcome_invariance(self):
        a, b = Router("context"), Router("context")
        for r, loss in ((a, (0, 20, 20, 20)), (b, (20, 0, 0, 0))):
            r.weights(10, (0,))
            r.observe(Episode(10, 100, (0,), loss))
        for time in (20, 50, 100):
            self.assertEqual(a.weights(time, (1,))[0], b.weights(time, (1,))[0])

    def test_invalid_feedback_order(self):
        r = Router("recent")
        with self.assertRaises(ValueError):
            r.observe(Episode(10, 20, (0,), (0, 0, 0, 0)))
        r.weights(10, (0,))
        r.observe(Episode(10, 20, (0,), (0, 0, 0, 0)))
        with self.assertRaises(ValueError):
            r.observe(Episode(10, 20, (0,), (0, 0, 0, 0)))

    def test_clock_and_dimensions(self):
        r = Router("context")
        r.weights(10, (0,))
        r.observe(Episode(10, 11, (0,), (0, 0, 0, 0)))
        with self.assertRaises(ValueError):
            r.weights(10, (0,))
        with self.assertRaises(ValueError):
            r.weights(12, (0, 1))

    def test_checkpoint_roundtrip(self):
        for mode in Router.MODES:
            r = Router(mode)
            for t in range(1, 8):
                r.weights(t, (t % 2,))
                r.observe(Episode(t, t + 2, (t % 2,), (1, 2, 3, 4)))
            q = Router.restore(json.loads(json.dumps(r.state())))
            self.assertEqual(r.weights(10, (0,)), q.weights(10, (0,)))

    def test_weights_bounded_and_normalized(self):
        for mode in Router.MODES:
            r = Router(mode)
            for t in range(1, 80):
                w, _ = r.weights(t, (t % 3,))
                self.assertAlmostEqual(sum(w), 1)
                self.assertTrue(all(0 < v < 1 for v in w))
                r.observe(Episode(t, t + 1, (t % 3,), (100, 0, 3, 7)))
            self.assertLessEqual(len(r.memory), 60)

    def test_random_count_and_age_bucket_match(self):
        a, b = Router("context"), Router("random")
        dates = {}
        for t in range(1, 50):
            time = t * 30 * 86400
            for r in (a, b):
                r.weights(time, (t % 3,))
                r.observe(Episode(time, time + 86400, (t % 3,), (1, 2, 3, 4)))
            dates[time] = time
        now = 51 * 30 * 86400
        da = a.weights(now, (0,))[1]
        db = b.weights(now, (0,))[1]
        buckets = lambda d: sorted(int((now - t) / (182.625 * 86400)) for t in d["episode_times"])
        self.assertEqual(buckets(da), buckets(db))

    def test_linear_solver(self):
        x = solve([[4, 1], [1, 3]], [1, 2])
        self.assertAlmostEqual(4 * x[0] + x[1], 1)
        self.assertAlmostEqual(x[0] + 3 * x[1], 2)

    def test_restore_does_not_alias_original(self):
        r = Router("recent")
        r.weights(1, (0,))
        r.observe(Episode(1, 2, (0,), (1, 2, 3, 4)))
        state = copy.deepcopy(r.state())
        q = Router.restore(state)
        q.weights(3, (0,))
        self.assertEqual(len(r.pending), 1)


if __name__ == "__main__":
    unittest.main()
