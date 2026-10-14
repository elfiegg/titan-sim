import unittest
from titan_sim.pacing import pace_host_sync


class FakeClock:
    def __init__(self):
        self.now = 10.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class PacingTests(unittest.TestCase):
    def test_virtual_default_never_calls_clock_or_sleep(self):
        def forbidden(*args):
            raise AssertionError('virtual mode must not pace')
        result = pace_host_sync({'timeline': [{'kind': 'host_sync', 'end_us': 1000}]},
                                clock=forbidden, sleep=forbidden)
        self.assertEqual(result['mode'], 'virtual')
        self.assertEqual(result['sync_count'], 1)
        self.assertEqual(result['requested_sleep_s'], 0)

    def test_sleeps_only_at_sync_completions_in_time_order(self):
        fake = FakeClock()
        timeline = [{'event_id': 'kernel', 'kind': 'compute', 'end_us': 9000000},
                    {'event_id': 'sync_b', 'kind': 'host_sync', 'rank': 1, 'end_us': 2000000},
                    {'event_id': 'wait', 'kind': 'wait', 'end_us': 7000000},
                    {'event_id': 'sync_a', 'kind': 'host_sync', 'rank': 0, 'end_us': 500000}]
        result = pace_host_sync({'timeline': timeline}, realtime=True, clock=fake.clock, sleep=fake.sleep)
        self.assertEqual(fake.sleeps, [0.5, 1.5])
        self.assertEqual(result['elapsed_s'], 2)
        self.assertFalse(result['live_interception'])
        self.assertEqual([sync['event_id'] for sync in result['syncs']], ['sync_a', 'sync_b'])

    def test_oversleep_is_not_added_again(self):
        fake = FakeClock()
        def oversleep(seconds):
            fake.sleeps.append(seconds)
            fake.now += seconds + 2
        result = pace_host_sync({'timeline': [{'kind': 'host_sync', 'end_us': 1000000},
                                               {'kind': 'host_sync', 'end_us': 2000000}]},
                                realtime=True, clock=fake.clock, sleep=oversleep)
        self.assertEqual(fake.sleeps, [1])
        self.assertEqual(result['elapsed_s'], 3)

    def test_invalid_or_stuck_clock_rejected(self):
        with self.assertRaises(ValueError):
            pace_host_sync({'timeline': [{'kind': 'host_sync', 'end_us': -1}]})
        with self.assertRaisesRegex(ValueError, 'advance'):
            pace_host_sync({'timeline': [{'kind': 'host_sync', 'end_us': 1}]},
                           realtime=True, clock=lambda: 1, sleep=lambda seconds: None)


if __name__ == '__main__':
    unittest.main()
