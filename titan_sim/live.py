"""Live Python submission and host waits against the simulator's virtual clock.

This bridge runs submitted event graphs, not numerical GPU kernels. It supports
one coordinated host clock; callers must submit a closed distributed transaction
after all participants arrive. It does not provide native CUDA DeviceGuard.
CPU tracing time does not advance modeled time. Wall-clock pacing is optional
and occurs at host waits, never at asynchronous launches or stream waits.
"""
from __future__ import annotations

from copy import deepcopy
import time

from .events import Event, nonnegative
from .runtime import simulate


class LiveExecution:
    def __init__(self, hardware, *, timing='roofline', calibrations=None,
                 realtime=False, clock=time.monotonic, sleep=time.sleep):
        self.hardware = deepcopy(hardware)
        self.timing, self.calibrations = timing, deepcopy(calibrations)
        self.realtime, self.clock, self.sleep = realtime, clock, sleep
        self.host_time_us = 0.0
        self.events = []
        self.waits = []
        self._ids = set()
        self._result = None
        self._capture_counts = {}

    def submit(self, events):
        """Submit a closed batch atomically; dependencies may reference its peers."""
        return self._submit(events, enqueued=False)

    def submit_enqueued(self, events):
        """Resolve previously enqueued distributed work at its recorded time.

        Only the rendezvous adapter should call this. Existing event timings
        must remain unchanged; conflicting retrospective insertion is rejected.
        This allows host work between async enqueue and Work.wait without
        moving the device launch to the time of that wait.
        """
        if any('host_time_us' not in event for event in events):
            raise ValueError('resolved work requires recorded enqueue timestamps')
        return self._submit(events, enqueued=True)

    def _submit(self, events, *, enqueued):
        batch = deepcopy(events)
        new_ids = set()
        for raw in batch:
            raw.setdefault('host_time_us', self.host_time_us)
            event = Event.from_dict(raw)
            if event.event_id in self._ids or event.event_id in new_ids:
                raise ValueError('duplicate live event ID: ' + event.event_id)
            if not enqueued and event.host_time_us < self.host_time_us:
                raise ValueError('cannot submit work before the committed host clock')
            new_ids.add(event.event_id)
        available = self._ids | new_ids
        for raw in batch:
            if set(raw.get('deps', ())) - available:
                raise ValueError('live transaction has unresolved dependencies')
        # Validate before accepting: a failed transaction cannot poison the queue.
        combined = self.events + batch
        if enqueued:
            # Runtime submission floors follow program order. Appending an old
            # enqueue behind newer host work would silently shift its launch.
            combined.sort(key=lambda event: event['host_time_us'])
        result = simulate(combined, self.hardware, self.timing, self.calibrations)
        if enqueued and self._result:
            old_times = {row['event_id']: (row['start_us'], row['end_us'])
                         for row in self._result['timeline']}
            for row in result['timeline']:
                if row['event_id'] in old_times and old_times[row['event_id']] != (row['start_us'], row['end_us']):
                    raise ValueError('resolved work would change already accepted event timing')
        self.events = combined
        self._ids.update(new_ids)
        self._result = result
        return [raw['event_id'] for raw in batch]

    def ingest_capture(self, capture):
        key = id(capture)
        count = self._capture_counts.get(key, 0)
        self.submit(capture.events[count:])
        self._capture_counts[key] = len(capture.events)

    def revise_storage_metadata(self, event):
        """Propagate late parameter/optimizer classification without retiming work."""
        combined = deepcopy(self.events)
        old = next((row for row in combined if row['event_id'] == event['event_id']), None)
        if old is None or old['kind'] != 'allocate':
            raise ValueError('storage metadata revision requires an accepted allocation')
        for key in ('category', 'device'):
            if key in event:
                old[key] = event[key]
        result = simulate(combined, self.hardware, self.timing, self.calibrations)
        self.events, self._result = combined, result

    def _event(self, event_id):
        if self._result is None or event_id not in self._ids:
            raise ValueError('unknown or unresolved live event: ' + str(event_id))
        return next(row for row in self._result['timeline'] if row['event_id'] == event_id)

    def query(self, event_id):
        return self._event(event_id)['end_us'] <= self.host_time_us

    def wait(self, event_id):
        completion = self._event(event_id)['end_us']
        delay = max(0.0, completion - self.host_time_us)
        if self.realtime and delay:
            start = nonnegative(self.clock(), 'clock')
            now = start
            while now - start < delay / 1e6:
                self.sleep(delay / 1e6 - (now - start))
                after = nonnegative(self.clock(), 'clock')
                if after <= now:
                    raise ValueError('pacing clock must advance after sleep')
                now = after
        self.host_time_us += delay
        self.waits.append({'event_id': event_id, 'wait_us': delay,
                           'host_time_us': self.host_time_us, 'realtime': self.realtime})
        return True

    def advance_host(self, duration_us):
        """Advance by explicitly supplied host work, excluding tracing overhead."""
        self.host_time_us += nonnegative(duration_us, 'host duration')

    def elapsed_time_ms(self, first, last):
        if not self.query(first) or not self.query(last):
            raise RuntimeError('timing events have not completed')
        delta = self._event(last)['end_us'] - self._event(first)['end_us']
        if delta < 0:
            raise ValueError('end event precedes start event')
        return delta / 1000

    def document(self):
        result = deepcopy(self._result or simulate([], self.hardware, self.timing, self.calibrations))
        result['live_execution'] = {
            'python_control_interception': True, 'native_cuda_interception': False,
            'host_clock_scope': 'one coordinated host clock',
            'host_time_us': self.host_time_us, 'waits': deepcopy(self.waits),
            'realtime': self.realtime,
            'limitation': 'Closed submitted event graphs only; no native GPU kernels or full-model coverage implied.'}
        return result
