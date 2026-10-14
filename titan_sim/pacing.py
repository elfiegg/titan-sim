"""Optional wall-clock demonstration of host synchronization from a saved timeline.

This is not a live TorchTitan/CUDA interception layer. Normal simulation uses
virtual time and never sleeps. Playback sleeps only for host_sync completions,
not for individual kernels or stream waits.
"""
import time
from collections.abc import Callable

from .events import nonnegative


def pace_host_sync(result: dict, *, realtime: bool = False,
                   clock: Callable[[], float] = time.monotonic,
                   sleep: Callable[[float], None] = time.sleep) -> dict:
    """Replay rank host-sync completion times against one monotonic clock origin.

    ``clock`` returns seconds; ``sleep`` accepts seconds. Completion times from
    multiple ranks are sorted on their common simulated clock. Passing injected
    functions supports deterministic tests without real waiting.
    """
    syncs = []
    for row in result.get('timeline', []):
        if row.get('kind') == 'host_sync':
            syncs.append({'event_id': str(row.get('event_id', '')),
                          'rank': row.get('rank', 0),
                          'completion_us': nonnegative(row.get('end_us', 0), 'host_sync end_us')})
    syncs.sort(key=lambda row: row['completion_us'])
    report = {'mode': 'wall_clock_demonstration' if realtime else 'virtual',
              'live_interception': False, 'sync_count': len(syncs),
              'requested_sleep_s': 0.0, 'elapsed_s': 0.0, 'syncs': syncs}
    if not realtime or not syncs:
        return report
    origin = nonnegative(clock(), 'monotonic clock')
    previous = origin
    for sync in syncs:
        target = origin + sync['completion_us'] * 1e-6
        now = nonnegative(clock(), 'monotonic clock')
        if now < previous:
            raise ValueError('injected monotonic clock moved backwards')
        while target - now > 1e-9:
            remaining = target - now
            sleep(remaining)
            report['requested_sleep_s'] += remaining
            after = nonnegative(clock(), 'monotonic clock')
            if after <= now:
                raise ValueError('injected clock must advance after sleep')
            now = after
        sync['observed_elapsed_s'] = now - origin
        previous = now
    report['elapsed_s'] = previous - origin
    return report
