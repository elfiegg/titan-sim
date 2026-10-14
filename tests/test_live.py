import unittest

from titan_sim.live import LiveExecution

HW = {'name': 'test', 'memory_capacity_bytes': 1000,
      'memory_bandwidth_bytes_s': 1e9, 'peak_flops': {'float32': 1e12}}


def work(identifier, duration=100, **kwargs):
    return dict(event_id=identifier, kind='compute', duration_us=duration,
                duration_provenance='test scenario', **kwargs)


class LiveTests(unittest.TestCase):
    def test_submit_and_device_wait_are_different(self):
        execution = LiveExecution(HW)
        execution.submit([work('first')])
        self.assertFalse(execution.query('first'))
        self.assertEqual(execution.host_time_us, 0)
        execution.wait('first')
        self.assertTrue(execution.query('first'))
        self.assertEqual(execution.host_time_us, 100)
        execution.submit([work('second', 50)])
        self.assertEqual(execution.events[-1]['host_time_us'], 100)
        execution.wait('second')
        self.assertEqual(execution.host_time_us, 150)

    def test_wait_paces_only_remaining_virtual_work(self):
        class Clock:
            now = 10
            sleeps = []
            def read(self): return self.now
            def sleep(self, seconds):
                self.sleeps.append(seconds)
                self.now += seconds
        clock = Clock()
        execution = LiveExecution(HW, realtime=True, clock=clock.read, sleep=clock.sleep)
        execution.submit([work('first', 1e6)])
        execution.advance_host(250000)
        execution.wait('first')
        execution.wait('first')
        self.assertEqual(clock.sleeps, [.75])
        self.assertEqual(execution.host_time_us, 1e6)

    def test_invalid_transaction_does_not_corrupt_state(self):
        execution = LiveExecution(HW)
        with self.assertRaises(ValueError):
            execution.submit([work('bad', deps=['missing'])])
        self.assertEqual(execution.events, [])
        execution.submit([work('ok')])
        with self.assertRaises(ValueError):
            execution.submit([work('ok')])
        execution.wait('ok')
        with self.assertRaises(ValueError):
            execution.submit([work('past', host_time_us=0)])

    def test_closed_transaction_can_contain_forward_dependencies(self):
        execution = LiveExecution(HW)
        execution.submit([work('later', stream='a', deps=['earlier']),
                          work('earlier', stream='b')])
        execution.wait('later')
        self.assertEqual(execution.host_time_us, 200)

    def test_late_rendezvous_preserves_enqueue_time(self):
        execution = LiveExecution(HW)
        execution.advance_host(100)
        execution.submit_enqueued([work('network', 16, host_time_us=0)])
        execution.wait('network')
        self.assertEqual(execution.host_time_us, 100)
        self.assertEqual(execution.document()['timeline'][0]['end_us'], 16)
        self.assertEqual(execution.waits[-1]['wait_us'], 0)
        with self.assertRaises(ValueError):
            execution.submit_enqueued([work('missing_timestamp')])

    def test_resolved_enqueue_precedes_later_host_submissions(self):
        execution = LiveExecution(HW)
        execution.advance_host(100)
        execution.submit([work('later_compute', 20)])
        execution.advance_host(100)
        execution.submit_enqueued([dict(event_id='earlier_comm', kind='collective',
            duration_us=10, duration_provenance='test', stream='comm', host_time_us=0)])
        rows = {row['event_id']: row for row in execution.document()['timeline']}
        self.assertEqual(rows['earlier_comm']['end_us'], 10)
        self.assertEqual(rows['later_compute']['start_us'], 100)

    def test_real_python_cuda_controls_wait_on_live_backend(self):
        import torch
        import torch._dynamo
        from titan_sim.capture import TensorCapture
        from titan_sim.fake_device import FakeCudaRuntime
        capture = TensorCapture(modeled_device='cuda')
        execution = LiveExecution(HW)
        properties = dict(name='test', major=10, minor=0, total_memory=1000, multi_processor_count=1)
        with FakeCudaRuntime(capture, properties=properties, execution=execution):
            first = torch.cuda.Event(enable_timing=True)
            last = torch.cuda.Event(enable_timing=True)
            first.record()
            capture._event('compute', 'kernel', duration_us=1000, duration_provenance='test')
            execution.advance_host(50)
            last.record()
            self.assertFalse(last.query())
            self.assertEqual(execution.host_time_us, 50)
            torch.cuda.current_stream().wait_event(last)
            self.assertEqual(execution.host_time_us, 50)
            last.synchronize()
            self.assertTrue(last.query())
            self.assertEqual(first.elapsed_time(last), 1.0)
            self.assertEqual(execution.host_time_us, 1000)
        self.assertIsNone(capture._host_time_provider)
        self.assertTrue(execution.document()['live_execution']['python_control_interception'])
        self.assertFalse(execution.document()['live_execution']['native_cuda_interception'])

    def test_capture_blocking_operation_advances_host_before_return(self):
        import torch._dynamo
        from titan_sim.capture import TensorCapture
        from titan_sim.fake_device import FakeCudaRuntime
        capture = TensorCapture(modeled_device='cuda')
        execution = LiveExecution(HW)
        properties = dict(name='test', major=10, minor=0, total_memory=1000, multi_processor_count=1)
        with FakeCudaRuntime(capture, properties=properties, execution=execution):
            event = capture._event('compute', 'copy', duration_us=75, duration_provenance='test')
            capture._event('host_sync', 'blocking_host_device_copy', deps=[event['event_id']], sync_scope='event')
            self.assertEqual(execution.host_time_us, 75)
            self.assertEqual(len(execution.waits), 1)

    def test_late_optimizer_storage_classification_reaches_live_memory(self):
        import torch
        import torch._dynamo
        from torch._subclasses.fake_tensor import FakeTensorMode
        from titan_sim.capture import TensorCapture
        from titan_sim.fake_device import FakeCudaRuntime
        capture = TensorCapture(modeled_device='cuda')
        execution = LiveExecution(HW)
        properties = dict(name='test', major=10, minor=0, total_memory=1000, multi_processor_count=1)
        with FakeCudaRuntime(capture, properties=properties, execution=execution), FakeTensorMode():
            value = torch.empty(2)
            capture.register_tensor(value)
            self.assertEqual(execution.document()['summary']['peak_allocated_bytes'], 8)
            capture.register_tensor(value, category='optimizer', modeled_device='cpu')
            self.assertEqual(execution.document()['summary']['peak_allocated_bytes'], 0)
            self.assertEqual(execution.events[0]['category'], 'optimizer')
