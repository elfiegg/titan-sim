import copy
import importlib.util
import unittest
from titan_sim.compiled import artifact_digest, export_cpu_graph, lower_compiled_manifest
from titan_sim.runtime import simulate
from titan_sim.quantization import lower_quantized_training
from tests.test_quantization import training

HW = {'peak_flops': {'float32': 1e6, 'mxfp8': 2e6}, 'memory_bandwidth_bytes_s': 1e6,
      'memory_capacity_bytes': 1000000000}


def kernel(kid, reads, writes, ops=None):
    return {'id': kid, 'name': 'fused_' + kid, 'dtype': 'float32', 'flops': 100,
            'bytes_read': 16, 'bytes_write': 16, 'reads': reads, 'writes': writes,
            'logical_ops': ops or [kid], 'provenance': 'declared fused boundary traffic'}


def manifest():
    return {'schema_version': 1, 'id': 'c', 'plan_source': 'declared',
            'provenance': 'unit-test serial compiled allocation plan',
            'external_storages': {'x': 16},
            'slots': [{'id': 'scratch', 'size_bytes': 16, 'category': 'workspace'}],
            'buffers': [{'id': 'a', 'slot': 'scratch', 'size_bytes': 16, 'offset_bytes': 0,
                         'first_kernel': 'one', 'last_kernel': 'two'},
                        {'id': 'b', 'slot': 'scratch', 'size_bytes': 16, 'offset_bytes': 0,
                         'first_kernel': 'three', 'last_kernel': 'three'}],
            'kernels': [kernel('one', ['x'], ['a'], ['mm', 'relu']),
                        kernel('two', ['a'], ['x']), kernel('three', ['x'], ['b'])],
            'graph': {'mode': 'none'}}


def external_alloc():
    return {'event_id': 'allocate_x', 'kind': 'allocate', 'storage_id': 'x', 'size_bytes': 16}


class CompiledTests(unittest.TestCase):
    def test_fusion_traffic_and_physical_reuse(self):
        document = lower_compiled_manifest(manifest())
        allocations = [e for e in document['events'] if e['kind'] == 'allocate']
        self.assertEqual(len(allocations), 1)
        self.assertEqual(sum(e.get('bytes_read', 0) for e in document['events']), 48)
        result = simulate([external_alloc()] + document['events'], HW)
        self.assertEqual(sum(e['flops'] for e in result['timeline']), 300)
        self.assertEqual(result['summary']['peak_allocated_bytes'], 32)
        slot = next(s for s in result['storage_lifetimes'] if s['storage_id'] == 'c:slot:scratch')
        self.assertEqual(slot['reclaim_us'], result['summary']['runtime_us'])

    def test_graph_pool_capture_replay_and_retention(self):
        plan = manifest()
        plan['graph'] = {'mode': 'capture_replay', 'replays': 2,
                         'private_pool': {'id': 'g', 'size_bytes': 64, 'slot_offsets': {'scratch': 0}},
                         'release_pool': False}
        document = lower_compiled_manifest(plan)
        result = simulate([external_alloc()] + document['events'], HW)
        self.assertEqual(sum(e['flops'] for e in result['timeline']), 900)
        self.assertEqual(result['summary']['peak_allocated_bytes'], 80)
        self.assertEqual(result['summary']['per_rank']['0']['final_allocated_bytes'], 80)
        self.assertEqual(len([e for e in document['events'] if e['kind'] == 'allocate']), 1)
        plan['graph']['release_pool'] = True
        result = simulate([external_alloc()] + lower_compiled_manifest(plan)['events'], HW)
        self.assertEqual(result['summary']['per_rank']['0']['final_allocated_bytes'], 16)

    def test_overlapping_reuse_requires_alias_and_alias_fits(self):
        plan = manifest()
        plan['buffers'][1].update(first_kernel='two', last_kernel='three')
        with self.assertRaisesRegex(ValueError, 'overlapping live'):
            lower_compiled_manifest(plan)
        plan['buffers'][0]['last_kernel'] = 'three'
        plan['buffers'][1]['alias_of'] = 'a'
        document = lower_compiled_manifest(plan)
        self.assertEqual(len([e for e in document['events'] if e['kind'] == 'allocate']), 1)
        plan['buffers'][1]['offset_bytes'] = 1
        with self.assertRaises(ValueError):
            lower_compiled_manifest(plan)

    def test_unknown_storage_lifetime_and_graph_policy(self):
        plan = manifest()
        plan['kernels'][0]['reads'] = ['missing']
        with self.assertRaisesRegex(ValueError, 'unknown buffer'):
            lower_compiled_manifest(plan)
        plan = manifest()
        plan['kernels'][0]['writes'] = ['b']
        with self.assertRaisesRegex(ValueError, 'outside its lifetime'):
            lower_compiled_manifest(plan)
        plan = manifest()
        plan['graph'] = {'mode': 'infer_pool'}
        with self.assertRaisesRegex(ValueError, 'graph mode'):
            lower_compiled_manifest(plan)

    def test_artifact_mismatch_and_cannot_claim_fx_gpu_allocations(self):
        plan = manifest()
        artifact = {'kind': 'torch_export_fx', 'nodes': [{'name': name} for name in ['mm', 'relu', 'two', 'three']]}
        plan['artifact_sha256'] = artifact_digest(artifact)
        lower_compiled_manifest(plan, artifact=artifact)
        with self.assertRaisesRegex(ValueError, 'artifact mismatch'):
            lower_compiled_manifest(plan, artifact={**artifact, 'changed': True})
        plan['plan_source'] = 'compiler_artifact'
        with self.assertRaisesRegex(ValueError, 'not a GPU'):
            lower_compiled_manifest(plan, artifact=artifact)

    def test_compiler_artifact_binds_plan(self):
        plan = manifest()
        plan['plan_source'] = 'compiler_artifact'
        artifact = {'kind': 'gpu_compiler_allocation_plan', 'plan': copy.deepcopy({
            key: plan.get(key, {}) for key in ('external_storages', 'slots', 'buffers', 'kernels', 'graph')})}
        plan['artifact_sha256'] = artifact_digest(artifact)
        lower_compiled_manifest(plan, artifact=artifact)
        plan['kernels'][0]['flops'] = 1
        with self.assertRaisesRegex(ValueError, 'allocation plan mismatch'):
            lower_compiled_manifest(plan, artifact=artifact)

    def test_quantized_training_buffers_feed_fused_region(self):
        quantized = lower_quantized_training(training())
        plan = manifest()
        plan['external_storages'] = {sid: b['size_bytes'] for sid, b in quantized['buffers'].items()}
        weight_ids = ['q:weight:rowwise:payload', 'q:weight:rowwise:scale']
        for k in plan['kernels']:
            k['reads'] = weight_ids
            k['writes'] = ['a'] if k['id'] in ('one', 'two') else ['b']
        plan['kernels'][0]['dtype'] = 'mxfp8'
        compiled = lower_compiled_manifest(plan)
        result = simulate(quantized['events'] + compiled['events'], HW)
        self.assertEqual(sum(e['flops'] for e in result['timeline']), 597)
        self.assertEqual(result['summary']['per_rank']['0']['final_allocated_bytes'], 99 * 16 + 198)

    def test_graph_quantized_allocator_counts_pool_once(self):
        quantized = lower_quantized_training(training())
        plan = manifest()
        plan['external_storages'] = {sid: b['size_bytes'] for sid, b in quantized['buffers'].items()}
        for k in plan['kernels']:
            k['reads'] = ['q:weight:rowwise:payload', 'q:weight:rowwise:scale']
            k['writes'] = ['a'] if k['id'] in ('one', 'two') else ['b']
        plan['graph'] = {'mode': 'capture_replay', 'replays': 2,
                         'private_pool': {'id': 'g', 'size_bytes': 64, 'slot_offsets': {'scratch': 0}},
                         'release_pool': False}
        compiled = lower_compiled_manifest(plan)
        events = quantized['events'] + compiled['events']
        result = simulate(events, HW, allocator={'policy': 'caching', 'alignment_bytes': 1, 'segment_bytes': 1})
        self.assertEqual(result['allocator']['summary']['peak_reserved_bytes'], 99 * 16 + 198 + 32 + 64)
        pools = [a for a in result['allocator']['allocations'] if a['pool'] == 'c:pool:g']
        self.assertEqual(len(pools), 1)
        self.assertEqual(pools[0]['size_bytes'], 64)
        self.assertFalse(result['allocator']['summary']['capacity_exceeded'])

    def test_retained_output_rejects_later_overwrite(self):
        plan = manifest()
        plan['retained_buffers'] = ['a']
        with self.assertRaisesRegex(ValueError, 'overlapping live'):
            lower_compiled_manifest(plan)
        plan['retained_buffers'] = ['b']
        result = lower_compiled_manifest(plan)
        self.assertEqual(result['metadata']['retained_storage_ids'], ['c:slot:scratch'])

    def test_calibration_keys_bind_compiler_context(self):
        from titan_sim.costs import calibration_key
        plan = manifest()
        plan['calibration_context'] = {'backend': 'test', 'compiler_version': '1'}
        first = next(e for e in lower_compiled_manifest(plan)['events'] if e['kind'] == 'compute')
        plan['calibration_context']['compiler_version'] = '2'
        second = next(e for e in lower_compiled_manifest(plan)['events'] if e['kind'] == 'compute')
        self.assertNotEqual(calibration_key(first), calibration_key(second))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch is optional')
    def test_actual_export_graph_provenance(self):
        artifact = export_cpu_graph()
        self.assertEqual(artifact['kind'], 'torch_export_fx')
        self.assertEqual(len(artifact['nodes']), 2)
        plan = manifest()
        plan['slots'], plan['buffers'] = [], []
        plan['kernels'] = [kernel('fused', ['x'], ['x'], [node['name'] for node in artifact['nodes']])]
        plan['artifact_sha256'] = artifact_digest(artifact)
        result = lower_compiled_manifest(plan, artifact=artifact)
        self.assertEqual(result['metadata']['plan_source'], 'declared')
        plan['kernels'][0]['logical_ops'] = ['invented']
        with self.assertRaisesRegex(ValueError, 'graph nodes'):
            lower_compiled_manifest(plan, artifact=artifact)
