import copy
import unittest
from titan_sim.quantization import packed_layout, lower_quantized_training
from titan_sim.runtime import simulate

HW = {'peak_flops': {'float32': 1e6, 'mxfp8': 2e6}, 'memory_bandwidth_bytes_s': 1e6,
      'memory_capacity_bytes': 1e9}


def layout(fmt='mxfp8', shape=None):
    return {'format': fmt, 'shape': shape or [3, 33], 'orientations': ['rowwise'],
            'layout': 'tensorwise' if fmt == 'fp8' else 'rowwise_block'}


def training():
    return {'schema_version': 1, 'id': 'q', 'provenance': 'unit-test declared layout', 'tensors': [
        {'name': 'weight', 'layout': layout(),
         'states': {'parameter': 'bfloat16', 'master_parameter': 'float32', 'gradient': 'bfloat16',
                    'first_moment': 'float32', 'second_moment': 'float32'},
         'conversion': {'source_state': 'parameter', 'execution_dtype': 'float32', 'flops': 297,
                        'scratch_bytes': 32, 'scratch_read_bytes': 32, 'scratch_write_bytes': 32,
                        'provenance': 'declared conversion work and temporary storage'}}]}


class QuantizationTests(unittest.TestCase):
    def test_fp8_tensor_scale(self):
        result = packed_layout(layout('fp8'))
        self.assertEqual(result['total_bytes'], 99 + 4)
        self.assertIsNone(result['block_elements'])

    def test_mxfp8_blocks_and_both_orientations(self):
        spec = layout()
        spec['orientations'].append('columnwise')
        result = packed_layout(spec)
        row, col = result['components']
        self.assertEqual(row['padded_shape'], [3, 64])
        self.assertEqual(row['payload_bytes'], 192)
        self.assertEqual(row['scale_bytes'], 6)
        self.assertEqual(col['padded_shape'], [33, 32])
        self.assertEqual(col['scale_bytes'], 33)
        self.assertEqual(result['total_bytes'], 198 + 1089)

    def test_nvfp4_odd_dimensions_and_padded_scale_tiles(self):
        spec = layout('nvfp4', [3, 17])
        spec.update(scale_row_alignment=128, scale_column_alignment=4)
        result = packed_layout(spec)['components'][0]
        self.assertEqual(result['payload_bytes'], 3 * 32 // 2)
        self.assertEqual(result['scale_shape'], [128, 4])
        self.assertEqual(result['total_bytes'], 48 + 512 + 4)

    def test_nvfp4_weight_two_dimensional_blocks(self):
        spec = layout('nvfp4', [17, 17])
        spec['layout'] = 'tile_16x16'
        spec['scale_storage'] = 'replicated_rows'
        result = packed_layout(spec)
        self.assertEqual(result['block_geometry'], [16, 16])
        self.assertEqual(result['components'][0]['padded_shape'], [32, 32])
        self.assertEqual(result['total_bytes'], 512 + 64 + 4)
        spec['scale_storage'] = 'compact_tiles'
        self.assertEqual(packed_layout(spec)['total_bytes'], 512 + 4 + 4)

    def test_training_state_is_independent_of_compute_bits(self):
        document = lower_quantized_training(training())
        sizes = document['buffers']
        self.assertEqual(sizes['q:weight:master_parameter']['size_bytes'], 99 * 4)
        self.assertEqual(sizes['q:weight:first_moment']['size_bytes'], 99 * 4)
        self.assertEqual(sizes['q:weight:gradient']['size_bytes'], 99 * 2)
        conversion = next(e for e in document['events'] if e['kind'] == 'compute')
        self.assertEqual(conversion['bytes_read'], 198 + 32)
        self.assertEqual(conversion['bytes_write'], 198 + 32)
        simulation = simulate(document['events'], HW)
        expected = 99 * 16 + 198
        self.assertEqual(simulation['summary']['peak_allocated_bytes'], expected + 32)
        self.assertEqual(simulation['summary']['per_rank']['0']['final_allocated_bytes'], expected)
        scratch = next(s for s in simulation['storage_lifetimes'] if s['category'] == 'quantization_workspace')
        self.assertGreater(scratch['reclaim_us'], 0)

    def test_unknown_layout_and_incomplete_state_rejected(self):
        spec = layout()
        spec['layout'] = 'te_magic'
        with self.assertRaisesRegex(ValueError, 'layout'):
            packed_layout(spec)
        spec = training()
        del spec['tensors'][0]['states']['first_moment']
        with self.assertRaisesRegex(ValueError, 'states'):
            lower_quantized_training(spec)

    def test_invalid_padding_orientation_and_scratch(self):
        for update in ({'data_alignment_elements': 0}, {'orientations': ['diagonal']},
                       {'orientations': ['rowwise', 'rowwise']}, {'shape': [True, 3]}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                packed_layout({**layout(), **update})
        spec = training()
        spec['tensors'][0]['conversion']['scratch_bytes'] = 0
        with self.assertRaisesRegex(ValueError, 'scratch'):
            lower_quantized_training(spec)

    def test_master_can_be_explicitly_absent(self):
        spec = training()
        spec['tensors'][0]['states']['master_parameter'] = None
        document = lower_quantized_training(spec)
        self.assertNotIn('q:weight:master_parameter', document['buffers'])
        broken = copy.deepcopy(spec)
        broken['tensors'][0]['conversion']['source_state'] = 'master_parameter'
        with self.assertRaisesRegex(ValueError, 'source_state'):
            lower_quantized_training(broken)
