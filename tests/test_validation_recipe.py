"""Optional integration check against the actual pinned upstream configuration."""
import os
import unittest


@unittest.skipUnless(os.environ.get('TORCHTITAN_ROOT'), 'set TORCHTITAN_ROOT for upstream recipe integration')
class ValidationRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from titan_sim.recipes import import_torchtitan
        import_torchtitan(os.environ['TORCHTITAN_ROOT'])

    def test_production_uses_full_checkpoint_policy_and_requested_parallelism(self):
        from titan_sim.recipes import load_validation_recipe
        from titan_sim.provenance import execution_policy
        from torchtitan.distributed.activation_checkpoint import FullAC
        config = load_validation_recipe()
        self.assertIsInstance(config.activation_checkpoint, FullAC.Config)
        self.assertEqual(config.parallelism.expert_parallel_degree, 64)
        self.assertEqual(config.parallelism.pipeline_parallel_degree, 1)
        self.assertEqual(config.parallelism.tensor_parallel_degree, 1)
        self.assertEqual(config.parallelism.data_parallel_shard_degree, -1)
        self.assertEqual(config.training.max_context_length, 16384)
        self.assertEqual(config.training.num_tokens_per_microbatch_per_dp_rank, 65536)
        self.assertEqual(config.compile.regions, ['gated_rmsnorm', 'loss', 'swiglu'])
        policy = execution_policy(config, routing_scenario='strict')
        self.assertEqual(len(policy['policy_fingerprint']), 64)
        self.assertIn('activation_checkpoint', policy['execution_policy'])
        self.assertIn('compile', policy['execution_policy'])

    def test_debug_explicitly_changes_expert_degree_and_batch_size(self):
        from titan_sim.recipes import load_validation_recipe
        config = load_validation_recipe(flavor='debugmodel', seq_len=128, tokens=1024, expert_parallel_degree=1)
        self.assertEqual(config.parallelism.expert_parallel_degree, 1)
        self.assertTrue(config.training.disable_cuda_graphs)
        self.assertEqual(config.training.num_tokens_per_microbatch_per_dp_rank, 1024)

    def test_upstream_resolves_dense_256_and_expert_fsdp_4(self):
        from torchtitan.distributed.parallelism_context import ParallelismContext
        from titan_sim.recipes import resolved_validation_topology
        context = ParallelismContext(dp_replicate=1, dp_shard=-1, cp=1, tp=1, pp=1,
                                     ep=64, world_size=256, enable_sequence_parallel=True)
        self.assertEqual(resolved_validation_topology(context, world_size=256, expert_parallel_degree=64),
                         {'dense_fsdp': 256, 'expert_parallel': 64, 'expert_fsdp': 4, 'input_data_parallel': 256})
