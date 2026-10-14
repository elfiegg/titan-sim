import json
import os
import subprocess
import sys
import unittest
from unittest.mock import patch
from titan_sim.native_guard import probe_native_metadata_guard


class NativeGuardTests(unittest.TestCase):
    def test_missing_compiler_is_explicit(self):
        with patch('titan_sim.native_guard.shutil.which', return_value=None):
            result = probe_native_metadata_guard()
        self.assertEqual(result['status'], 'blocked')
        self.assertIn('compiler', result['reason'])

    def test_probe_runs_in_child_with_hidden_gpus_and_optin(self):
        seen = {}
        def run(command, **kwargs):
            seen.update(kwargs)
            return subprocess.CompletedProcess(command, 0, json.dumps({'status': 'blocked', 'reason': 'fixture'}), '')
        with patch('titan_sim.native_guard.shutil.which', return_value='/compiler'), patch('titan_sim.native_guard.subprocess.run', side_effect=run):
            result = probe_native_metadata_guard(allow_cuda_build=True)
        self.assertEqual(seen['env']['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(seen['env']['TITAN_SIM_ALLOW_CUDA_GUARD_REPLACEMENT'], '1')
        self.assertEqual(result['status'], 'blocked')

    def test_inherited_replacement_optin_does_not_enable_default_probe(self):
        seen = {}
        def run(command, **kwargs):
            seen.update(kwargs)
            return subprocess.CompletedProcess(command, -6, '', 'native abort')
        with patch.dict(os.environ, {'TITAN_SIM_ALLOW_CUDA_GUARD_REPLACEMENT': '1'}), patch('titan_sim.native_guard.shutil.which', return_value='/compiler'), patch('titan_sim.native_guard.subprocess.run', side_effect=run):
            result = probe_native_metadata_guard()
        self.assertNotIn('TITAN_SIM_ALLOW_CUDA_GUARD_REPLACEMENT', seen['env'])
        self.assertEqual(result['returncode'], -6)
        self.assertEqual(result['status'], 'blocked')

    def test_direct_child_refuses_without_subprocess_optin(self):
        env = os.environ.copy()
        env.pop('TITAN_SIM_NATIVE_METADATA_PROBE', None)
        result = subprocess.run([sys.executable, '-m', 'titan_sim.native_guard', '--child'], env=env,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Missing isolated-child opt-in', result.stdout)
