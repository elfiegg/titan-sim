"""Regression for NVIDIA wheel metadata differing from torch.__version__."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


class ValidationEnvironmentTests(unittest.TestCase):
    def test_constraint_preserves_installed_distribution_not_runtime_label(self):
        script = Path(__file__).resolve().parents[1] / 'scripts' / 'validate_container.sh'
        command = next(line for line in script.read_text().splitlines()
                       if line.startswith('python -c ') and 'torch-constraint.txt' in line)
        expression = shlex.split(command)[2]
        # These distinct values were read from the same cached NVIDIA 26.09 image.
        metadata_version = '2.14.0a0+b2c75dd062.nv26.9.68203377'
        runtime_version = '2.14.0a0+b2c75dd062.nv26.09'
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            distribution = directory / f'torch-{metadata_version}.dist-info'
            distribution.mkdir()
            (distribution / 'METADATA').write_text(
                f'Metadata-Version: 2.4\nName: torch\nVersion: {metadata_version}\n')
            package = directory / 'torch'
            package.mkdir()
            (package / '__init__.py').write_text(f'__version__ = {runtime_version!r}\n')
            # -S removes the host's packages; this imports only stdlib metadata
            # and the tiny fixture, never PyTorch or a CUDA runtime.
            environment = {**os.environ, 'PYTHONPATH': temporary}
            output = subprocess.check_output([sys.executable, '-S', '-c', expression],
                                             env=environment, text=True)
        self.assertEqual(output, f'torch==={metadata_version}\n')
        self.assertNotEqual(output, f'torch==={runtime_version}\n')


if __name__ == '__main__':
    unittest.main()
