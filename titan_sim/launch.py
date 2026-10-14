"""Run a capture/measurement entry point with explicit import compatibility."""
import argparse
import importlib
import sys
from contextlib import nullcontext
from .compat import nvidia2609_compat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['gpu_compare', 'project_engine', 'torchtitan_adapter'])
    parser.add_argument('--legacy-import-compat', action='store_true', help='Explicit opt-in for the superseded newer TorchTitan revision only')
    args, remainder = parser.parse_known_args()
    context = nvidia2609_compat() if args.legacy_import_compat else nullcontext({'policy': 'native APIs; no import or FSDP argument shims'})
    with context as compatibility:
        print('Import compatibility:', compatibility, flush=True)
        module = importlib.import_module('titan_sim.' + args.mode)
        module.IMPORT_COMPATIBILITY = compatibility
        sys.argv = [sys.argv[0], *remainder]
        module.main()


if __name__ == '__main__':
    main()
