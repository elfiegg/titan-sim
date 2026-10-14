"""Run a capture/measurement entry point with explicit import compatibility."""
import argparse
import importlib
import sys
from .compat import nvidia2609_compat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['gpu_compare', 'project_engine', 'torchtitan_adapter'])
    args, remainder = parser.parse_known_args()
    with nvidia2609_compat() as compatibility:
        print('Import compatibility:', compatibility, flush=True)
        module = importlib.import_module('titan_sim.' + args.mode)
        module.IMPORT_COMPATIBILITY = compatibility
        sys.argv = [sys.argv[0], *remainder]
        module.main()


if __name__ == '__main__':
    main()
