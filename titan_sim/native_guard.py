"""Subprocess-only native CUDA DeviceGuard feasibility diagnostic.

This is intentionally NOT a FakeCudaRuntime backend. It never installs a guard
in the caller. CUDA-enabled builds require explicit subprocess-only opt-in.
The diagnostic can establish forward edge construction, not streaming autograd
support. A native metadata backend additionally needs AcceleratorHooks and
canonical-stream support in the pinned PyTorch build.
"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile


def probe_native_metadata_guard(*, timeout=45, allow_cuda_build=False):
    """Compile and exercise the strict guard in a fresh CPU-only child.

    Return structured phases and native errors, including a precise unsupported
    reason. No output from this probe constitutes a GPU timing measurement.
    """
    compiler = shutil.which(os.environ.get('CXX', 'c++'))
    if compiler is None:
        return {'status': 'blocked', 'reason': 'A C++20 compiler is required for the native metadata guard diagnostic'}
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES='', TITAN_SIM_NATIVE_METADATA_PROBE='isolated-child-v1')
    environment['PYTHONPATH'] = os.pathsep.join(sys.path)
    environment.pop('TITAN_SIM_ALLOW_CUDA_GUARD_REPLACEMENT', None)
    if allow_cuda_build:
        environment['TITAN_SIM_ALLOW_CUDA_GUARD_REPLACEMENT'] = '1'
    with tempfile.TemporaryDirectory(prefix='titan-native-metadata-') as directory:
        command = [sys.executable, '-m', 'titan_sim.native_guard', '--child',
                   '--compiler', compiler, '--build-directory', directory]
        try:
            child = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            return {'status': 'blocked', 'reason': 'Native metadata diagnostic timed out',
                    'stderr': str(error), 'cpu_only': True}
    diagnostics = []
    for line in child.stdout.splitlines():
        try:
            diagnostics.append(json.loads(line))
        except ValueError:
            continue
    result = next((x for x in reversed(diagnostics) if isinstance(x, dict) and 'status' in x),
                  {'status': 'blocked', 'reason': 'Native metadata diagnostic exited without a structured result'})
    return {**result, 'returncode': child.returncode, 'diagnostics': diagnostics,
            'stderr': child.stderr[-12000:], 'cpu_only': True,
            'scope': 'isolated hidden-GPU native guard feasibility diagnostic; no runtime backend installed',
            'allow_cuda_build': allow_cuda_build}


def _child(compiler, build_directory):
    import ctypes
    import torch
    import torch._dynamo
    from torch._subclasses.fake_tensor import FakeTensorMode
    from .capture import TensorCapture
    from .fake_device import FakeCudaRuntime

    def emit(**record):
        print(json.dumps(record), flush=True)
    if os.environ.get('TITAN_SIM_NATIVE_METADATA_PROBE') != 'isolated-child-v1':
        emit(status='blocked', reason='Missing isolated-child opt-in')
        return
    if torch.version.cuda is not None and os.environ.get('TITAN_SIM_ALLOW_CUDA_GUARD_REPLACEMENT') != '1':
        emit(status='blocked', reason='Refusing native guard registration in CUDA-enabled PyTorch', torch=torch.__version__)
        return
    root = Path(torch.__file__).parent
    include = root / 'include'
    library = Path(build_directory) / ('metadata_guard.dylib' if sys.platform == 'darwin' else 'metadata_guard.so')
    args = [compiler, '-std=c++20', '-shared', '-fPIC', '-O0',
            str(Path(__file__).parent / 'native' / 'metadata_cuda_guard.cpp'),
            '-I' + str(include), '-I' + str(include / 'torch' / 'csrc' / 'api' / 'include'),
            '-L' + str(root / 'lib'), '-Wl,-rpath,' + str(root / 'lib'), '-ltorch_cpu', '-lc10', '-o', str(library)]
    if sys.platform != 'darwin':
        args.append('-D_GLIBCXX_USE_CXX11_ABI=' + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)))
    build = subprocess.run(args, text=True, capture_output=True)
    if build.returncode:
        emit(status='blocked', reason='Native metadata guard build failed', error=build.stderr[-10000:])
        return
    native = ctypes.CDLL(str(library))
    native.titan_metadata_error.restype = ctypes.c_char_p
    accelerator = native.titan_metadata_accelerator()
    # c10 DeviceType enum: CUDA=1, MPS=13. Keep raw enum for newer builds.
    accelerator_name = {-1: 'none', 1: 'cuda', 13: 'mps'}.get(accelerator, f'device_type_{accelerator}')
    emit(phase='native_build', torch=torch.__version__, torch_git=torch.version.git_version,
         selected_accelerator=accelerator_name)
    if native.titan_metadata_install():
        emit(status='blocked', reason='Native guard registration rejected',
             error=native.titan_metadata_error().decode())
        return
    try:
        record = _exercise(native, accelerator, accelerator_name, emit)
    except Exception as error:
        record = dict(status='blocked', reason='Native metadata guard exercise failed',
                      error_type=type(error).__name__, error=str(error))
    finally:
        restored = native.titan_metadata_restore() == 0
        emit(phase='native_restore', original_guard_restored=restored)
    emit(**record)


def _exercise(native, accelerator, accelerator_name, emit):
    import ctypes
    import torch
    from unittest.mock import patch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from .capture import TensorCapture
    from .fake_device import FakeCudaRuntime
    native.titan_metadata_set_stream.argtypes = [ctypes.c_int64]
    native.titan_metadata_edge_count.restype = ctypes.c_int64
    if native.titan_metadata_self_test():
        raise RuntimeError('Native virtual-stream/allocator self-test failed: ' + native.titan_metadata_error().decode())
    emit(phase='native_guard', virtual_stream_exchange=True, event_edges_recorded=True,
         real_generic_cuda_allocator_trapped=True, completion_queries_rejected=True)
    properties = dict(name='metadata diagnostic', major=10, minor=0, total_memory=1, multi_processor_count=1)
    real_allocation_rejected = False
    try:
        torch.empty(1, device='cuda')
    except Exception:
        real_allocation_rejected = True
    if not real_allocation_rejected:
        raise RuntimeError('Real CUDA allocation unexpectedly succeeded')
    emit(phase='allocation_safety', real_cuda_allocation_rejected=True)
    phase = 'tensor_creation'
    try:
        with FakeCudaRuntime(TensorCapture(), properties=properties) as runtime:
            original_set_stream = runtime.set_stream
            def set_stream(stream):
                if stream is not None and native.titan_metadata_set_stream(stream.stream_id):
                    raise RuntimeError('Native stream exchange failed')
                original_set_stream(stream)
            runtime.set_stream = set_stream
            set_stream(runtime._current)
            with patch.object(torch.cuda, 'set_stream', set_stream), FakeTensorMode(allow_fallback_kernels=False):
                value = torch.empty(2, device='cuda', requires_grad=True)
                emit(phase=phase, complete=True)
                phase = 'forward'
                loss = value.square().sum()
                emit(phase=phase, complete=True)
                phase = 'backward'
                loss.backward()
                emit(phase=phase, complete=True)
                phase = 'multistream_autograd'
                first, second = torch.cuda.Stream(), torch.cuda.Stream()
                with torch.cuda.stream(first):
                    other = torch.empty(2, device='cuda', requires_grad=True)
                    squared = other.square()
                with torch.cuda.stream(second):
                    second.wait_stream(first)
                    other_loss = squared.sum()
                torch.cuda.current_stream().wait_stream(second)
                before_edges = native.titan_metadata_edge_count()
                other_loss.backward()
                after_edges = native.titan_metadata_edge_count()
                emit(phase=phase, complete=True, native_backward_event_edges=after_edges-before_edges)
    except Exception as error:
        reason = ('DeviceGuard registration alone does not provide a CUDA accelerator backend: '
                  'Node::stream selects the compiled accelerator and cannot select CUDA input streams '
                  f'when getAccelerator() returns {accelerator_name}.') if accelerator != 1 else (
                  'Native guard diagnostic encountered unsupported autograd/runtime behavior')
        return dict(status='blocked', reason=reason, failed_phase=phase,
             error_type=type(error).__name__, error=str(error),
             selected_accelerator=accelerator_name, torch=torch.__version__, torch_git=torch.version.git_version,
             next_step='Provide a pinned PyTorch metadata accelerator/hooks implementation with canonical CUDA streams and explicit event dependencies; a DeviceGuard plugin alone may be insufficient.')
    return dict(status='complete', torch=torch.__version__,
                selected_accelerator=accelerator_name,
                limitation='Only elementary default/multistream autograd metadata verified; native event edges are diagnostic and not yet integrated with the runtime scheduler.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--allow-cuda-build', action='store_true')
    parser.add_argument('--compiler')
    parser.add_argument('--build-directory')
    args = parser.parse_args()
    if args.child:
        _child(args.compiler, args.build_directory)
    else:
        print(json.dumps(probe_native_metadata_guard(allow_cuda_build=args.allow_cuda_build), indent=2))
