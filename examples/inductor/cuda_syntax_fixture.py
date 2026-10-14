# HANDWRITTEN SYNTAX FIXTURE. This is not captured GPU compiler output.
# Used only to validate static CUDA guard, stream, layout and alias parsing.
import torch

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4), (4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        stream0 = get_raw_stream(0)
        triton_copy.run(arg0_1, buf0, 16, grid=grid(16), stream=stream0)
        del arg0_1
        buf1 = reinterpret_tensor(buf0, (2, 4), (4, 1), 4)
        del buf0
        triton_scale.run(buf1, 8, grid=grid(8), stream=stream0)
        return (buf1,)

def get_args():
    arg0_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    return [arg0_1]
