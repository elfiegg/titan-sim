from pathlib import Path

path = Path('/workspace/DeepEP/setup.py')
s = path.read_text()
s = s.replace('"-std=c++17"', '"-std=c++20"')
s = s.replace("include_dirs = ['csrc/']", "include_dirs = ['csrc/', '/usr/local/cuda/include/cccl/']")
path.write_text(s)

path = Path('/workspace/DeepEP/csrc/hybrid_ep/allocator/allocator.cu')
s = path.read_text()
if 'titansim_log_allocation' not in s:
    helper = r'''
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

// Optional diagnostic metadata only; do not retain GPU allocations or synchronize.
static void titansim_log_allocation(const char* action, void* ptr, size_t requested, size_t physical) {
  const char* dir = std::getenv("TITANSIM_DEEPEP_MEMORY_LOG_DIR");
  if (!dir) return;
  const char* rank = std::getenv("RANK");
  char filename[4096];
  std::snprintf(filename, sizeof(filename), "%s/rank%s-pid%d.jsonl", dir, rank ? rank : "unknown", getpid());
  FILE* file = std::fopen(filename, "a");
  if (!file) return;
  std::fprintf(file, "{\"event\":\"%s\",\"address\":\"%p\",\"requested_bytes\":%zu,\"physical_bytes\":%zu}\n", action, ptr, requested, physical);
  std::fclose(file);
}
'''
    s = s.replace('#include "allocator.cuh"', '#include "allocator.cuh"\n' + helper, 1)
    old = '    CU_CHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &access_desc, 1));'
    assert s.count(old) == 2
    s = s.replace(old, old + '\n    titansim_log_allocation("allocate_fabric", *ptr, size_raw, size);', 1)
    old = '    CUDA_CHECK(cudaMalloc(ptr, size_raw));'
    assert s.count(old) == 1
    s = s.replace(old, old + '\n    titansim_log_allocation("allocate_cuda", *ptr, size_raw, size_raw);', 1)
    old = 'void ExtendedMemoryAllocator::free(void* ptr) {\n  if (ptr == nullptr) {\n    return;\n  }'
    assert s.count(old) == 1
    s = s.replace(old, old + '\n  titansim_log_allocation("free", ptr, 0, 0);', 1)
    path.write_text(s)
