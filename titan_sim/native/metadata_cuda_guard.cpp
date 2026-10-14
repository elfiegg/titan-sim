// Isolated metadata subprocess experiment. No numerical kernels or CUDA hooks.
#include <ATen/DeviceAccelerator.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/core/Allocator.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {
thread_local std::string error;
thread_local c10::StreamId current_stream = 0;
std::atomic<int64_t> next_stream{1024};
std::mutex edges_mutex;
struct Edge { int64_t producer, consumer, version; };
std::vector<Edge> edges;
struct Event { int64_t stream = 0, version = 0; };
const c10::impl::DeviceGuardImplInterface* previous_guard = nullptr;
c10::Allocator* previous_allocator = nullptr;
bool installed = false;
void check_device(c10::Device d) {
  TORCH_CHECK(d.type() == c10::DeviceType::CUDA && d.index() == 0,
              "Metadata diagnostic supports only cuda:0");
}
struct RejectAllocator final : c10::Allocator {
  c10::DataPtr allocate(size_t) override {
    TORCH_CHECK(false, "Metadata diagnostic rejects real CUDA allocation");
  }
  void copy_data(void*, const void*, size_t) const override {
    TORCH_CHECK(false, "Metadata diagnostic rejects real CUDA copy");
  }
};
RejectAllocator reject_allocator;
struct MetadataCudaGuard final : c10::impl::DeviceGuardImplInterface {
  c10::DeviceType type() const override { return c10::DeviceType::CUDA; }
  c10::Device getDevice() const override { return {type(), 0}; }
  c10::Device exchangeDevice(c10::Device d) const override {
    check_device(d); return getDevice();
  }
  void setDevice(c10::Device d) const override { check_device(d); }
  void uncheckedSetDevice(c10::Device) const noexcept override {}
  c10::Stream getStream(c10::Device d) const override {
    check_device(d); return c10::Stream(c10::Stream::UNSAFE, d, current_stream);
  }
  c10::Stream getDefaultStream(c10::Device d) const override {
    check_device(d); return c10::Stream(c10::Stream::DEFAULT, d);
  }
  c10::Stream getNewStream(c10::Device d, int priority=0) const override {
    check_device(d);
    TORCH_CHECK(priority == 0, "Metadata diagnostic does not model stream priority");
    return c10::Stream(c10::Stream::UNSAFE, d, next_stream++);
  }
  c10::Stream getStreamFromGlobalPool(c10::Device d, bool high=false) const override {
    TORCH_CHECK(!high, "Metadata diagnostic does not model stream priority");
    return getNewStream(d);
  }
  c10::Stream exchangeStream(c10::Stream s) const override {
    check_device(s.device());
    auto previous = getStream(s.device()); current_stream = s.id(); return previous;
  }
  c10::DeviceIndex deviceCount() const noexcept override { return 1; }
  void record(void** ptr, const c10::Stream& s, c10::DeviceIndex index,
              c10::EventFlag) const override {
    check_device(s.device()); TORCH_CHECK(index == 0 || index == -1, "Unknown event device");
    if (!*ptr) *ptr = new Event();
    auto* event = static_cast<Event*>(*ptr);
    event->stream = s.id(); ++event->version;
  }
  void block(void* ptr, const c10::Stream& s) const override {
    check_device(s.device());
    if (!ptr) return;
    auto* event = static_cast<Event*>(ptr);
    std::lock_guard<std::mutex> lock(edges_mutex);
    edges.push_back({event->stream, s.id(), event->version});
  }
  void destroyEvent(void* ptr, c10::DeviceIndex) const noexcept override {
    delete static_cast<Event*>(ptr);
  }
  void recordDataPtrOnStream(const c10::DataPtr& ptr, const c10::Stream&) const override {
    TORCH_CHECK(!ptr.get(), "Metadata diagnostic rejects real storage on virtual streams");
  }
  // Query, synchronization, elapsed-time and native-handle APIs intentionally
  // retain explicit unsupported errors; event edges do not imply completion.
};
MetadataCudaGuard guard;
}
extern "C" const char* titan_metadata_error() { return error.c_str(); }
extern "C" int titan_metadata_accelerator() {
  try { const auto a = at::accelerator::getAccelerator(); return a ? static_cast<int>(*a) : -1; }
  catch (const std::exception& e) { error = e.what(); return -2; }
}
extern "C" int titan_metadata_install() {
  try {
    const char* opt = std::getenv("TITAN_SIM_NATIVE_METADATA_PROBE");
    const char* hidden = std::getenv("CUDA_VISIBLE_DEVICES");
    TORCH_CHECK(opt && std::strcmp(opt, "isolated-child-v1") == 0 && hidden && !*hidden,
                "Native metadata diagnostic requires isolated child and hidden GPUs");
    TORCH_CHECK(!installed, "Metadata guard already installed");
    const bool cuda_build = at::detail::getCUDAHooks().isBuilt();
    const char* allow = std::getenv("TITAN_SIM_ALLOW_CUDA_GUARD_REPLACEMENT");
    TORCH_CHECK(!cuda_build || (allow && std::strcmp(allow, "1") == 0),
                "CUDA-build guard replacement requires explicit diagnostic opt-in");
    try { previous_allocator = c10::GetAllocator(c10::DeviceType::CUDA); }
    catch (...) { previous_allocator = nullptr; }
    c10::SetAllocator(c10::DeviceType::CUDA, &reject_allocator, 255);
    previous_guard = c10::impl::device_guard_impl_registry[static_cast<size_t>(c10::DeviceType::CUDA)].exchange(&guard);
    installed = true; return 0;
  } catch (const std::exception& e) { error = e.what(); return 1; }
}
extern "C" int titan_metadata_restore() {
  if (!installed) return 1;
  c10::impl::device_guard_impl_registry[static_cast<size_t>(c10::DeviceType::CUDA)].store(previous_guard);
  if (previous_allocator) c10::SetAllocator(c10::DeviceType::CUDA, previous_allocator, 255);
  installed = false; return 0;
}
extern "C" int titan_metadata_set_stream(int64_t id) {
  if (!installed || id < 0) return 1;
  current_stream = id; return 0;
}
extern "C" int titan_metadata_self_test() {
  try {
    const auto d = guard.getDevice();
    const auto s = guard.getNewStream(d);
    const auto old = guard.exchangeStream(s);
    TORCH_CHECK(guard.getStream(d) == s, "Virtual stream exchange failed");
    void* event = nullptr;
    guard.record(&event, s, 0, c10::EventFlag::PYTORCH_DEFAULT);
    guard.block(event, old); guard.destroyEvent(event, 0);
    guard.exchangeStream(old);
    try { guard.queryStream(old); return 1; } catch (const c10::Error&) {}
    try { c10::GetAllocator(c10::DeviceType::CUDA)->allocate(1); return 2; } catch (const c10::Error&) {}
    return 0;
  } catch (const std::exception& e) { error = e.what(); return 3; }
}
extern "C" int64_t titan_metadata_edge_count() {
  std::lock_guard<std::mutex> lock(edges_mutex); return edges.size();
}
