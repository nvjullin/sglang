#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <limits>

// Page-granular HiCache transfers for the page_first_direct host layout.
//
// A host pool is a contiguous [host_page, host_layer, page_bytes] byte array; each
// device layer is a separate allocation of [device_page, page_bytes] bytes whose base
// address comes from a per-host-layer pointer table on the GPU. Every transferred page
// is one opaque block of `kPageBytes` on both sides, so the KV pool (page_size token
// rows) and the DSA indexer pool (one index row per KV page) share these kernels.
//
// Index tensors hold token slots on the GPU; page `p` of a transfer starts at
// `indices[p * kIndexStride]`, which is page aligned, so its page number is that slot
// divided by `kIndexStride`.

namespace sglang {

namespace device::hicache_direct {

struct alignas(32) Vec32 {
  uint32_t x[8];
};

SGL_DEVICE uint4 load_streaming_16(const void* ptr) {
  uint4 v;
  asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(ptr)
               : "memory");
  return v;
}

SGL_DEVICE void store_streaming_16(void* ptr, const uint4& v) {
  asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}

// Read-only, L1-bypassing load; host memory is fetched in whole 128 B lines.
SGL_DEVICE uint4 load_host_16(const void* ptr) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.L2::128B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(ptr));
  return v;
}

SGL_DEVICE uint4 load_no_allocate_16(const void* ptr) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(ptr));
  return v;
}

SGL_DEVICE void store_16(void* ptr, const uint4& v) {
  asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

#if SGL_ARCH_BLACKWELL_OR_GREATER
// 256-bit global accesses exist on sm_100 and newer only.
SGL_DEVICE Vec32 load_streaming_32(const void* ptr) {
  Vec32 v;
  asm volatile(
      "ld.global.cs.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=r"(v.x[0]), "=r"(v.x[1]), "=r"(v.x[2]), "=r"(v.x[3]), "=r"(v.x[4]), "=r"(v.x[5]), "=r"(v.x[6]), "=r"(v.x[7])
      : "l"(ptr)
      : "memory");
  return v;
}

SGL_DEVICE void store_streaming_32(void* ptr, const Vec32& v) {
  asm volatile("st.global.cs.v8.u32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::"l"(ptr),
               "r"(v.x[0]),
               "r"(v.x[1]),
               "r"(v.x[2]),
               "r"(v.x[3]),
               "r"(v.x[4]),
               "r"(v.x[5]),
               "r"(v.x[6]),
               "r"(v.x[7])
               : "memory");
}
#endif

template <int64_t kIndexStride>
SGL_DEVICE int64_t page_of(const int64_t* __restrict__ indices, uint32_t page) {
  return __ldg(indices + static_cast<int64_t>(page) * kIndexStride) / kIndexStride;
}

SGL_DEVICE const uint8_t* layer_base(const uint64_t* __restrict__ layer_ptrs, uint32_t layer) {
  return reinterpret_cast<const uint8_t*>(__ldg(layer_ptrs + layer));
}

struct WriteBackParams {
  const uint64_t* __restrict__ layer_ptrs;  // [num_layers] device layer bases
  uint8_t* __restrict__ host;               // [host_page, num_layers, page_bytes]
  const int64_t* __restrict__ device_indices;
  const int64_t* __restrict__ host_indices;
  uint32_t num_layers;
  uint32_t num_pages;
};

struct LoadParams {
  const uint8_t* __restrict__ host;  // [host_page, num_layers, page_bytes]
  uint8_t* __restrict__ device;      // one device layer, [device_page, page_bytes]
  const int64_t* __restrict__ host_indices;
  const int64_t* __restrict__ device_indices;
  uint32_t num_layers;
  uint32_t layer;
};

/// \brief Write-back with one CTA per (page, layer); the layer index varies fastest so
/// neighbouring CTAs fill one host page contiguously.
template <int64_t kPageBytes, int64_t kIndexStride, uint32_t kThreads>
__global__ __launch_bounds__(kThreads, 4) void write_back_page_layer(const __grid_constant__ WriteBackParams p) {
  constexpr int64_t kVecBytes = 16;
  constexpr int64_t kStride = kThreads * kVecBytes;
  static_assert(kPageBytes % kStride == 0, "page must tile across the CTA");
  constexpr int kItems = kPageBytes / kStride;

  const uint32_t page = blockIdx.x / p.num_layers;
  const uint32_t layer = blockIdx.x - page * p.num_layers;
  const int64_t device_page = page_of<kIndexStride>(p.device_indices, page);
  const int64_t host_page = page_of<kIndexStride>(p.host_indices, page);
  const int64_t offset = static_cast<int64_t>(threadIdx.x) * kVecBytes;
  const uint8_t* src = layer_base(p.layer_ptrs, layer) + device_page * kPageBytes + offset;
  uint8_t* dst = p.host + (host_page * p.num_layers + layer) * kPageBytes + offset;
#pragma unroll
  for (int i = 0; i < kItems; ++i) {
    store_streaming_16(dst + i * kStride, load_streaming_16(src + i * kStride));
  }
}

/// \brief Write-back with one CTA per page that walks every layer, using the widest
/// vector access the architecture has.
template <int64_t kPageBytes, int64_t kIndexStride, uint32_t kThreads>
__global__ __launch_bounds__(kThreads, kThreads <= 576 ? 3 : 1) void write_back_page_fused(
    const __grid_constant__ WriteBackParams p) {
  constexpr int64_t kVecBytes = kMaxVecBytes;
  constexpr int64_t kStride = kThreads * kVecBytes;
  static_assert(kPageBytes % kStride == 0, "page must tile across the CTA");
  constexpr int kItems = kPageBytes / kStride;

  const uint32_t page = blockIdx.x;
  const int64_t device_page = page_of<kIndexStride>(p.device_indices, page);
  const int64_t host_page = page_of<kIndexStride>(p.host_indices, page);
  const int64_t offset = static_cast<int64_t>(threadIdx.x) * kVecBytes;
  const int64_t src_offset = device_page * kPageBytes + offset;
  uint8_t* host_page_base = p.host + host_page * p.num_layers * kPageBytes + offset;
#pragma unroll 1
  for (uint32_t layer = 0; layer < p.num_layers; ++layer) {
    const uint8_t* src = layer_base(p.layer_ptrs, layer) + src_offset;
    uint8_t* dst = host_page_base + static_cast<int64_t>(layer) * kPageBytes;
#pragma unroll
    for (int i = 0; i < kItems; ++i) {
#if SGL_ARCH_BLACKWELL_OR_GREATER
      store_streaming_32(dst + i * kStride, load_streaming_32(src + i * kStride));
#else
      store_streaming_16(dst + i * kStride, load_streaming_16(src + i * kStride));
#endif
    }
  }
}

/// \brief Write-back in 128 B lines: 8 lanes move one line, and each thread loads
/// `kUnroll` lines grid-stride apart before issuing any host store, which keeps enough
/// remote stores in flight for small pages.
template <int64_t kPageBytes, int64_t kIndexStride, uint32_t kUnroll>
__global__ __launch_bounds__(256) void write_back_lines(
    const __grid_constant__ WriteBackParams p, uint32_t total_lines, uint32_t line_stride) {
  constexpr uint32_t kLineBytes = 128;
  constexpr uint32_t kLanesPerLine = kLineBytes / 16;
  static_assert(kPageBytes % kLineBytes == 0, "page must be whole 128 B lines");
  constexpr uint32_t kLinesPerPage = kPageBytes / kLineBytes;

  const uint32_t sub = threadIdx.x % kLanesPerLine;
  const uint32_t first_line = blockIdx.x * (blockDim.x / kLanesPerLine) + threadIdx.x / kLanesPerLine;

  uint4 values[kUnroll];
  uint8_t* destinations[kUnroll];
#pragma unroll
  for (uint32_t i = 0; i < kUnroll; ++i) {
    const uint32_t line_id = first_line + i * line_stride;
    destinations[i] = nullptr;
    if (line_id < total_lines) {
      const uint32_t row = line_id / kLinesPerPage;
      const uint32_t line = line_id - row * kLinesPerPage;
      const uint32_t page = row / p.num_layers;
      const uint32_t layer = row - page * p.num_layers;
      const int64_t byte_offset = static_cast<int64_t>(line) * kLineBytes + sub * 16;
      const int64_t device_page = page_of<kIndexStride>(p.device_indices, page);
      const int64_t host_page = page_of<kIndexStride>(p.host_indices, page);
      values[i] = load_no_allocate_16(layer_base(p.layer_ptrs, layer) + device_page * kPageBytes + byte_offset);
      destinations[i] = p.host + (host_page * p.num_layers + layer) * kPageBytes + byte_offset;
    }
  }
#pragma unroll
  for (uint32_t i = 0; i < kUnroll; ++i) {
    if (destinations[i] != nullptr) {
      store_16(destinations[i], values[i]);
    }
  }
}

/// \brief Load-back of one layer: each page is split into `kChunks` CTAs.
template <int64_t kPageBytes, int64_t kIndexStride, uint32_t kChunks, uint32_t kThreads>
__global__ __launch_bounds__(kThreads) void load_back_chunks(const __grid_constant__ LoadParams p) {
  constexpr int64_t kVecBytes = 16;
  static_assert(kPageBytes % (kVecBytes * kChunks) == 0, "chunks must split the page into 16 B vectors");
  constexpr uint32_t kChunkVecs = kPageBytes / kVecBytes / kChunks;

  const uint32_t page = blockIdx.x / kChunks;
  const uint32_t chunk = blockIdx.x - page * kChunks;
  const int64_t host_page = page_of<kIndexStride>(p.host_indices, page);
  const int64_t device_page = page_of<kIndexStride>(p.device_indices, page);
  const int64_t chunk_offset = static_cast<int64_t>(chunk) * kChunkVecs * kVecBytes;
  const uint8_t* src = p.host + (host_page * p.num_layers + p.layer) * kPageBytes + chunk_offset;
  uint8_t* dst = p.device + device_page * kPageBytes + chunk_offset;
#pragma unroll
  for (uint32_t v = threadIdx.x; v < kChunkVecs; v += kThreads) {
    store_streaming_16(dst + v * kVecBytes, load_host_16(src + v * kVecBytes));
  }
}

}  // namespace device::hicache_direct

/// \brief Write-back variants; the Python wrapper picks one per page size.
enum HiCacheDirectWriteBack : int {
  kHiCacheDirectPageLayer = 0,
  kHiCacheDirectPageFused = 1,
  kHiCacheDirectLines = 2,
};

/**
 * \brief Page-granular HiCache transfers between per-layer device pools and a
 *        page_first_direct host pool.
 *
 * \tparam kPageBytes        Bytes of one page of one layer, identical on host and device.
 * \tparam kIndexStride      Token slots per page; page `p` starts at `indices[p * kIndexStride]`.
 * \tparam kWriteBackVariant One of `HiCacheDirectWriteBack`.
 * \tparam kWriteBackParam   CTA threads for the page variants, lines in flight per thread
 *                           for `kHiCacheDirectLines`.
 * \tparam kLoadChunks       CTAs per page on load-back.
 */
template <
    int64_t kPageBytes,
    int64_t kIndexStride,
    int kWriteBackVariant,
    uint32_t kWriteBackParam,
    uint32_t kLoadChunks>
struct HiCacheDirectKernel {
  static_assert(kPageBytes > 0 && kPageBytes % 16 == 0, "pages move as 16 B vectors");
  static_assert(kIndexStride > 0, "index stride must be positive");

  static constexpr uint32_t kLoadThreads = 256;
  static constexpr uint32_t kLinesThreads = 256;

  /// \brief Validate the host pool; returns its layer count.
  static uint32_t check_host(const tvm::ffi::TensorView host_pool) {
    using namespace host;
    CHECK_HOST(host_pool.is_contiguous()) << "HiCacheDirect: host pool must be contiguous";
    CHECK_HOST(host_pool.ndim() >= 3) << "HiCacheDirect: host pool must be [page, layer, ...]";
    const int64_t num_layers = host_pool.size(1);
    CHECK_HOST(num_layers > 0 && num_layers <= std::numeric_limits<uint16_t>::max())
        << "HiCacheDirect: unsupported layer count " << num_layers;
    const int64_t page_bytes = host_pool.numel() / (host_pool.size(0) * num_layers) * dtype_bytes(host_pool.dtype());
    CHECK_HOST(page_bytes == kPageBytes) << "HiCacheDirect: host page is " << page_bytes << " B, kernel expects "
                                         << kPageBytes;
    return static_cast<uint32_t>(num_layers);
  }

  /// \brief Validate a host/device index pair; returns the number of pages.
  static uint32_t check_indices(
      const tvm::ffi::TensorView host_indices, const tvm::ffi::TensorView device_indices, const DLDevice device) {
    using namespace host;
    auto N = SymbolicSize{"num_indices"};
    auto index_device = SymbolicDevice{};
    TensorMatcher({N})  //
        .with_dtype<int64_t>()
        .with_device<kDLGPU>(index_device)
        .verify(host_indices)
        .verify(device_indices);
    CHECK_HOST(index_device.unwrap().device_id == device.device_id)
        << "HiCacheDirect: indices must live on the device pool's GPU";
    const int64_t n = N.unwrap();
    CHECK_HOST(n % kIndexStride == 0) << "HiCacheDirect: " << n << " indices are not whole pages of " << kIndexStride;
    const int64_t num_pages = n / kIndexStride;
    CHECK_HOST(num_pages <= std::numeric_limits<int32_t>::max()) << "HiCacheDirect: too many pages " << num_pages;
    return static_cast<uint32_t>(num_pages);
  }

  /**
   * \brief Copy every layer of the given pages from device to host.
   *
   * \param host_pool      Host pool, contiguous [host_page, num_layers, ...], kPageBytes per (page, layer).
   * \param layer_ptrs     uint64 [num_layers] on the GPU: device base of each host layer.
   * \param host_indices   int64 token slots on the GPU, whole pages.
   * \param device_indices int64 token slots on the GPU, paired with `host_indices`.
   */
  static void write_back(
      const tvm::ffi::TensorView host_pool,
      const tvm::ffi::TensorView layer_ptrs,
      const tvm::ffi::TensorView host_indices,
      const tvm::ffi::TensorView device_indices) {
    using namespace host;
    const uint32_t num_layers = check_host(host_pool);
    auto device_ = SymbolicDevice{};
    TensorMatcher({static_cast<int64_t>(num_layers)})  //
        .with_dtype<uint64_t>()
        .with_device<kDLGPU>(device_)
        .verify(layer_ptrs);
    const DLDevice device = device_.unwrap();
    const uint32_t num_pages = check_indices(host_indices, device_indices, device);
    if (num_pages == 0) {
      return;
    }
    const auto params = device::hicache_direct::WriteBackParams{
        .layer_ptrs = static_cast<const uint64_t*>(layer_ptrs.data_ptr()),
        .host = static_cast<uint8_t*>(runtime::get_device_accessible_ptr(host_pool)),
        .device_indices = static_cast<const int64_t*>(device_indices.data_ptr()),
        .host_indices = static_cast<const int64_t*>(host_indices.data_ptr()),
        .num_layers = num_layers,
        .num_pages = num_pages,
    };
    if constexpr (kWriteBackVariant == kHiCacheDirectPageLayer) {
      const uint64_t num_blocks = static_cast<uint64_t>(num_pages) * num_layers;
      CHECK_HOST(num_blocks <= std::numeric_limits<int32_t>::max()) << "HiCacheDirect: grid too large";
      LaunchKernel(static_cast<uint32_t>(num_blocks), kWriteBackParam, device)(
          device::hicache_direct::write_back_page_layer<kPageBytes, kIndexStride, kWriteBackParam>, params);
    } else if constexpr (kWriteBackVariant == kHiCacheDirectPageFused) {
      LaunchKernel(num_pages, kWriteBackParam, device)(
          device::hicache_direct::write_back_page_fused<kPageBytes, kIndexStride, kWriteBackParam>, params);
    } else {
      static_assert(kWriteBackVariant == kHiCacheDirectLines, "unknown write-back variant");
      constexpr uint64_t kLinesPerCta = kLinesThreads / 8;
      const uint64_t total_lines = static_cast<uint64_t>(num_pages) * num_layers * (kPageBytes / 128);
      const uint64_t num_blocks = div_ceil(total_lines, kLinesPerCta * kWriteBackParam);
      CHECK_HOST(num_blocks * kLinesPerCta * kWriteBackParam <= std::numeric_limits<uint32_t>::max())
          << "HiCacheDirect: transfer too large";
      LaunchKernel(static_cast<uint32_t>(num_blocks), kLinesThreads, device)(
          device::hicache_direct::write_back_lines<kPageBytes, kIndexStride, kWriteBackParam>,
          params,
          static_cast<uint32_t>(total_lines),
          static_cast<uint32_t>(num_blocks * kLinesPerCta));
    }
  }

  /**
   * \brief Copy one host layer of the given pages from host to device.
   *
   * \param device_layer   One device layer, contiguous, whole pages of kPageBytes.
   * \param host_pool      Host pool, contiguous [host_page, num_layers, ...], kPageBytes per (page, layer).
   * \param host_indices   int64 token slots on the GPU, whole pages.
   * \param device_indices int64 token slots on the GPU, paired with `host_indices`.
   * \param layer          Host layer to load.
   */
  static void load(
      const tvm::ffi::TensorView device_layer,
      const tvm::ffi::TensorView host_pool,
      const tvm::ffi::TensorView host_indices,
      const tvm::ffi::TensorView device_indices,
      const int64_t layer) {
    using namespace host;
    const uint32_t num_layers = check_host(host_pool);
    CHECK_HOST(0 <= layer && layer < num_layers) << "HiCacheDirect: layer " << layer << " out of range";
    const DLDevice device = device_layer.device();
    CHECK_HOST(device.device_type == kDLGPU) << "HiCacheDirect: device layer must be on the GPU";
    CHECK_HOST(device_layer.is_contiguous()) << "HiCacheDirect: device layer must be contiguous";
    const uint32_t num_pages = check_indices(host_indices, device_indices, device);
    if (num_pages == 0) {
      return;
    }
    const uint64_t num_blocks = static_cast<uint64_t>(num_pages) * kLoadChunks;
    CHECK_HOST(num_blocks <= std::numeric_limits<int32_t>::max()) << "HiCacheDirect: grid too large";
    const auto params = device::hicache_direct::LoadParams{
        .host = static_cast<const uint8_t*>(runtime::get_device_accessible_ptr(host_pool)),
        .device = static_cast<uint8_t*>(device_layer.data_ptr()),
        .host_indices = static_cast<const int64_t*>(host_indices.data_ptr()),
        .device_indices = static_cast<const int64_t*>(device_indices.data_ptr()),
        .num_layers = num_layers,
        .layer = static_cast<uint32_t>(layer),
    };
    LaunchKernel(static_cast<uint32_t>(num_blocks), kLoadThreads, device)(
        device::hicache_direct::load_back_chunks<kPageBytes, kIndexStride, kLoadChunks, kLoadThreads>, params);
  }
};

}  // namespace sglang
