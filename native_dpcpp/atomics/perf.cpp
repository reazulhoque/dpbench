//dpcpp perf.cpp -o perf_slow
//dpcpp -DSYCL_USE_NATIVE_FP_ATOMICS perf.cpp -o perf_fast
#include <CL/sycl.hpp>

// Benchmark configuration
static constexpr uint32_t SIMD = 16;
static constexpr uint64_t ALIGNMENT = 64;
using data_t = float;

int main(int argc, char* argv[])
{
	size_t N = 1024 * 1024 * 500;
	printf("N = %lu\n", N);
	printf("SIMD = %u\n", SIMD);
	printf("ALIGNMENT = %lu\n", ALIGNMENT);
	printf("\n");

	// Use one tile where possible
	sycl::device root = sycl::device(sycl::gpu_selector());
	sycl::device device;
	try {
		std::vector<sycl::device> tiles = root.create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>
											  (sycl::info::partition_affinity_domain::next_partitionable);
		device = tiles[0];
	} catch (sycl::exception& e) {
		device = root;
	}
	sycl::queue q{device, sycl::property::queue::enable_profiling{}};

	// Align data to ensure one cache line
	data_t* data = (data_t*) sycl::aligned_alloc_device(ALIGNMENT, N * sizeof(data_t), q);
	q.memset(data, 0, N * sizeof(data_t));
	q.wait();

	printf("Atomic Performance\n");
	printf("------------------------------\n");
	sycl::event ev = q.parallel_for(N, [=](sycl::id<1> i) [[intel::reqd_sub_group_size(SIMD)]] {
	  sycl::ONEAPI::atomic_ref<data_t, sycl::ONEAPI::memory_order::relaxed, sycl::ONEAPI::memory_scope::device, sycl::access::address_space::global_space>(data[i]) += 1;
	});
	ev.wait();
	uint64_t ns = ev.get_profiling_info<sycl::info::event_profiling::command_end>() - ev.get_profiling_info<sycl::info::event_profiling::command_start>();
	printf("Time = %2e seconds\n", ns * 1e-09);
	printf("\n");

	return 0;
}

