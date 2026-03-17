#pragma once

#include <mpi.h>
#include <cstddef>
#include <cstdio>
#include <utility>
#include <vector>

#ifndef MAL_INITIAL_SIZE

	#define MAL_INITIAL_SIZE 1

#endif

#ifndef MAL_EPOCH_INTERVAL_MS

	#define MAL_EPOCH_INTERVAL_MS 3000

#endif

#ifndef MAL_LOG_LEVEL

	#define MAL_LOG_LEVEL MAL_LOG_INFO

#endif

int mal_rank();

enum MalLogLevel {
	MAL_LOG_DEBUG,
	MAL_LOG_INFO,
	MAL_LOG_WARN,
	MAL_LOG_ERROR,
};

const char* mal_log_level_name(MalLogLevel level);
const char* mal_log_level_color(MalLogLevel level);
const char* mal_log_reset_color();

#define MAL_LOG(level, fmt, ...) do { if ((int)(level) >= (int)(MAL_LOG_LEVEL)) printf("%s[%8.3f][%-6s][R%d] " fmt "%s\n", mal_log_level_color((level)), MPI_Wtime(), mal_log_level_name((level)), mal_rank(), ##__VA_ARGS__, mal_log_reset_color()); } while (0)
#define MAL_LOG_L(level, tag, fmt, ...) do { if ((int)(level) >= (int)(MAL_LOG_LEVEL)) printf("%s[%8.3f][%-5s][%-6s][R%d] " fmt "%s\n", mal_log_level_color((level)), MPI_Wtime(), mal_log_level_name((level)), (tag), mal_rank(), ##__VA_ARGS__, mal_log_reset_color()); } while (0)

struct MalVec;
struct MalAcc;

enum MalDimMode {
	MAL_DIM_PARTITIONED,
	MAL_DIM_SHARED,
};

enum MalHaloMode {
	MAL_HALO_CLAMP,
	MAL_HALO_ZERO,
	MAL_HALO_PERIODIC,
};

enum MalAttachPolicy {
	MAL_ATTACH_DEFAULT,
	MAL_ATTACH_ONCE_ALL,
};

template<typename T> struct MpiType;
template<> struct MpiType<int> { static MPI_Datatype value() { return MPI_INT; } };
template<> struct MpiType<long> { static MPI_Datatype value() { return MPI_LONG; } };
template<> struct MpiType<long long> { static MPI_Datatype value() { return MPI_LONG_LONG; } };
template<> struct MpiType<unsigned> { static MPI_Datatype value() { return MPI_UNSIGNED; } };
template<> struct MpiType<unsigned long> { static MPI_Datatype value() { return MPI_UNSIGNED_LONG; } };
template<> struct MpiType<float> { static MPI_Datatype value() { return MPI_FLOAT; } };
template<> struct MpiType<double> { static MPI_Datatype value() { return MPI_DOUBLE; } };

struct MalFor {

	long start{0};
	long end{0};
	long current{0};
	long* user_iter{nullptr};
	long* user_limit{nullptr};

	std::vector<std::pair<long,long>> extra_ranges;
	std::vector<long> extra_range_local_bases;
	size_t extra_idx{0};
	long range_local_base{0};

	std::vector<std::pair<long,long>> plan_ranges;
	std::vector<long> plan_local_bases;

	std::vector<MalVec*> vecs;
	std::vector<MalAcc*> accs;

	MalFor() = default;
	MalFor(const MalFor&) = delete;
	MalFor& operator=(const MalFor&) = delete;
	MalFor(MalFor&&) noexcept = default;
	MalFor& operator=(MalFor&&) = delete;

};

void mal_init();
void mal_finalize();

void mal_set_epoch_interval_ms(int ms);
void mal_set_resize_enabled(bool enabled);
void mal_set_resize_sequence(const int* seq, size_t count);
void mal_reset_resize_sequence_default();

[[nodiscard]] MalFor mal_for(long total_iters, long& iter, long& limit);
void mal_check_for(MalFor& f);

void mal_attach_vec(MalFor& f, void** user_ptr, size_t elem_size, long total_N, int result_rank = -1, MalAttachPolicy policy = MAL_ATTACH_DEFAULT);

namespace detail {

	struct AccDesc {

		void* ptr;
		MPI_Datatype dtype;
		MPI_Op dop;
		size_t esz;
		void (*fn_get) (const void* p, void* dst);
		void (*fn_set) (void* p, const void* src);
		void (*fn_add) (void* p, const void* src);
		void (*fn_reset)(void* p);

	};

	void acc_register(MalFor& f, AccDesc d, int result_rank);

}

template<typename T>
inline void mal_attach_acc(MalFor& f, T& acc, MPI_Datatype dtype, MPI_Op op, int result_rank = 0) {

	detail::acc_register(f, {
		&acc, dtype, op, sizeof(T),
		[](const void* p, void* d) { *static_cast<T*>(d) = *static_cast<const T*>(p); },
		[](void* p, const void* s) { *static_cast<T*>(p) = *static_cast<const T*>(s); },
		[](void* p, const void* s) { *static_cast<T*>(p) += *static_cast<const T*>(s); },
		[](void* p) { *static_cast<T*>(p) = T{}; },
	}, result_rank);

}

template<typename T>
inline void mal_attach_acc(MalFor& f, T& acc, int result_rank = 0) {

	mal_attach_acc(f, acc, MpiType<T>::value(), MPI_SUM, result_rank);

}

void mal_attach_mat(MalFor& f, void** user_ptr, size_t elem_size, long primary_n, long secondary_n, MalDimMode mode, int result_rank = -1, MalAttachPolicy policy = MAL_ATTACH_DEFAULT);

void mal_attach_halo(MalFor& f, void** user_ptr, int halo, MalHaloMode mode = MAL_HALO_CLAMP, MalAttachPolicy policy = MAL_ATTACH_DEFAULT);
void mal_exchange_halo(MalFor& f);
