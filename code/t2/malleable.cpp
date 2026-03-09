#include "malleable.hpp"

#include <algorithm>
#include <chrono>
#include <mpi.h>
#include <thread>
#include <vector>

MalState mal_state;

static const int TEST_SEQ[] = { 2, 4, 4, 3, 1, 4 };
static constexpr int NUM_SEQ = static_cast<int>(sizeof TEST_SEQ / sizeof TEST_SEQ[0]);
static int seq_index = 0;

static inline void set_current(MalFor& f, long val) {
	f.current = val;
	*f.user_iter = val;
}

static inline void set_limit(MalFor& f, long val) {
	f.end = val;
	*f.user_limit = val;
}

static void distribute(long total, int nprocs, int rank, long& my_start, long& my_end) {

	long base = total / nprocs;
	long rem = total % nprocs;

	my_start = (long)rank * base + std::min((long)rank, rem);
	my_end = my_start + base + (rank < rem ? 1 : 0);

}

template<typename Pred> static void compute_wait(Pred pred) {

	mal_state.secure_synchronization = true;
	mal_state.compute_thread_cv.notify_all();

	{
		std::unique_lock<std::mutex> lock(mal_state.compute_thread_mutex);
		mal_state.compute_thread_cv.wait(lock, pred);
	}

	mal_state.secure_synchronization = false;

}

static void execute_resize(int target_size) {

	if (target_size == mal_state.current_size) return;

	mal_state.waiting_for_resize = true;

	if (!mal_state.secure_synchronization) {

		MAL_LOG("RESIZE", "Aux. thread waiting for secure synchronization");

		std::unique_lock<std::mutex> lock(mal_state.compute_thread_mutex);

		mal_state.compute_thread_cv.wait(lock, [] {

			return mal_state.secure_synchronization || mal_state.should_stop;

		});

	}

	MAL_LOG("RESIZE", "Starting resize operation to target size %d", target_size);
	double t0 = MPI_Wtime();

	const bool is_reduce = (mal_state.active_reduce != nullptr);

	int universe_size;
	MPI_Comm_size(mal_state.universe_comm, &universe_size);

	std::vector<long> local_flat;
	long local_acc = 0;

	if (mal_state.active_loop != nullptr && mal_state.active_comm != MPI_COMM_NULL) {

		long rs = *mal_state.active_loop->user_iter + 1;
		long re = mal_state.active_loop->end;

		if (rs < re) {

			local_flat.push_back(rs);
			local_flat.push_back(re);

		}

		for (const auto& r : mal_state.active_loop->extra_ranges) {

			if (r.first < r.second) {

				local_flat.push_back(r.first);
				local_flat.push_back(r.second);

			}

		}

		if (is_reduce) {

			local_acc = *mal_state.active_reduce->user_acc;

		}

	}

	long local_meta[2] = { (long)(local_flat.size() / 2), local_acc };
	std::vector<long> all_meta(universe_size * 2);
	MPI_Allgather(local_meta, 2, MPI_LONG, all_meta.data(), 2, MPI_LONG, mal_state.universe_comm);

	std::vector<int> recv_counts(universe_size), recv_displs(universe_size);

	for (int i = 0; i < universe_size; ++i) {

		recv_counts[i] = (int)(all_meta[i * 2] * 2);
		recv_displs[i] = (i == 0) ? 0 : recv_displs[i-1] + recv_counts[i-1];

	}

	int total_flat = recv_displs[universe_size-1] + recv_counts[universe_size-1];
	std::vector<long> all_flat(total_flat);

	MPI_Allgatherv(local_flat.empty() ? nullptr : local_flat.data(), (int)local_flat.size(), MPI_LONG, all_flat.empty() ? nullptr : all_flat.data(), recv_counts.data(), recv_displs.data(), MPI_LONG, mal_state.universe_comm);

	long total_acc = 0;

	if (is_reduce) {

		for (int i = 0; i < universe_size; ++i) {

			total_acc += all_meta[i*2+1];

		}

	}

	std::vector<std::pair<long,long>> remaining;
	remaining.reserve(total_flat / 2);
	long total_remaining = 0;

	for (int i = 0; i < total_flat; i += 2) {

		remaining.push_back({all_flat[i], all_flat[i+1]});
		total_remaining += all_flat[i+1] - all_flat[i];

	}

	auto slice = [&](long vstart, long vend) -> std::vector<std::pair<long,long>> {

		std::vector<std::pair<long,long>> out;

		long offset = 0;

		for (const auto& r : remaining) {

			long len = r.second - r.first;

			if (offset + len <= vstart) {

				offset += len;
				continue;

			}

			if (offset >= vend) {

				break;

			}

			long s = r.first + std::max(0L, vstart - offset);
			long e = r.first + std::min(len, vend - offset);

			if (s < e) {

				out.push_back({s, e});

			}

			offset += len;

		}

		return out;

	};

	if (mal_state.active_comm != MPI_COMM_NULL) {

		MPI_Comm_free(&mal_state.active_comm);
		mal_state.active_comm = MPI_COMM_NULL;

	}

	int color = (mal_state.universe_rank < target_size) ? 0 : MPI_UNDEFINED;
	MPI_Comm_split(mal_state.universe_comm, color, mal_state.universe_rank, &mal_state.active_comm);

	if (mal_state.active_comm != MPI_COMM_NULL) {

		MPI_Comm_rank(mal_state.active_comm, &mal_state.current_rank);
		MPI_Comm_size(mal_state.active_comm, &mal_state.current_size);

		long vstart, vend;
		distribute(total_remaining, mal_state.current_size, mal_state.current_rank, vstart, vend);
		auto assigned = slice(vstart, vend);

		MAL_LOG("RESIZE", "Rank %d assigned %zu real range(s) covering %ld iters", mal_state.current_rank, assigned.size(), vend - vstart);

		if (mal_state.active_loop != nullptr) {

			if (!assigned.empty()) {

				mal_state.active_loop->start = assigned[0].first;
				set_limit(*mal_state.active_loop, assigned[0].second);
				set_current(*mal_state.active_loop, assigned[0].first);
				mal_state.active_loop->extra_ranges.assign(assigned.begin() + 1, assigned.end());

			} else {

				long cur = *mal_state.active_loop->user_iter;
				mal_state.active_loop->start = cur;
				set_limit(*mal_state.active_loop, cur);
				set_current(*mal_state.active_loop, cur);
				mal_state.active_loop->extra_ranges.clear();

			}

			if (is_reduce) {

				mal_state.active_reduce->global_acc += total_acc;
				*mal_state.active_reduce->user_acc = 0;

			}

		} else {

			mal_state.pending_ranges = assigned;

			if (is_reduce) {

				mal_state.pending_global_acc = total_acc;

			}

			mal_state.secure_synchronization = true;

		}

	} else {

		mal_state.current_rank = -1;
		mal_state.current_size = 0;

		if (mal_state.active_loop != nullptr) {

			long cur = *mal_state.active_loop->user_iter;
			mal_state.active_loop->start = cur;
			set_limit(*mal_state.active_loop, cur);
			set_current(*mal_state.active_loop, cur);
			mal_state.active_loop->extra_ranges.clear();

			if (is_reduce) {

				*mal_state.active_reduce->user_acc = 0;

			}

		}

		mal_state.secure_synchronization = true;

	}

	MAL_LOG("RESIZE", "Resize to %d completed in %.8f s (new_size=%d)", target_size, MPI_Wtime() - t0, mal_state.current_size);

}

static void mal_progress_thread() {

	while (!mal_state.should_stop) {

		{

			std::unique_lock<std::mutex> lock(mal_state.compute_thread_mutex);
			mal_state.compute_thread_cv.wait_for(
				lock,
				std::chrono::milliseconds(MAL_EPOCH_INTERVAL_MS),
				[] { return mal_state.should_stop.load(); }
			);

		}

		if (mal_state.should_stop) {

			break;

		}

		if (seq_index < NUM_SEQ) {

			MAL_LOG("EPOCH", "[%d] target=%d", seq_index, TEST_SEQ[seq_index]);

			execute_resize(TEST_SEQ[seq_index]);

			MAL_LOG("EPOCH", "[%d] done new_size=%d", seq_index, mal_state.current_size);

			if (++seq_index == NUM_SEQ) {

				mal_state.should_stop = true;

			}

		}

		mal_state.waiting_for_resize = false;
		mal_state.compute_thread_cv.notify_all();

	}

	mal_state.compute_thread_cv.notify_all();

}

void mal_init() {

	MPI_Session_init(MPI_INFO_NULL, MPI_ERRORS_RETURN, &mal_state.session);
	MPI_Group_from_session_pset(mal_state.session, "mpi://WORLD", &mal_state.world_group);
	MPI_Comm_create_from_group(mal_state.world_group, "malleable.universe", MPI_INFO_NULL, MPI_ERRORS_RETURN, &mal_state.universe_comm);
	MPI_Comm_rank(mal_state.universe_comm, &mal_state.universe_rank);

	int color = (mal_state.universe_rank < MAL_INITIAL_SIZE) ? 0 : MPI_UNDEFINED;
	MPI_Comm_split(mal_state.universe_comm, color, mal_state.universe_rank, &mal_state.active_comm);

	mal_state.mal_thread = std::thread(mal_progress_thread);

	if (mal_state.active_comm != MPI_COMM_NULL) {

		MPI_Comm_rank(mal_state.active_comm, &mal_state.current_rank);
		MPI_Comm_size(mal_state.active_comm, &mal_state.current_size);

	} else {

		mal_state.current_rank = -1;
		mal_state.current_size = 0;

		compute_wait([] {

			return mal_state.active_comm != MPI_COMM_NULL || mal_state.should_stop;

		});

	}

}

void mal_finalize() {

	mal_state.should_stop = true;
	mal_state.compute_thread_cv.notify_all();
	mal_state.mal_thread.join();

	if (mal_state.active_comm != MPI_COMM_NULL) {

		MPI_Comm_free(&mal_state.active_comm);

	}

	MPI_Comm_free(&mal_state.universe_comm);
	MPI_Group_free(&mal_state.world_group);
	MPI_Session_finalize(&mal_state.session);

}

static void init_loop(MalFor& f, long total_iters, long& iter, long& limit) {

	f.user_iter = &iter;
	f.user_limit = &limit;
	f.extra_ranges.clear();

	if (!mal_state.pending_ranges.empty()) {

		f.start = mal_state.pending_ranges[0].first;
		f.end = mal_state.pending_ranges[0].second;
		f.extra_ranges.assign(mal_state.pending_ranges.begin() + 1, mal_state.pending_ranges.end());
		mal_state.pending_ranges.clear();

	} else if (mal_state.current_size > 0) {

		distribute(total_iters, mal_state.current_size, mal_state.current_rank, f.start, f.end);

	} else {

		f.start = f.end = 0;

	}

	set_current(f, f.start);
	limit = f.end;

	mal_state.secure_synchronization = false;
	mal_state.active_loop = &f;

	while (f.start == f.end && !mal_state.should_stop) {

		f.current = f.end;

		compute_wait([] {

			return mal_state.should_stop || (mal_state.active_loop && mal_state.active_loop->current < mal_state.active_loop->end);

		});

	}

}

MalFor mal_for(long total_iters, long& iter, long& limit) {

	mal_state.active_reduce = nullptr;
	MalFor f;
	init_loop(f, total_iters, iter, limit);
	return f;

}

MalReduce mal_for_reduce(long total_iters, long& iter, long& limit, long& acc, int result_rank) {

	MalReduce r;
	init_loop(r, total_iters, iter, limit);
	r.user_acc = &acc;
	r.result_rank = result_rank;
	acc = 0;
	r.global_acc = mal_state.pending_global_acc;
	mal_state.pending_global_acc = 0;
	mal_state.active_reduce = &r;
	return r;

}

static void check_loop_internal(MalFor& f) {

	f.current = *f.user_iter;

	if (mal_state.waiting_for_resize) {

		compute_wait([] {

			return !mal_state.waiting_for_resize || mal_state.should_stop;

		});

		if (mal_state.active_comm != MPI_COMM_NULL && f.start < f.end) {

			set_current(f, f.start - 1);
			return;

		}

	}

	if (*f.user_iter + 1 < f.end) {

		return;

	}

	if (!f.extra_ranges.empty()) {

		auto next = f.extra_ranges.front();

		f.extra_ranges.erase(f.extra_ranges.begin());
		f.start = next.first;

		set_limit(f, next.second);
		set_current(f, next.first - 1);

		MAL_LOG("RANGE", "Advancing to extra range [%ld, %ld)", next.first, next.second);

		return;

	}

	if (mal_state.should_stop) {

		return;

	}

	f.current = f.end;

	compute_wait([] {

		return mal_state.should_stop || (mal_state.active_loop && mal_state.active_loop->current < mal_state.active_loop->end);

	});

	if (!mal_state.should_stop) {

		set_current(f, f.start - 1);

	}

}

void mal_check_for(MalFor& f) {

	check_loop_internal(f);

}

void mal_check_reduce(MalReduce& r) {

	check_loop_internal(r);

	if (mal_state.should_stop && r.extra_ranges.empty() && *r.user_iter + 1 >= r.end) {

		mal_reduce_result(r);

	}

}

void mal_reduce_result(MalReduce& r) {

	if (r.result_done) return;
	r.result_done = true;

	if (!mal_state.should_stop) {

		compute_wait([] {

			return mal_state.should_stop.load();

		});

	}

	if (mal_state.active_comm == MPI_COMM_NULL) {

		return;

	}

	long local = *r.user_acc, global = 0;
	MPI_Reduce(&local, &global, 1, MPI_LONG, MPI_SUM, r.result_rank, mal_state.active_comm);

	if (mal_state.current_rank == r.result_rank) {

		*r.user_acc = r.global_acc + global;

	}

	mal_state.active_reduce = nullptr;
	mal_state.active_loop = nullptr;

}

MalReduce::~MalReduce() {

	if (!result_done) {

		mal_reduce_result(*this);

	}

}