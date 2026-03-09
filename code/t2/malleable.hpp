#pragma once

#include <mpi.h>
#include <atomic>
#include <cstdio>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#ifndef MAL_INITIAL_SIZE
	#define MAL_INITIAL_SIZE 1
#endif

#ifndef MAL_EPOCH_INTERVAL_MS
	#define MAL_EPOCH_INTERVAL_MS 3000
#endif

struct MalFor {

	long start{0};
	long end{0};
	long current{0};
	long* user_iter{nullptr};
	long* user_limit{nullptr};
	std::vector<std::pair<long,long>> extra_ranges;

	MalFor() = default;
	MalFor(const MalFor&) = delete;
	MalFor& operator=(const MalFor&) = delete;
	MalFor(MalFor&& o) noexcept : start(o.start), end(o.end), current(o.current), user_iter(std::exchange(o.user_iter, nullptr)), user_limit(std::exchange(o.user_limit, nullptr)), extra_ranges(std::move(o.extra_ranges)) {}
	MalFor& operator=(MalFor&&) = delete;

};

struct MalReduce : MalFor {

	long* user_acc{nullptr};
	long global_acc{0};
	int result_rank{0};
	bool result_done{false};

	MalReduce() = default;
	~MalReduce();
	MalReduce(const MalReduce&) = delete;
	MalReduce& operator=(const MalReduce&) = delete;
	MalReduce(MalReduce&& o) noexcept : MalFor(std::move(o)), user_acc(std::exchange(o.user_acc, nullptr)), global_acc(o.global_acc), result_rank(o.result_rank), result_done(std::exchange(o.result_done, true)) {}
	MalReduce& operator=(MalReduce&&) = delete;

};

struct MalState {

	MPI_Session session{MPI_SESSION_NULL};
	MPI_Group world_group{MPI_GROUP_NULL};
	MPI_Comm universe_comm{MPI_COMM_NULL};
	int universe_rank{-1};

	MPI_Comm active_comm{MPI_COMM_NULL};
	int current_rank{-1};
	int current_size{0};

	std::thread mal_thread;

	std::atomic<bool> should_stop{false};
	std::atomic<bool> secure_synchronization{false};
	std::atomic<bool> waiting_for_resize{false};

	std::mutex compute_thread_mutex;
	std::condition_variable compute_thread_cv;

	MalFor* active_loop{nullptr};
	MalReduce* active_reduce{nullptr};

	std::vector<std::pair<long,long>> pending_ranges;
	long pending_global_acc{0};

	MalState() noexcept {}
	MalState(const MalState&) = delete;
	MalState& operator=(const MalState&) = delete;

};

extern MalState mal_state;

#define MAL_LOG(tag, fmt, ...) printf("[%8.3f][%-6s][R%d] " fmt "\n", MPI_Wtime(), (tag), mal_state.universe_rank, ##__VA_ARGS__)

void mal_init();
void mal_finalize();

MalFor mal_for(long total_iters, long& iter, long& limit);
MalReduce mal_for_reduce(long total_iters, long& iter, long& limit, long& acc, int result_rank = 0);

void mal_check_for(MalFor& f);
void mal_check_reduce(MalReduce& r);

void mal_reduce_result(MalReduce& r);