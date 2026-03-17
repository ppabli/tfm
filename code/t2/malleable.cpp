#include "malleable.hpp"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <numeric>
#include <thread>
#include <unordered_map>
#include <vector>

static constexpr int kDefaultSeq[] = {2, 4, 4, 3, 1, 4};

class BufferPool {

	struct Entry {
		void* ptr;
		size_t capacity;
	};

	std::unordered_map<size_t, std::vector<Entry>> buckets_;
	std::mutex mtx_;

	static size_t bucket_key(size_t bytes) {

		if (bytes == 0) {

			return 0;

		}

		size_t key = 1UL << (64 - __builtin_clzl(bytes - 1));
		return key;

	}

public:

	~BufferPool() {

		std::lock_guard lk(mtx_);

		for (auto& [key, entries] : buckets_) {

			for (auto& e : entries) {

				std::free(e.ptr);

			}

		}

	}

	void* acquire(size_t min_bytes) {

		if (min_bytes == 0) {

			min_bytes = 1;

		}

		size_t key = bucket_key(min_bytes);

		{

			std::lock_guard lk(mtx_);

			auto it = buckets_.find(key);

			if (it != buckets_.end() && !it->second.empty()) {

				Entry e = it->second.back();
				it->second.pop_back();

				return e.ptr;

			}

		}

		void* p = std::malloc(key);

		if (!p) {

			throw std::bad_alloc();

		}

		return p;

	}

	void release(void* ptr, size_t capacity) {

		if (!ptr) {

			return;

		}

		size_t key = bucket_key(capacity > 0 ? capacity : 1);

		std::lock_guard lk(mtx_);
		buckets_[key].push_back({ptr, capacity});

	}

};

static BufferPool g_buffer_pool;

struct MalData {

	virtual ~MalData() = default;
	virtual void free_resources() = 0;

};

struct MalVec : MalData {

	void* buf{nullptr};
	size_t elem_size{0};
	long local_n{0};
	long done_n{0};
	long buf_global_start{0};
	long total_N{0};
	void** user_ptr{nullptr};
	int gather_root{-1};
	void* result_buf{nullptr};
	long plan_origin_n{0};

	std::vector<std::pair<long,long>> done_segs;

	int halo_n{0};
	MalHaloMode halo_mode{MAL_HALO_CLAMP};
	void* halo_buf{nullptr};
	size_t halo_buf_bytes{0};
	int halo_lnbr{MPI_PROC_NULL};
	int halo_rnbr{MPI_PROC_NULL};
	MPI_Comm cart_comm{MPI_COMM_NULL};
	bool fully_replicated{false};
	bool halo_static_once{false};
	bool halo_initialized{false};

	void sync_user_ptr();
	void free_resources() override;

};

struct MalAcc {

	int result_rank{0};
	void* ptr{nullptr};
	int dtype_idx{1};
	int dop_idx{0};
	size_t esz{sizeof(long)};

	void (*fn_get) (const void*, void*){nullptr};
	void (*fn_set) (void*, const void*){nullptr};
	void (*fn_add) (void*, const void*){nullptr};
	void (*fn_reset)(void*) {nullptr};

	std::vector<char> epoch_buf;

};

struct SharedMat : MalData {

	void* buf{nullptr};
	size_t total_bytes{0};
	bool user_owned{false};
	void** user_ptr{nullptr};

	void free_resources() override;

};

struct PendingActivation {

	std::vector<std::pair<long,long>> ranges;
	std::vector<long> range_local_bases;
	std::vector<void*> vec_slices;
	std::vector<std::vector<char>> acc_epoch_bufs;
	std::vector<void*> shared_mats;
	size_t acc_idx{0}, shared_idx{0};

	~PendingActivation() {

		for (void* p : vec_slices) {

			std::free(p);

		}

		for (void* p : shared_mats) {

			std::free(p);

		}

	}

};

struct MalState {

	struct AsyncBcastTask {

		void* buf{nullptr};
		size_t bytes{0};
		int root{0};
		bool done{false};

		std::mutex mu;
		std::condition_variable cv;

	};

	struct CommInfo {

		MPI_Session session{MPI_SESSION_NULL};
		MPI_Group world_group{MPI_GROUP_NULL};
		MPI_Comm universe{MPI_COMM_NULL};
		int u_rank{-1};
		int u_size{0};
		MPI_Comm active{MPI_COMM_NULL};
		int a_rank{-1};
		int a_size{0};

	} comm;

	struct Config {

		std::vector<int> sequence;
		size_t seq_idx{0};
		std::atomic<int> epoch_ms{MAL_EPOCH_INTERVAL_MS};
		std::atomic<bool> enabled{true};
		std::mutex mu;

	} cfg;

	struct SyncBarrier {

		std::mutex mu;
		std::condition_variable cv;
		std::atomic<bool> compute_ready{false};
		std::atomic<bool> resize_pending{false};
		std::atomic<bool> attach_pending{false};
		std::atomic<bool> stop{false};

		template<typename Pred>
		void compute_wait(Pred ready) {

			compute_ready = true;

			cv.notify_all();

			{
				std::unique_lock lk(mu);
				cv.wait(lk, ready);
			}

			compute_ready = false;

		}

		void wait_for_compute() {

			if (compute_ready) {

				return;

			}

			std::unique_lock lk(mu);

			cv.wait(lk, [this] {

				return compute_ready.load() || stop.load();

			});

		}

		void notify() {

			cv.notify_all();

		}

	} sync;

	std::mutex attach_mu;
	std::deque<std::shared_ptr<AsyncBcastTask>> attach_tasks;

	MalFor* loop{nullptr};

	std::vector<std::unique_ptr<MalVec>> vecs;
	std::vector<std::unique_ptr<MalAcc>> accs;
	std::vector<std::unique_ptr<SharedMat>> shared;

	std::unique_ptr<PendingActivation> pending;

	std::thread worker;

	MalState() noexcept = default;
	MalState(const MalState&) = delete;
	MalState& operator=(const MalState&) = delete;

};

static MalState g;

int mal_rank() {

	return g.comm.u_rank;

}

const char* mal_log_level_name(MalLogLevel level) {

	switch (level) {

		case MAL_LOG_DEBUG: return "DEBUG";
		case MAL_LOG_INFO: return "INFO";
		case MAL_LOG_WARN: return "WARN";
		case MAL_LOG_ERROR: return "ERROR";

	}

	return "INFO";

}

const char* mal_log_level_color(MalLogLevel level) {

	switch (level) {

		case MAL_LOG_DEBUG: return "\x1b[36m";
		case MAL_LOG_INFO: return "\x1b[32m";
		case MAL_LOG_WARN: return "\x1b[33m";
		case MAL_LOG_ERROR: return "\x1b[31m";

	}

	return "";

}

const char* mal_log_reset_color() {

	return "\x1b[0m";

}

static const MPI_Datatype kDtypeTbl[] = {MPI_INT, MPI_LONG, MPI_LONG_LONG, MPI_UNSIGNED, MPI_UNSIGNED_LONG, MPI_FLOAT, MPI_DOUBLE};

static const MPI_Op kDopTbl[] = {MPI_SUM, MPI_PROD, MPI_MAX, MPI_MIN};

static MPI_Datatype tag_dtype(int t) {

	return (t >= 0 && t < (int)std::size(kDtypeTbl)) ? kDtypeTbl[t] : MPI_LONG;

}
static int dtype_tag(MPI_Datatype d) {

	for (int i = 0; i < (int)std::size(kDtypeTbl); ++i) {

		if (kDtypeTbl[i] == d) return i;

	}

	return 1;

}
static int dop_tag(MPI_Op d) {

	for (int i = 0; i < (int)std::size(kDopTbl); ++i) {

		if (kDopTbl[i] == d) {

			return i;

		}

	}

	return 0;

}
static MPI_Op tag_dop(int t) {

	return (t >= 0 && t < (int)std::size(kDopTbl)) ? kDopTbl[t] : MPI_SUM;

}

static void write_identity(char* dst, int dtype_tag, int dop_tag, int esz) {

	union { int i; long l; long long ll; unsigned u; unsigned long ul; float f; double d; } v{};

	switch (dop_tag) {

		case 0: break;
		case 1:
			switch (dtype_tag) {
				case 5: v.f = 1.0f; break;
				case 6: v.d = 1.0; break;
				default: v.l = 1; break;
			} break;
		case 2:

			switch (dtype_tag) {

				case 0: v.i = INT_MIN; break;
				case 1: v.l = LONG_MIN; break;
				case 2: v.ll = LLONG_MIN; break;
				case 3: v.u = 0; break;
				case 4: v.ul = 0; break;
				case 5: v.f = -FLT_MAX; break;
				case 6: v.d = -DBL_MAX; break;

			} break;

		case 3:

			switch (dtype_tag) {

				case 0: v.i = INT_MAX; break;
				case 1: v.l = LONG_MAX; break;
				case 2: v.ll = LLONG_MAX; break;
				case 3: v.u = UINT_MAX; break;
				case 4: v.ul = ULONG_MAX; break;
				case 5: v.f = FLT_MAX; break;
				case 6: v.d = DBL_MAX; break;

			} break;

	}

	std::memcpy(dst, &v, (size_t)esz);

}

static void* checked_realloc(void* p, size_t n, const char* ctx) {

	void* nb = std::realloc(p, n);

	if (!nb) {

		MAL_LOG_L(MAL_LOG_ERROR, "ALLOC", "realloc failed in %s", ctx);
		MPI_Abort(g.comm.universe, 1);

	}

	return nb;

}

static void* pool_alloc(size_t bytes) {

	return g_buffer_pool.acquire(bytes);

}

static void pool_free(void* ptr, size_t capacity) {

	g_buffer_pool.release(ptr, capacity);

}

static void mpi_bcast_bytes(void* buf, size_t bytes, int root, MPI_Comm comm) {

	char* p = static_cast<char*>(buf);
	size_t off = 0;

	while (off < bytes) {

		int chunk = (int)std::min(bytes - off, (size_t)INT_MAX);
		MPI_Bcast(p + off, chunk, MPI_BYTE, root, comm);
		off += (size_t)chunk;

	}

}

static void run_async_bcast_once_all(void* buf, size_t bytes, int root = 0) {

	if (g.comm.universe == MPI_COMM_NULL || bytes == 0) {

		return;

	}

	auto task = std::make_shared<MalState::AsyncBcastTask>();
	task->buf = buf;
	task->bytes = bytes;
	task->root = root;

	{
		std::lock_guard lk(g.attach_mu);
		g.attach_tasks.push_back(task);
		g.sync.attach_pending = true;
	}

	g.sync.notify();

	std::unique_lock lk(task->mu);
	task->cv.wait(lk, [&] { return task->done || g.sync.stop.load(); });

}


static bool has_work_or_stop() {

	return g.sync.stop.load() || g.sync.attach_pending.load() || (g.loop && g.loop->current < g.loop->end);

}

template<typename GetAcc, typename OnResult>
static void batched_allreduce(int n, GetAcc get_acc, OnResult on_result) {

	if (n == 0) {

		return;

	}

	struct Meta { int dt, dp, esz; };

	std::vector<Meta> meta(n);

	for (int k = 0; k < n; k++) {

		if (MalAcc* a = get_acc(k)) {

			meta[k] = {a->dtype_idx, a->dop_idx, (int)a->esz};

		} else {

			meta[k] = {1, 0, (int)sizeof(long)};

		}

	}

	MPI_Bcast(meta.data(), n * (int)sizeof(Meta), MPI_BYTE, 0, g.comm.universe);

	int ai = 0;

	std::vector<char> send, recv;
	send.reserve(512);
	recv.reserve(512);

	while (ai < n) {

		int ae = ai + 1;
		int esz = meta[ai].esz;

		while (ae < n && meta[ae].dt == meta[ai].dt && meta[ae].dp == meta[ai].dp && meta[ae].esz == esz) {

			ae++;

		}

		int gsz = ae - ai;

		send.resize(gsz * esz);
		recv.resize(gsz * esz);

		for (int k = ai; k < ae; k++) {

			char* slot = send.data() + (k - ai) * esz;

			MalAcc* a = get_acc(k);

			if (a) {

				a->fn_get(a->ptr, slot);

				if (g.comm.u_rank == 0) {

					a->fn_add(slot, a->epoch_buf.data());

				}

			} else {

				write_identity(slot, meta[ai].dt, meta[ai].dp, esz);

			}

		}

		MPI_Allreduce(send.data(), recv.data(), gsz, tag_dtype(meta[ai].dt), tag_dop(meta[ai].dp), g.comm.universe);

		for (int k = ai; k < ae; k++) {

			on_result(k, recv.data() + (k - ai) * esz, esz);

		}

		ai = ae;

	}

}

static void set_iter(MalFor& f, long v) {

	f.current = v;
	*f.user_iter = v;

}

static void set_limit(MalFor& f, long v) {

	f.end = v;
	*f.user_limit = v;

}

static void prime_range_start(MalFor& f) {

	set_iter(f, f.start - 1);

}

static void sync_vec_mapping_for_current_range(MalFor& f) {

	for (MalVec* v : f.vecs) {

		if (!v || v->fully_replicated) {

			continue;

		}

		long new_global_start = f.start - (v->done_n + f.range_local_base);

		if (v->buf_global_start != new_global_start) {
			v->buf_global_start = new_global_start;
			v->sync_user_ptr();
		}

	}

}

static void append_done_segments(MalVec& v, const MalFor& f, long local_origin, long from_local, long to_local) {

	if (to_local <= from_local || from_local >= to_local) {

		return;

	}

	if (f.plan_ranges.size() == 1 && f.plan_local_bases.size() >= 1) {

		const auto [gs, ge] = f.plan_ranges[0];
		const long base = local_origin + f.plan_local_bases[0];
		const long rs = std::max(from_local, base);
		const long re = std::min(to_local, base + (ge - gs));

		if (rs < re) {

			v.done_segs.push_back({gs + (rs - base), re - rs});

		}

		return;

	}

	v.done_segs.reserve(f.plan_ranges.size());

	for (size_t ri = 0; ri < f.plan_ranges.size(); ++ri) {

		if (ri >= f.plan_local_bases.size()) {

			break;

		}

		const auto [gs, ge] = f.plan_ranges[ri];
		const long base = local_origin + f.plan_local_bases[ri];
		const long len = ge - gs;
		const long rs = std::max(from_local, base);
		const long re = std::min(to_local, base + len);

		if (rs < re) {

			v.done_segs.push_back({gs + (rs - base), re - rs});

		}

	}

}

static void freeze_loop_at_current(MalFor& f) {

	long cur = *f.user_iter;
	f.start = cur;
	set_limit(f, cur);
	set_iter(f, cur);
	f.extra_ranges.clear();
	f.extra_range_local_bases.clear();
	f.extra_idx = 0;
	f.range_local_base = 0;
	f.plan_ranges.clear();
	f.plan_local_bases.clear();

}

static void distribute(long total, int nprocs, int rank, long& start, long& end) {

	long base = total / nprocs;
	long rem = total % nprocs;
	start = (long)rank * base + std::min((long)rank, rem);
	end = start + base + (rank < rem ? 1 : 0);

}

static std::vector<int> make_displs(const std::vector<int>& counts) {

	std::vector<int> d(counts.size());
	std::exclusive_scan(counts.begin(), counts.end(), d.begin(), 0);
	return d;

}

void MalVec::sync_user_ptr() {

	if (!user_ptr) {

		return;

	}

	if (halo_n > 0 && halo_buf) {

		long active_start = buf_global_start + (long)done_n;
		*user_ptr = static_cast<char*>(halo_buf) + (long)halo_n * (long)elem_size - active_start * (long)elem_size;

	} else {

		*user_ptr = static_cast<char*>(buf) - buf_global_start * (long)elem_size;

	}

}

void MalVec::free_resources() {

	if (buf) {

		pool_free(buf, 0);
		buf = nullptr;

	}

	if (halo_buf) {

		pool_free(halo_buf, halo_buf_bytes);
		halo_buf = nullptr;
		halo_buf_bytes = 0;

	}

	if (cart_comm != MPI_COMM_NULL) {

		MPI_Comm_free(&cart_comm);
		cart_comm = MPI_COMM_NULL;

	}

}

void SharedMat::free_resources() {

	if (!user_owned && buf) {

		std::free(buf);

	}

	buf = nullptr;

	if (user_ptr) {

		*user_ptr = nullptr;

	}

}

static void compute_halo_neighbors(MalVec& v) {

	if (g.comm.active == MPI_COMM_NULL || v.halo_n <= 0) {

		v.halo_lnbr = v.halo_rnbr = MPI_PROC_NULL;

		if (v.cart_comm != MPI_COMM_NULL) {

			MPI_Comm_free(&v.cart_comm);
			v.cart_comm = MPI_COMM_NULL;

		}

		return;

	}

	int dims[1] = {g.comm.a_size};
	int periods[1] = {v.halo_mode == MAL_HALO_PERIODIC ? 1 : 0};

	if (v.cart_comm != MPI_COMM_NULL) {

		int cached_dims[1], cached_periods[1];
		MPI_Cart_get(v.cart_comm, 1, cached_dims, cached_periods, nullptr);

		if (cached_dims[0] == g.comm.a_size && cached_periods[0] == periods[0]) {

			MPI_Cart_shift(v.cart_comm, 0, 1, &v.halo_lnbr, &v.halo_rnbr);
			return;

		} else {

			MPI_Comm_free(&v.cart_comm);
			v.cart_comm = MPI_COMM_NULL;

		}

	}

	MPI_Cart_create(g.comm.active, 1, dims, periods, 0, &v.cart_comm);

	if (v.cart_comm != MPI_COMM_NULL) {

		MPI_Cart_shift(v.cart_comm, 0, 1, &v.halo_lnbr, &v.halo_rnbr);

	} else {

		v.halo_lnbr = v.halo_rnbr = MPI_PROC_NULL;

	}

}

static std::vector<std::pair<long,long>> slice_remaining(const std::vector<std::pair<long,long>>& remaining, long vstart, long vend) {

	std::vector<std::pair<long,long>> out;
	out.reserve(remaining.size());

	long offset = 0;

	for (auto [a, b] : remaining) {

		long len = b - a;

		if (offset + len <= vstart) {

			offset += len;

			continue;

		}

		if (offset >= vend) {

			break;

		}

		long s = a + std::max(0L, vstart - offset);
		long e = a + std::min(len, vend - offset);

		if (s < e) {

			out.push_back({s, e});

		}

		offset += len;

	}

	return out;

}

class Resizer {

	std::vector<std::pair<long,long>> remaining_;
	long total_rem_{0};

	struct VecTask {
		MalVec* v{nullptr};
		size_t esz{0};
		char* gathered{nullptr};
		size_t gathered_bytes{0};
	};

	std::vector<VecTask> vtasks_;
	std::vector<long> rem_per_rank_;

	std::vector<std::vector<char>> new_epoch_bufs_;

	void collect_ranges();
	void redistribute_vecs();
	void reduce_accs();
	void apply_active();
	void apply_inactive();
	void broadcast_shared_mats();

	int target_;
	int old_a_size_{0};
	long my_new_vs_{0};
	long my_new_count_{0};


public:

	explicit Resizer(int target) : target_(target) {}

	~Resizer() {

		for (auto& t : vtasks_) {

			if (t.gathered) {

				pool_free(t.gathered, t.gathered_bytes);

			}

		}

	}

	void run();

};

void Resizer::collect_ranges() {

	std::vector<long> local;
	local.reserve(16);

	if (g.loop && g.comm.active != MPI_COMM_NULL) {

		auto push = [&](long s, long e) {

			if (s < e) {

				local.push_back(s);
				local.push_back(e);

			}

		};

		push(*g.loop->user_iter + 1, g.loop->end);

		for (size_t ri = g.loop->extra_idx; ri < g.loop->extra_ranges.size(); ri++) {

			push(g.loop->extra_ranges[ri].first, g.loop->extra_ranges[ri].second);

		}

	}

	int my_count = (int)local.size();
	int max_count = 0;

	MPI_Allreduce(&my_count, &max_count, 1, MPI_INT, MPI_MAX, g.comm.universe);

	if (max_count == 0) {

		remaining_.clear();
		total_rem_ = 0;
		rem_per_rank_.assign(g.comm.u_size, 0);
		g.sync.stop = true;
		return;

	}

	std::vector<int> flat_counts(g.comm.u_size);

	MPI_Allgather(&my_count, 1, MPI_INT, flat_counts.data(), 1, MPI_INT, g.comm.universe);

	std::vector<int> flat_displs = make_displs(flat_counts);

	int total = flat_displs.back() + flat_counts.back();
	std::vector<long> flat(total > 0 ? total : 1);

	MPI_Allgatherv(local.empty() ? nullptr : local.data(), my_count, MPI_LONG, flat.empty() ? nullptr : flat.data(), flat_counts.data(), flat_displs.data(), MPI_LONG, g.comm.universe);

	remaining_.clear();
	remaining_.reserve(total / 2 + 1);

	total_rem_ = 0;
	rem_per_rank_.assign(g.comm.u_size, 0);

	for (int k = 0; k < g.comm.u_size; k++) {

		int disp = flat_displs[k];
		int nranges = flat_counts[k] / 2;

		for (int p = 0; p < nranges; p++) {

			long s = flat[disp + p * 2], e = flat[disp + p * 2 + 1];
			long len = e - s;

			remaining_.push_back({s, e});

			rem_per_rank_[k] += len;
			total_rem_ += len;

		}

	}

	if (total_rem_ == 0) {

		g.sync.stop = true;

	}

}

void Resizer::redistribute_vecs() {

	int nvecs = g.loop ? (int)g.loop->vecs.size() : 0;
	int n = nvecs;

	MPI_Bcast(&n, 1, MPI_INT, 0, g.comm.universe);

	if (n == 0) {

		return;

	}

	std::vector<size_t> esizes(n, 0);

	for (int vi = 0; vi < nvecs; vi++)
		esizes[vi] = g.loop->vecs[vi]->elem_size;

	MPI_Bcast(esizes.data(), n * (int)sizeof(size_t), MPI_BYTE, 0, g.comm.universe);

	std::vector<long> old_vs(g.comm.u_size + 1, 0);
	std::inclusive_scan(rem_per_rank_.begin(), rem_per_rank_.end(), old_vs.begin() + 1);

	struct Transfer { int old_rank, new_rank; long v_start, v_count; };

	std::vector<Transfer> plan;

	plan.reserve(g.comm.u_size + target_);

	int oi = 0, ni = 0;

	long nv_s{0}, nv_e{0};

	if (target_ > 0) {

		distribute(total_rem_, target_, 0, nv_s, nv_e);

	}

	while (oi < g.comm.u_size && rem_per_rank_[oi] == 0) {

		oi++;

	}

	while (oi < g.comm.u_size && ni < target_) {

		long ov_e = old_vs[oi + 1];
		long seg_s = std::max(old_vs[oi], nv_s);
		long seg_e = std::min(ov_e, nv_e);

		if (seg_s < seg_e) {

			plan.push_back({oi, ni, seg_s, seg_e - seg_s});

		}

		if (ov_e <= nv_e) {

			oi++;

			while (oi < g.comm.u_size && rem_per_rank_[oi] == 0) {

				oi++;

			}

		}

		if (nv_e <= ov_e) {

			if (ni++ < target_) {

				distribute(total_rem_, target_, ni, nv_s, nv_e);

			}

		}

	}

	bool was_active = (g.comm.active != MPI_COMM_NULL);

	vtasks_.resize(n);

	for (int vi = 0; vi < n; vi++) {

		auto& t = vtasks_[vi];

		t.esz = esizes[vi];
		t.v = (vi < nvecs) ? g.loop->vecs[vi] : nullptr;

		if (!t.v) {

			continue;

		}

		long nd = t.v->done_n;

		if (was_active) {

			nd = std::clamp(*g.loop->user_iter - t.v->buf_global_start + 1, 0L, t.v->local_n);

		}

		if (nd > t.v->done_n) {

			append_done_segments(*t.v, *g.loop, t.v->plan_origin_n, t.v->done_n, nd);

		}

		t.v->done_n = nd;

	}

	my_new_vs_ = 0;
	my_new_count_ = 0;
	bool am_receiver = (g.comm.u_rank < target_);

	if (am_receiver) {

		long my_nv_e;

		distribute(total_rem_, target_, g.comm.u_rank, my_new_vs_, my_nv_e);

		my_new_count_ = my_nv_e - my_new_vs_;

	}

	bool will_receive_data = false;
	if (am_receiver && my_new_count_ > 0) {
		for (const auto& tr : plan) {
			if (tr.new_rank == g.comm.u_rank) {
				will_receive_data = true;
				break;
			}
		}
	}

	if (will_receive_data) {

		for (int vi = 0; vi < n; vi++) {

			size_t bytes = my_new_count_ * esizes[vi];
			vtasks_[vi].gathered = static_cast<char*>(pool_alloc(bytes));
			vtasks_[vi].gathered_bytes = bytes;

		}

	}

	std::vector<MPI_Request> reqs;
	reqs.reserve(plan.size() * (size_t)n * 2);

	for (const auto& tr : plan) {

		for (int vi = 0; vi < n; vi++) {

			auto& t = vtasks_[vi];
			long bytes64 = tr.v_count * (long)t.esz;

			if (bytes64 > INT_MAX) {

				MAL_LOG_L(MAL_LOG_ERROR, "RESIZE", "Transfer size overflow (%ld bytes) in redistribute_vecs", bytes64);
				MPI_Abort(g.comm.universe, 1);

			}

			int byte_count = (int)bytes64;

			if (byte_count == 0) {

				continue;

			}

			const char* send_base = (t.v && was_active) ? static_cast<char*>(t.v->buf) + t.v->done_n * t.esz : nullptr;

		if (tr.old_rank == tr.new_rank) {

			if (tr.old_rank == g.comm.u_rank && send_base && t.gathered) {

				long src_off = (tr.v_start - old_vs[g.comm.u_rank]) * (long)t.esz;
				long dst_off = (tr.v_start - my_new_vs_ ) * (long)t.esz;

				std::memmove(t.gathered + dst_off, send_base + src_off, byte_count);

			}

			continue;

		}

			if (tr.old_rank == g.comm.u_rank && send_base) {

				MPI_Request req;

				MPI_Isend(send_base + (tr.v_start - old_vs[tr.old_rank]) * (long)t.esz, byte_count, MPI_BYTE, tr.new_rank, vi, g.comm.universe, &req);

				reqs.push_back(req);

			}

			if (tr.new_rank == g.comm.u_rank && t.gathered) {

				char* dst = t.gathered + (tr.v_start - my_new_vs_) * (long)t.esz;

				MPI_Request req;

				MPI_Irecv(dst, byte_count, MPI_BYTE, tr.old_rank, vi, g.comm.universe, &req);

				reqs.push_back(req);

			}

		}

	}

	MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

}

void Resizer::reduce_accs() {

	int naccs = g.loop ? (int)g.loop->accs.size() : 0;
	int n = naccs;

	MPI_Bcast(&n, 1, MPI_INT, 0, g.comm.universe);

	if (n == 0) {

		return;

	}

	new_epoch_bufs_.resize(n);

	batched_allreduce(n,
		[naccs](int k) -> MalAcc* { return k < naccs ? g.loop->accs[k] : nullptr; },
		[this, naccs](int k, const char* r, int esz) {

			new_epoch_bufs_[k].assign(r, r + esz);

			if (k >= naccs) {

				return;

			}

			MalAcc* a = g.loop->accs[k];

			a->epoch_buf.assign(r, r + esz);
			a->fn_reset(a->ptr);

		});

}

void Resizer::apply_active() {

	long vstart, vend;
	distribute(total_rem_, g.comm.a_size, g.comm.a_rank, vstart, vend);
	auto assigned = slice_remaining(remaining_, vstart, vend);

	long new_asgn = vend - vstart;

	MAL_LOG_L(MAL_LOG_INFO, "RESIZE", "a_rank=%d assigned %zu range(s) (%ld iters)", g.comm.a_rank, assigned.size(), new_asgn);

	if (g.loop) {

		for (auto& t : vtasks_) {

			if (!t.v) {

				continue;

			}

			long new_local = t.v->done_n + new_asgn;

			t.v->buf = checked_realloc(t.v->buf, std::max(1L, new_local) * (long)t.esz, "apply_active");

			if (new_asgn > 0 && t.gathered && !assigned.empty()) {

				long buf_off = t.v->done_n;

				for (auto [g_start, g_end] : assigned) {

					long len = g_end - g_start;

					if (len > 0) {

						long src_off = g_start - my_new_vs_;

						if (src_off >= 0 && src_off + len <= my_new_count_) {

							std::memcpy(static_cast<char*>(t.v->buf) + buf_off * t.esz, static_cast<char*>(t.gathered) + src_off * t.esz, len * t.esz);

						}

						buf_off += len;

					}

				}

			}

			t.v->local_n = new_local;
			t.v->plan_origin_n = t.v->done_n;

			if (!assigned.empty()) {

				t.v->buf_global_start = assigned[0].first - t.v->done_n;

			}

			t.v->sync_user_ptr();

			std::free(t.gathered); t.gathered = nullptr;

		}

		if (!assigned.empty()) {

			g.loop->start = assigned[0].first;
			g.loop->range_local_base = 0;

			set_limit(*g.loop, assigned[0].second);
			set_iter (*g.loop, assigned[0].first);

			g.loop->plan_ranges = assigned;
			g.loop->plan_local_bases.clear();
			g.loop->plan_local_bases.reserve(assigned.size());
			g.loop->plan_local_bases.push_back(0);

			if (assigned.size() == 1) {

				g.loop->extra_ranges.clear();
				g.loop->extra_range_local_bases.clear();

			} else {

				g.loop->extra_ranges.assign(assigned.begin() + 1, assigned.end());
				g.loop->extra_range_local_bases.clear();
				g.loop->extra_range_local_bases.reserve(assigned.size() - 1);

				long local_base = assigned[0].second - assigned[0].first;

				for (size_t i = 1; i < assigned.size(); ++i) {

					g.loop->extra_range_local_bases.push_back(local_base);
					g.loop->plan_local_bases.push_back(local_base);
					local_base += assigned[i].second - assigned[i].first;

				}

			}

		} else {

			freeze_loop_at_current(*g.loop);

		}

	} else {

		auto pa = std::make_unique<PendingActivation>();

		pa->ranges = std::move(assigned);
		pa->range_local_bases.clear();
		pa->range_local_bases.reserve(pa->ranges.size());
		long local_base = 0;
		for (const auto& rg : pa->ranges) {

			pa->range_local_bases.push_back(local_base);
			local_base += rg.second - rg.first;

		}
		pa->vec_slices.resize(vtasks_.size(), nullptr);

		for (int vi = 0; vi < (int)vtasks_.size(); vi++) {

			auto& t = vtasks_[vi];

			if (new_asgn > 0 && t.gathered && t.esz > 0) {

				pa->vec_slices[vi] = t.gathered;
				t.gathered = nullptr;

			}

		}

		pa->acc_epoch_bufs = std::move(new_epoch_bufs_);
		g.pending = std::move(pa);
		g.sync.compute_ready = true;

	}

	broadcast_shared_mats();

	for (auto& vp : g.vecs) {

		if (vp->halo_n > 0) {

			compute_halo_neighbors(*vp);

		}

	}

}

void Resizer::broadcast_shared_mats() {

	bool any_new = (target_ > old_a_size_);
	int n_shared = (int)g.shared.size();

	MPI_Bcast(&n_shared, 1, MPI_INT, 0, g.comm.active);

	if (!any_new || n_shared == 0) {

		return;

	}

	bool is_new = (g.loop == nullptr);

	std::vector<size_t> tots(n_shared, 0);

	if (!is_new) {

		for (int si = 0; si < n_shared; si++) {

			tots[si] = g.shared[si]->total_bytes;

		}

	}

	MPI_Bcast(tots.data(), n_shared * (int)sizeof(size_t), MPI_BYTE, 0, g.comm.active);

	if (is_new) g.pending->shared_mats.reserve(n_shared);

	for (int si = 0; si < n_shared; si++) {

		size_t tot = tots[si];
		void* buf = is_new ? std::malloc(tot > 0 ? tot : 1) : g.shared[si]->buf;

		MPI_Bcast(buf, (int)tot, MPI_BYTE, 0, g.comm.active);

		if (is_new) g.pending->shared_mats.push_back(buf);

	}

}

void Resizer::apply_inactive() {

	g.comm.a_rank = -1;
	g.comm.a_size = 0;

	for (auto& t : vtasks_) {

		std::free(t.gathered); t.gathered = nullptr;

		if (!t.v) {

			continue;

		}

		t.v->buf = checked_realloc(t.v->buf, std::max(1L, t.v->done_n) * (long)t.esz, "apply_inactive");
		t.v->local_n = t.v->done_n;

		t.v->sync_user_ptr();

	}

	if (g.loop) {

		freeze_loop_at_current(*g.loop);

	}

	g.sync.compute_ready = true;

}

void Resizer::run() {

	if (target_ == g.comm.a_size) {

		return;

	}

	g.sync.wait_for_compute();
	MPI_Barrier(g.comm.universe);

	MAL_LOG_L(MAL_LOG_INFO, "RESIZE", "Starting resize to %d (current=%d)", target_, g.comm.a_size);
	double t0 = MPI_Wtime();

	old_a_size_ = g.comm.a_size;

	collect_ranges();
	redistribute_vecs();
	reduce_accs();

	if (g.comm.active != MPI_COMM_NULL) {

		MPI_Comm_free(&g.comm.active);
		g.comm.active = MPI_COMM_NULL;

	}

	int color = (g.comm.u_rank < target_) ? 0 : MPI_UNDEFINED;
	MPI_Comm_split(g.comm.universe, color, g.comm.u_rank, &g.comm.active);

	if (g.comm.active != MPI_COMM_NULL) {

		MPI_Comm_rank(g.comm.active, &g.comm.a_rank);
		MPI_Comm_size(g.comm.active, &g.comm.a_size);

		apply_active();

	} else {

		apply_inactive();

	}

	MAL_LOG_L(MAL_LOG_INFO, "RESIZE", "Resize to %d done in %.4f s", target_, MPI_Wtime() - t0);

}

static void progress_thread() {

	{

		std::lock_guard lk(g.cfg.mu);

		if (g.cfg.sequence.empty()) {

			g.cfg.sequence.assign(std::begin(kDefaultSeq), std::end(kDefaultSeq));
			g.cfg.seq_idx = 0;

		}

	}

	while (!g.sync.stop) {

		for (;;) {

			std::shared_ptr<MalState::AsyncBcastTask> task;

			{
				std::lock_guard lk(g.attach_mu);
				if (g.attach_tasks.empty()) {
					g.sync.attach_pending = false;
					break;
				}
				task = g.attach_tasks.front();
				g.attach_tasks.pop_front();
			}

			mpi_bcast_bytes(task->buf, task->bytes, task->root, g.comm.universe);

			{
				std::lock_guard lk(task->mu);
				task->done = true;
			}

			task->cv.notify_all();
			g.sync.notify();

		}

		const int wait_ms = g.cfg.epoch_ms.load();

		{

			std::unique_lock lk(g.sync.mu);

			g.sync.cv.wait_for(lk, std::chrono::milliseconds(wait_ms > 0 ? wait_ms : MAL_EPOCH_INTERVAL_MS), [] { return g.sync.stop.load() || g.sync.attach_pending.load(); });

		}

		if (g.sync.stop) {

			break;

		}

		int target = -1;
		size_t idx = 0;

		{

			std::lock_guard lk(g.cfg.mu);

			if (g.cfg.enabled.load() && g.cfg.seq_idx < g.cfg.sequence.size()) {

				idx = g.cfg.seq_idx;
				target = g.cfg.sequence[g.cfg.seq_idx];

			}

		}

		if (target >= 0) {

			MAL_LOG_L(MAL_LOG_INFO, "EPOCH", "[%zu] target=%d", idx, target);

			g.sync.resize_pending = true;

			Resizer(target).run();

			bool done = false;

			{
				std::lock_guard lk(g.cfg.mu);
				done = (++g.cfg.seq_idx >= g.cfg.sequence.size());
			}

			if (done) {

				g.cfg.enabled.store(false);

			}

			g.sync.resize_pending = false;

			MAL_LOG_L(MAL_LOG_INFO, "EPOCH", "[%zu] done (active=%d)", idx, g.comm.a_size);

		}

		g.sync.notify();
	}

	g.sync.notify();

}

void mal_set_epoch_interval_ms(int ms) {

	if (ms > 0) {

		g.cfg.epoch_ms.store(ms);

	} else {

		MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring invalid epoch interval ms=%d (must be > 0)", ms);

	}

	g.sync.notify();

}

void mal_set_resize_enabled(bool b) {

	g.cfg.enabled.store(b);
	g.sync.notify();

}

void mal_set_resize_sequence(const int* seq, size_t count) {

	std::lock_guard lk(g.cfg.mu);

	if (seq && count > 0) {

		g.cfg.sequence.clear();
		g.cfg.sequence.reserve(count);

		for (size_t i = 0; i < count; ++i) {

			int target = seq[i];

			if (target <= 0 || target > g.comm.u_size) {

				MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring invalid resize target seq[%zu]=%d (valid range: 1..%d)", i, target, g.comm.u_size);
				continue;

			}

			g.cfg.sequence.push_back(target);

		}

		if (g.cfg.sequence.empty()) {

			MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Resize sequence became empty after filtering invalid values");

		}

	} else {

		if (count > 0 && !seq) {

			MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring resize sequence: seq is null but count=%zu", count);

		}

		g.cfg.sequence.clear();

	}

	g.cfg.seq_idx = 0;
	g.sync.notify();

}

void mal_reset_resize_sequence_default() {

	mal_set_resize_sequence(kDefaultSeq, std::size(kDefaultSeq));

}

static void load_env_config() {

	if (const char* v = std::getenv("MAL_EPOCH_INTERVAL_MS")) {

		long ms = std::strtol(v, nullptr, 10);

		if (ms > 0) {

			g.cfg.epoch_ms.store((int)ms);

		} else {

			MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring MAL_EPOCH_INTERVAL_MS='%s' (must be > 0)", v);

		}

	}

	if (const char* v = std::getenv("MAL_RESIZE_ENABLED")) {

		g.cfg.enabled.store(std::strtol(v, nullptr, 10) != 0);

	}

	if (const char* v = std::getenv("MAL_RESIZE_SEQ")) {

		std::vector<int> seq;
		char* end;
		bool found_invalid = false;

		while (*v) {

			long n = std::strtol(v, &end, 10);

			if (end == v) {

				found_invalid = true;
				break;

			}

			if (n > 0) {

				seq.push_back((int)n);

			} else {

				found_invalid = true;

			}

			v = end;

			while (*v == ',' || *v == ' ') ++v;

		}

		if (!seq.empty()) {

			mal_set_resize_sequence(seq.data(), seq.size());

		} else {

			MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring MAL_RESIZE_SEQ because it has no valid positive targets");

		}

		if (found_invalid) {

			MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "MAL_RESIZE_SEQ contains invalid tokens and/or non-positive values");

		}

	}

}

void mal_init() {

	load_env_config();

	MPI_Session_init(MPI_INFO_NULL, MPI_ERRORS_RETURN, &g.comm.session);
	MPI_Group_from_session_pset(g.comm.session, "mpi://WORLD", &g.comm.world_group);
	MPI_Comm_create_from_group(g.comm.world_group, "malleable.universe", MPI_INFO_NULL, MPI_ERRORS_RETURN, &g.comm.universe);
	MPI_Comm_rank(g.comm.universe, &g.comm.u_rank);
	MPI_Comm_size(g.comm.universe, &g.comm.u_size);

	if (MAL_INITIAL_SIZE <= 0 || MAL_INITIAL_SIZE > g.comm.u_size) {

		if (g.comm.u_rank == 0) {

			MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "MAL_INITIAL_SIZE=%d is invalid for universe size=%d", MAL_INITIAL_SIZE, g.comm.u_size);

		}

	}

	int color = (g.comm.u_rank < MAL_INITIAL_SIZE) ? 0 : MPI_UNDEFINED;
	MPI_Comm_split(g.comm.universe, color, g.comm.u_rank, &g.comm.active);

	g.worker = std::thread(progress_thread);

	if (g.comm.active != MPI_COMM_NULL) {

		MPI_Comm_rank(g.comm.active, &g.comm.a_rank);
		MPI_Comm_size(g.comm.active, &g.comm.a_size);

	} else {

		g.comm.a_rank = -1;
		g.comm.a_size = 0;

		g.sync.compute_wait([] {

			return g.comm.active != MPI_COMM_NULL || g.sync.stop.load();

		});

	}

}

static void vec_scatter(MalVec& v, const void* root_data) {
	std::vector<int> sc, sd;

	if (g.comm.a_rank == 0) {

		sc.resize(g.comm.a_size);

		for (int k = 0; k < g.comm.a_size; ++k) {

			long ks, ke;
			distribute(v.total_N, g.comm.a_size, k, ks, ke);
			sc[k] = (int)((ke - ks) * (long)v.elem_size);

		}

		sd = make_displs(sc);

	}
	int rc = (int)(v.local_n * (long)v.elem_size);

	MPI_Scatterv(root_data, g.comm.a_rank == 0 ? sc.data() : nullptr, g.comm.a_rank == 0 ? sd.data() : nullptr, MPI_BYTE, v.buf, rc, MPI_BYTE, 0, g.comm.active);

}

static void vec_gather(MalVec& v) {

	if (v.total_N == 0) {

		return;

	}

	if (v.local_n > v.done_n) {

		if (g.loop) {

			append_done_segments(v, *g.loop, v.plan_origin_n, v.done_n, v.local_n);

		} else {

			v.done_segs.push_back({v.buf_global_start + v.done_n, v.local_n - v.done_n});

		}

	}

	const int usiz = g.comm.u_size;
	const bool root = (g.comm.u_rank == v.gather_root);

	std::vector<long> my_seg_flat;

	my_seg_flat.reserve(v.done_segs.size() * 2);

	for (auto [s, c] : v.done_segs) {

		my_seg_flat.push_back(s); my_seg_flat.push_back(c);

	}

	int my_seg_count = (int)my_seg_flat.size();

	std::vector<int> seg_counts, seg_displs;

	if (root) {

		seg_counts.resize(usiz);

	}

	MPI_Gather(&my_seg_count, 1, MPI_INT, root ? seg_counts.data() : nullptr, 1, MPI_INT, v.gather_root, g.comm.universe);

	std::vector<long> all_segs;

	if (root) {

		seg_displs = make_displs(seg_counts);
		all_segs.resize(seg_displs.back() + seg_counts.back());

	}

	MPI_Gatherv(my_seg_flat.empty() ? nullptr : my_seg_flat.data(), my_seg_count, MPI_LONG, all_segs.empty() ? nullptr : all_segs.data(), root ? seg_counts.data() : nullptr, root ? seg_displs.data() : nullptr, MPI_LONG, v.gather_root, g.comm.universe);

	long my_data_bytes = v.local_n * (long)v.elem_size;

	std::vector<int> data_counts, data_displs;
	std::vector<char> recv_buf;

	if (root) {

		data_counts.resize(usiz);

		for (int k = 0; k < usiz; ++k) {

			long total = 0;

			for (int s = 0; s < seg_counts[k]/2; ++s) {

				total += all_segs[seg_displs[k] + s*2 + 1];

			}

			data_counts[k] = (int)(total * (long)v.elem_size);

		}

		data_displs = make_displs(data_counts);
		recv_buf.resize(data_displs.back() + data_counts.back());

	}

	MPI_Gatherv(my_data_bytes > 0 ? v.buf : nullptr, (int)my_data_bytes, MPI_BYTE, recv_buf.empty() ? nullptr : recv_buf.data(), data_counts.empty() ? nullptr : data_counts.data(), data_displs.empty() ? nullptr : data_displs.data(), MPI_BYTE, v.gather_root, g.comm.universe);

	if (root && v.result_buf) {

		long data_off = 0;

		for (int k = 0; k < usiz; ++k) {

			for (int s = 0; s < seg_counts[k]/2; ++s) {

				long gs = all_segs[seg_displs[k] + s*2];
				long cnt = all_segs[seg_displs[k] + s*2 + 1];

				std::memcpy(static_cast<char*>(v.result_buf) + gs * (long)v.elem_size, recv_buf.data() + data_off, cnt * (long)v.elem_size);
				data_off += cnt * (long)v.elem_size;

			}

		}

	}

}

void mal_finalize() {

	g.sync.stop = true;

	g.sync.notify();
	g.worker.join();

	MPI_Barrier(g.comm.universe);

	for (auto& vp : g.vecs) {

		MalVec& v = *vp;

		if (!v.buf) {

			continue;

		}

		if (v.gather_root >= 0) {

			vec_gather(v);

		}

		if (v.user_ptr) {

			*v.user_ptr = (g.comm.u_rank == v.gather_root && v.result_buf) ? v.result_buf : nullptr;

		}

		v.free_resources();

	}

	g.vecs.clear();

	int naccs = (int)g.accs.size();

	MPI_Bcast(&naccs, 1, MPI_INT, 0, g.comm.universe);
	batched_allreduce(naccs,
		[](int k) -> MalAcc* { return g.accs[k].get(); },
		[](int k, const char* r, int) {

			MalAcc* a = g.accs[k].get();

			if (!a->ptr) {
				return;
			}

			if (g.comm.u_rank == a->result_rank) {

				a->fn_set(a->ptr, r);

			} else {

				a->fn_reset(a->ptr);

			}

		}
	);

	g.accs.clear();

	for (auto& sp : g.shared) {

		sp->free_resources();

	}

	g.shared.clear();

	if (g.pending) {

		g.pending.reset();

	}

	if (g.comm.active != MPI_COMM_NULL) {

		MPI_Comm_free(&g.comm.active);

	}

	MPI_Comm_free(&g.comm.universe);
	MPI_Group_free(&g.comm.world_group);
	MPI_Session_finalize(&g.comm.session);

}

MalFor mal_for(long total_iters, long& iter, long& limit) {

	MalFor f;

	f.user_iter = &iter;
	f.user_limit = &limit;

	if (g.pending && !g.pending->ranges.empty()) {

		f.start = g.pending->ranges[0].first;
		f.end = g.pending->ranges[0].second;
		f.range_local_base = 0;

		f.plan_ranges = g.pending->ranges;
		f.plan_local_bases = g.pending->range_local_bases;

		if (g.pending->ranges.size() > 1) {

			f.extra_ranges.assign(g.pending->ranges.begin() + 1, g.pending->ranges.end());
			if (g.pending->range_local_bases.size() > 1) {

				f.extra_range_local_bases.assign(g.pending->range_local_bases.begin() + 1, g.pending->range_local_bases.end());

			}

		}

		g.pending->ranges.clear();
		g.pending->range_local_bases.clear();

	} else if (g.comm.a_size > 0 && !g.sync.stop) {

		distribute(total_iters, g.comm.a_size, g.comm.a_rank, f.start, f.end);
		f.range_local_base = 0;
		f.plan_ranges = {{f.start, f.end}};
		f.plan_local_bases = {0};

	} else {

		f.start = f.end = 0;
		f.plan_ranges.clear();
		f.plan_local_bases.clear();

	}

	set_iter(f, f.start);
	limit = f.end;

	g.sync.compute_ready = false;
	g.loop = &f;

	while (f.start == f.end && !g.sync.stop) {

		f.current = f.end;
		g.sync.compute_wait(has_work_or_stop);

	}

	return f;

}

static void advance_next_range(MalFor& f) {

	size_t idx = f.extra_idx++;
	auto [a, b] = f.extra_ranges[idx];

	f.start = a;
	f.range_local_base = (idx < f.extra_range_local_bases.size()) ? f.extra_range_local_bases[idx] : 0;
	sync_vec_mapping_for_current_range(f);

	set_limit(f, b);

	prime_range_start(f);

	MAL_LOG_L(MAL_LOG_DEBUG, "RANGE", "Next range [%ld, %ld) (base=%ld)", a, b, f.range_local_base);

}

void mal_check_for(MalFor& f) {

	if (g.sync.attach_pending) {

		g.sync.compute_wait([] {

			return !g.sync.attach_pending.load() || g.sync.stop.load();

		});

	}

	f.current = *f.user_iter;

	if (g.sync.resize_pending) {

		g.sync.compute_wait([] {

			return !g.sync.resize_pending.load() || g.sync.stop.load();

		});

		if (g.comm.active != MPI_COMM_NULL && f.start < f.end) {

			mal_exchange_halo(f);
			prime_range_start(f);

			return;

		}

	}

	if (*f.user_iter + 1 < f.end) {

		return;

	}

	if (f.extra_idx < f.extra_ranges.size()) {

		advance_next_range(f);

		return;

	}

	if (g.sync.stop) {

		return;

	}

	f.current = f.end;
	g.sync.compute_wait(has_work_or_stop);

	if (!g.sync.stop) {

		mal_exchange_halo(f);
		prime_range_start(f);

	}

}

void mal_attach_vec(MalFor& f, void** user_ptr, size_t elem_size, long total_N, int result_rank, MalAttachPolicy policy) {

	auto vp = std::make_unique<MalVec>();
	MalVec* v = vp.get();

	void* orig = user_ptr ? *user_ptr : nullptr;
	long n = f.end - f.start;

	v->elem_size = elem_size;
	v->local_n = n;
	v->buf_global_start = f.start;
	v->total_N = total_N;
	v->user_ptr = user_ptr;
	v->gather_root = result_rank;
	v->fully_replicated = false;

	if (result_rank >= 0 && g.comm.u_rank == result_rank) {

		v->result_buf = orig ? orig : checked_realloc(nullptr, total_N > 0 ? (size_t)total_N * elem_size : 1, "mal_attach_vec.result_buf");

	}

	bool once_all = (policy == MAL_ATTACH_ONCE_ALL);

	if (elem_size == 0) {

		MAL_LOG_L(MAL_LOG_WARN, "ATTACH", "mal_attach_vec called with elem_size=0");

	}

	if (total_N < 0) {

		MAL_LOG_L(MAL_LOG_WARN, "ATTACH", "mal_attach_vec called with negative total_N=%ld", total_N);

	}

	if (once_all && result_rank >= 0) {

		MAL_LOG_L(MAL_LOG_WARN, "ATTACH", "MAL_ATTACH_ONCE_ALL ignored for gather vector (result_rank=%d)", result_rank);
		once_all = false;

	}

	if (once_all) {

		n = total_N;
		v->local_n = n;
		v->done_n = 0;
		v->buf_global_start = 0;
		v->fully_replicated = true;

	}

	v->buf = static_cast<char*>(pool_alloc((v->local_n > 0 ? (size_t)v->local_n : 1) * elem_size));
	v->plan_origin_n = v->done_n;

	int idx = (int)f.vecs.size();

	if (g.pending && idx < (int)g.pending->vec_slices.size()) {

		auto stash = g.pending->vec_slices[idx];

		if (stash && n > 0) {

			std::memcpy(v->buf, stash, (size_t)n * elem_size);
			std::free(stash);

			g.pending->vec_slices[idx] = nullptr;

		}

	}

	v->sync_user_ptr();

	f.vecs.push_back(v);
	g.vecs.push_back(std::move(vp));

	if (!g.pending && once_all) {

		size_t total_bytes = (size_t)std::max(0L, total_N) * elem_size;

		if (g.comm.u_rank == 0 && total_bytes > 0 && !orig) {

			MAL_LOG_L(MAL_LOG_WARN, "ATTACH", "MAL_ATTACH_ONCE_ALL vector has null root pointer; broadcasting zero-initialized data");

		}

		if (total_bytes > 0) {

			std::memset(v->buf, 0, total_bytes);

		}

		if (g.comm.u_rank == 0 && orig && total_N > 0) {

			std::memcpy(v->buf, orig, (size_t)total_N * elem_size);

		}

			run_async_bcast_once_all(v->buf, total_bytes, 0);

		if (g.comm.u_rank == 0 && orig && result_rank < 0) {

			std::free(orig);

		}

	} else if (!g.pending && g.comm.active != MPI_COMM_NULL) {

		int do_scatter = (orig && result_rank < 0) ? 1 : 0;

		MPI_Bcast(&do_scatter, 1, MPI_INT, 0, g.comm.active);

		if (do_scatter) {

			vec_scatter(*v, orig);

			if (g.comm.u_rank == 0) {

				std::free(orig);

			}

		}

	}

	if (g.pending && idx + 1 == (int)g.pending->vec_slices.size()) {

		g.pending->vec_slices.clear();

	}

}

void detail::acc_register(MalFor& f, detail::AccDesc d, int result_rank) {

	auto ap = std::make_unique<MalAcc>();
	MalAcc* a = ap.get();

	a->result_rank = result_rank;
	a->ptr = d.ptr;
	a->dtype_idx = dtype_tag(d.dtype);
	a->dop_idx = dop_tag(d.dop);
	a->esz = d.esz;
	a->fn_get = d.fn_get;
	a->fn_set = d.fn_set;
	a->fn_add = d.fn_add;
	a->fn_reset = d.fn_reset;

	if (g.pending && g.pending->acc_idx < g.pending->acc_epoch_bufs.size()) {

		a->epoch_buf = std::move(g.pending->acc_epoch_bufs[g.pending->acc_idx++]);

	} else {

		a->epoch_buf.assign(a->esz, 0);

	}

	a->fn_reset(a->ptr);

	f.accs.push_back(a);
	g.accs.push_back(std::move(ap));

}

void mal_attach_mat(MalFor& f, void** user_ptr, size_t elem_size, long primary_n, long secondary_n, MalDimMode mode, int result_rank, MalAttachPolicy policy) {

	if (elem_size == 0 || primary_n < 0 || secondary_n < 0) {

		MAL_LOG_L(MAL_LOG_WARN, "ATTACH", "mal_attach_mat called with invalid shape/size elem_size=%zu primary_n=%ld secondary_n=%ld", elem_size, primary_n, secondary_n);

	}

	if (policy == MAL_ATTACH_ONCE_ALL && mode == MAL_DIM_PARTITIONED && result_rank < 0) {

		mal_attach_vec(f, user_ptr, elem_size * (size_t)secondary_n, primary_n, -1, MAL_ATTACH_ONCE_ALL);
		return;

	}

	if (policy == MAL_ATTACH_ONCE_ALL && mode == MAL_DIM_PARTITIONED && result_rank >= 0) {

		MAL_LOG_L(MAL_LOG_WARN, "ATTACH", "MAL_ATTACH_ONCE_ALL ignored for partitioned matrix with result_rank=%d", result_rank);

	}

	if (mode == MAL_DIM_PARTITIONED) {

		mal_attach_vec(f, user_ptr, elem_size * (size_t)secondary_n, primary_n, result_rank, MAL_ATTACH_DEFAULT);

		return;

	}

	bool once_all = (policy == MAL_ATTACH_ONCE_ALL);

	if (!once_all && g.comm.active == MPI_COMM_NULL) {

		return;

	}

	const size_t total_bytes = (size_t)primary_n * (size_t)secondary_n * elem_size;
	void* orig = user_ptr ? *user_ptr : nullptr;
	void* buf;

	if (g.pending && g.pending->shared_idx < g.pending->shared_mats.size() && !once_all) {

		buf = g.pending->shared_mats[g.pending->shared_idx];
		g.pending->shared_mats[g.pending->shared_idx] = nullptr;
		++g.pending->shared_idx;

	} else {

		if (once_all) {

			buf = std::malloc(total_bytes > 0 ? total_bytes : 1);

			if (total_bytes > 0) {

				std::memset(buf, 0, total_bytes);

			}

			if (g.comm.u_rank == 0 && orig && total_bytes > 0) {

				std::memcpy(buf, orig, total_bytes);

			}

			if (g.comm.u_rank == 0 && total_bytes > 0 && !orig) {

				MAL_LOG_L(MAL_LOG_WARN, "ATTACH", "MAL_ATTACH_ONCE_ALL matrix has null root pointer; broadcasting zero-initialized data");

			}

			run_async_bcast_once_all(buf, total_bytes, 0);

		} else {

			buf = (g.comm.a_rank == 0 && orig) ? orig : std::malloc(total_bytes > 0 ? total_bytes : 1);
			MPI_Bcast(buf, (int)total_bytes, MPI_BYTE, 0, g.comm.active);

		}

	}

	if (user_ptr) {

		*user_ptr = buf;

	}

	auto sp = std::make_unique<SharedMat>();

	sp->buf = buf;
	sp->total_bytes = total_bytes;
	sp->user_owned = once_all ? (g.comm.u_rank == 0 && orig != nullptr) : (g.comm.a_rank == 0 && orig != nullptr);
	sp->user_ptr = user_ptr;

	g.shared.push_back(std::move(sp));

}

void mal_attach_halo(MalFor& f, void** user_ptr, int halo, MalHaloMode mode, MalAttachPolicy policy) {

	if (halo <= 0) {

		MAL_LOG_L(MAL_LOG_WARN, "ATTACH", "Ignoring halo attach with halo=%d (must be > 0)", halo);

		return;

	}

	MalVec* v = nullptr;

	for (MalVec* vp : f.vecs) {

		if (vp->user_ptr == user_ptr) {

			v = vp;
			break;

		}

	}

	if (!v) {

		MAL_LOG_L(MAL_LOG_WARN, "ATTACH", "Ignoring halo attach: vector pointer was not previously attached");

		return;

	}

	v->halo_n = halo;
	v->halo_mode = mode;
	v->halo_static_once = (policy == MAL_ATTACH_ONCE_ALL);
	v->halo_initialized = false;

	if (v->fully_replicated && v->halo_static_once) {

		return;

	}

	long new_asgn = v->local_n - v->done_n;
	size_t bytes = (size_t)std::max(1L, (new_asgn + 2L * halo) * (long)v->elem_size);
	v->halo_buf = pool_alloc(bytes);
	v->halo_buf_bytes = bytes;

	compute_halo_neighbors(*v);
	v->sync_user_ptr();
	mal_exchange_halo(f);

	if (v->halo_static_once) {

		v->halo_initialized = true;

	}

}

void mal_exchange_halo(MalFor& f) {

	if (g.comm.active == MPI_COMM_NULL) {

		return;

	}

	for (int vi = 0; vi < (int)f.vecs.size(); ++vi) {

		MalVec& v = *f.vecs[vi];

		if (v.halo_n <= 0 || !v.halo_buf) {

			continue;

		}

		if (v.halo_static_once && v.halo_initialized) {

			continue;

		}

		if (v.fully_replicated) {

			continue;

		}

		int h = v.halo_n;
		long esz = (long)v.elem_size;
		long new_asgn = v.local_n - v.done_n;

		if (new_asgn < h) {

			continue;

		}

		int lnbr = v.halo_lnbr;
		int rnbr = v.halo_rnbr;

		size_t needed_bytes = (size_t)std::max(1L, (new_asgn + 2L * h) * esz);

		if (needed_bytes != v.halo_buf_bytes) {

			void* new_buf = pool_alloc(needed_bytes);
			if (v.halo_buf && v.halo_buf_bytes > 0) {

				std::memcpy(new_buf, v.halo_buf, std::min(needed_bytes, v.halo_buf_bytes));
				pool_free(v.halo_buf, v.halo_buf_bytes);

			}
			v.halo_buf = new_buf;
			v.halo_buf_bytes = needed_bytes;

		}

		char* hb = static_cast<char*>(v.halo_buf);
		char* owned = static_cast<char*>(v.buf) + v.done_n * esz;
		int cnt = (int)(h * esz);

		std::memcpy(hb + h * esz, owned, (size_t)(new_asgn * esz));

		MPI_Request reqs[4];

		MPI_Irecv(hb, cnt, MPI_BYTE, lnbr, vi*2, g.comm.active, &reqs[0]);
		MPI_Isend(owned + (new_asgn - h) * esz, cnt, MPI_BYTE, rnbr, vi*2, g.comm.active, &reqs[1]);
		MPI_Irecv(hb + (h + new_asgn) * esz, cnt, MPI_BYTE, rnbr, vi*2+1, g.comm.active, &reqs[2]);
		MPI_Isend(owned, cnt, MPI_BYTE, lnbr, vi*2+1, g.comm.active, &reqs[3]);
		MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

		if (v.halo_mode != MAL_HALO_PERIODIC) {

			if (lnbr == MPI_PROC_NULL) {

				if (v.halo_mode == MAL_HALO_CLAMP) {

					for (int k = 0; k < h; ++k) {

						std::memcpy(hb + k * esz, owned, (size_t)esz);

					}

				} else {

					std::memset(hb, 0, (size_t)(h * esz));

				}

			}
			if (rnbr == MPI_PROC_NULL) {

				char* rg = hb + (h + new_asgn) * esz;
				const char* edge = owned + (new_asgn - 1) * esz;

				if (v.halo_mode == MAL_HALO_CLAMP) {

					for (int k = 0; k < h; ++k) {

						std::memcpy(rg + k * esz, edge, (size_t)esz);

					}

				} else {

					std::memset(rg, 0, (size_t)(h * esz));

				}

			}

		}

		v.sync_user_ptr();

		if (v.halo_static_once) {

			v.halo_initialized = true;

		}

	}

}
