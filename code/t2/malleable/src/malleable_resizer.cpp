#ifndef MALLEABLE_RESIZER_CPP_INCLUDED
#define MALLEABLE_RESIZER_CPP_INCLUDED

#include "malleable_types.cpp"

class Resizer {

	std::vector<std::pair<long,long>> remaining_;
	std::vector<long> remaining_offsets_;
	std::vector<long> target_cuts_;
	long total_rem_{0};

	struct VecTask {
		MalVec* v{nullptr};
		StagedBuffer gathered;
	};

	struct VecMeta {
		size_t esz{0};
		int shared_active{0};
	};

	std::vector<VecTask> vtasks_;
	std::vector<VecMeta> vmeta_;
	std::vector<long> rem_per_rank_;
	std::vector<std::vector<char>> new_epoch_bufs_;
	std::vector<std::pair<long,long>> scratch_assigned_;
	std::vector<int> scratch_reuse_flags_;
	std::vector<int> scratch_all_reuse_flags_;

	void init_vec_meta(int nvecs, int n);
	std::vector<TransferPlanEntry> build_transfer_plan(const std::vector<long>& old_vs) const;
	void init_vec_tasks(int n, int nvecs, bool was_active);
	void reserve_receiver_buffers(int n, bool am_receiver, const std::vector<TransferPlanEntry>& plan, long my_new_count, const std::vector<int>& all_reuse_flags) ;
	void exchange_vec_data(int n, bool was_active, const std::vector<long>& old_vs, const std::vector<TransferPlanEntry>& plan, const std::vector<int>& all_reuse_flags);

	void collect_ranges();
	void redistribute_vecs();
	void reduce_accs();
	void apply_active();
	void apply_inactive();
	void broadcast_shared_vecs();
	void broadcast_shared_mats();
	void stash_gather_cache();

	int target_;
	int old_a_size_{0};
	long my_new_vs_{0};
	long my_new_count_{0};

public:

	explicit Resizer(int target) : target_(target) {}

	~Resizer() {

		for (auto& t : vtasks_) {

			if (t.gathered.ptr) {

				g_buffer_pool.release(t.gathered.ptr, t.gathered.bytes);

			}

		}

	}

	void prepare_phase();
	void commit_phase();

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

		for (size_t ri = g.loop->plan_idx + 1; ri < g.loop->plan_ranges.size(); ri++) {

			push(g.loop->plan_ranges[ri].first, g.loop->plan_ranges[ri].second);

		}

	}

	int my_count = (int)local.size();

	std::vector<int> flat_counts(g.comm.u_size);

	MPI_Allgather(&my_count, 1, MPI_INT, flat_counts.data(), 1, MPI_INT, g.comm.universe);

	std::vector<int> flat_displs = make_displs(flat_counts);

	int total = flat_displs.back() + flat_counts.back();

	if (total == 0) {

		remaining_.clear();
		remaining_offsets_.assign(1, 0);
		total_rem_ = 0;
		rem_per_rank_.assign(g.comm.u_size, 0);
		g.sync.stop = true;

		return;

	}

	std::vector<long> flat(total > 0 ? total : 1);

	MPI_Allgatherv(local.empty() ? nullptr : local.data(), my_count, MPI_LONG, flat.data(), flat_counts.data(), flat_displs.data(), MPI_LONG, g.comm.universe);

	remaining_.clear();
	remaining_.reserve(total / 2 + 1);
	remaining_offsets_.clear();
	remaining_offsets_.reserve(total / 2 + 2);
	remaining_offsets_.push_back(0);

	total_rem_ = 0;
	rem_per_rank_.assign(g.comm.u_size, 0);

	for (int k = 0; k < g.comm.u_size; k++) {

		int disp = flat_displs[k];
		int nranges = flat_counts[k] / 2;

		for (int p = 0; p < nranges; p++) {

			long s = flat[disp + p * 2];
			long e = flat[disp + p * 2 + 1];
			long len = e - s;

			remaining_.push_back({s, e});
			remaining_offsets_.push_back(total_rem_ + len);
			rem_per_rank_[k] += len;
			total_rem_ += len;

		}

	}

	if (total_rem_ == 0) {

		g.sync.stop = true;

	}

	double my_elapsed = MPI_Wtime() - g.lb.epoch_start_time;
	long my_done = std::max(0L, g.lb.epoch_assigned - rem_per_rank_[g.comm.u_rank]);

	double lbdata[2] = {(double)my_done, my_elapsed};
	std::vector<double> all_lb((size_t)g.comm.u_size * 2);

	MPI_Allgather(lbdata, 2, MPI_DOUBLE, all_lb.data(), 2, MPI_DOUBLE, g.comm.universe);

	double total_tp = 0.0;
	std::vector<double> tp(g.comm.u_size, 0.0);

	for (int k = 0; k < g.comm.u_size; k++) {

		long done = (long)all_lb[(size_t)k * 2];
		double elapsed = all_lb[(size_t)k * 2 + 1];

		if (done > 0 && elapsed > 1e-6) {

			tp[k] = (double)done / elapsed;
			total_tp += tp[k];

		}

	}

	if (total_tp > 0.0) {

		if ((int)g.lb.weights.size() < g.comm.u_size) {

			double fill = g.lb.weights.empty() ? (1.0 / g.comm.u_size) : g.lb.weights.back();
			g.lb.weights.resize(g.comm.u_size, fill);

		}

		for (int k = 0; k < g.comm.u_size; k++) {

			if (tp[k] > 0.0) {

				double norm_tp = tp[k] / total_tp;
				g.lb.weights[k] = g.lb.alpha * norm_tp + (1.0 - g.lb.alpha) * g.lb.weights[k];

			}

		}

		MAL_LOG(MAL_LOG_INFO, "LB: epoch done=%ld elapsed=%.3fs thr=%.1f iters/s weight=%.4f", my_done, my_elapsed, my_done > 0 && my_elapsed > 1e-6 ? (double)my_done / my_elapsed : 0.0, g.comm.u_rank < (int)g.lb.weights.size() ? g.lb.weights[g.comm.u_rank] : 0.0);

	}

}

void Resizer::init_vec_meta(int nvecs, int n) {

	vmeta_.resize(n);

	for (int vi = 0; vi < nvecs; vi++) {

		vmeta_[vi].esz = g.loop->vecs[vi]->elem_size;
		vmeta_[vi].shared_active = (g.loop->vecs[vi]->attach_policy == MAL_ATTACH_SHARED_ACTIVE) ? 1 : 0;

	}

	MPI_Bcast(vmeta_.data(), n * (int)sizeof(VecMeta), MPI_BYTE, 0, g.comm.universe);

}

std::vector<TransferPlanEntry> Resizer::build_transfer_plan(const std::vector<long>& old_vs) const {

	std::vector<TransferPlanEntry> plan;
	plan.reserve((size_t)g.comm.u_size + (size_t)target_);

	if (target_ <= 0) {

		return plan;

	}

	int oi = 0;
	int ni = 0;
	long nv_s = target_cuts_[0];
	long nv_e = target_cuts_[1];

	while (oi < g.comm.u_size && rem_per_rank_[oi] == 0) {

		oi++;

	}

	while (oi < g.comm.u_size && ni < target_) {

		long ov_e = old_vs[(size_t)oi + 1];
		long seg_s = std::max(old_vs[(size_t)oi], nv_s);
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

			ni++;

			if (ni < target_) {

				nv_s = target_cuts_[(size_t)ni];
				nv_e = target_cuts_[(size_t)ni + 1];

			}

		}

	}

	return plan;

}

void Resizer::init_vec_tasks(int n, int nvecs, bool was_active) {

	vtasks_.resize(n);

	if (g.gather_cache.size() < (size_t)n) {

		g.gather_cache.resize((size_t)n);

	}

	for (int vi = 0; vi < n; vi++) {

			auto& t = vtasks_[vi];

			t.v = (vi < nvecs) ? g.loop->vecs[vi] : nullptr;
			t.gathered = g.gather_cache[(size_t)vi];
			g.gather_cache[(size_t)vi] = {};

		if (vmeta_[vi].shared_active) {

			if (t.v) {

				t.v->done_n = 0;

			}

			continue;

		}

		if (!t.v) {

			continue;

		}

		long old_done = t.v->done_n;
		long new_done = old_done;

		if (was_active) {

			new_done = std::clamp(*g.loop->user_iter - t.v->buf_global_start + 1, 0L, t.v->local_n);

		}

		if (new_done > old_done) {

			append_done_segments(*t.v, *g.loop, t.v->plan_origin_n, old_done, new_done);

		}

		t.v->done_n = new_done;
		advance_read_only_cache_after_progress(*t.v, old_done, new_done);

	}

}

void Resizer::reserve_receiver_buffers(int n, bool am_receiver, const std::vector<TransferPlanEntry>& plan, long my_new_count, const std::vector<int>& all_reuse_flags) {

	if (!am_receiver || my_new_count <= 0) {

		return;

	}

	auto receiver_reuses = [&](int rank, int vi) {

		return all_reuse_flags[(size_t)rank * (size_t)n + (size_t)vi] != 0;

	};

	bool has_local_assignment = false;

	for (const auto& tr : plan) {

		if (tr.new_rank != g.comm.u_rank) {

			continue;

		}

		has_local_assignment = true;
		break;

	}

	if (!has_local_assignment) {

		return;

	}

	for (int vi = 0; vi < n; vi++) {

		if (vmeta_[vi].shared_active || receiver_reuses(g.comm.u_rank, vi)) {

			continue;

		}

			size_t bytes = (size_t)my_new_count * vmeta_[vi].esz;
			void* gp = vtasks_[vi].gathered.ptr;
			size_t gc = vtasks_[vi].gathered.bytes;

			pool_reserve(gp, gc, bytes, /*preserve_data=*/false);
			vtasks_[vi].gathered = {gp, gc};

	}

}

void Resizer::exchange_vec_data(int n, bool was_active, const std::vector<long>& old_vs, const std::vector<TransferPlanEntry>& plan, const std::vector<int>& all_reuse_flags) {

	auto receiver_reuses = [&](int rank, int vi) {

		return all_reuse_flags[(size_t)rank * (size_t)n + (size_t)vi] != 0;

	};

	std::vector<MPI_Request> reqs;
	reqs.reserve(plan.size() * (size_t)n * 2);

	for (const auto& tr : plan) {

		for (int vi = 0; vi < n; vi++) {

			if (vmeta_[vi].shared_active || receiver_reuses(tr.new_rank, vi)) {

				continue;

			}

			auto& t = vtasks_[vi];
			const size_t esz = vmeta_[vi].esz;
			long bytes64 = tr.v_count * (long)esz;

			if (MAL_UNLIKELY(bytes64 > INT_MAX)) {

				MAL_LOG_L(MAL_LOG_ERROR, "RESIZE", "Transfer size overflow (%ld bytes) in redistribute_vecs", bytes64);
				MPI_Abort(g.comm.universe, 1);

			}

			int byte_count = (int)bytes64;

			if (byte_count == 0) {

				continue;

			}

			const char* send_base = (t.v && was_active) ? static_cast<char*>(t.v->buf) + t.v->done_n * esz : nullptr;

				if (tr.old_rank == tr.new_rank) {

					if (tr.old_rank == g.comm.u_rank && send_base && t.gathered.ptr) {

						long src_off = (tr.v_start - old_vs[(size_t)g.comm.u_rank]) * (long)esz;
						long dst_off = (tr.v_start - my_new_vs_) * (long)esz;

						std::memmove(static_cast<char*>(t.gathered.ptr) + dst_off, send_base + src_off, byte_count);

				}

				continue;

			}

			if (tr.old_rank == g.comm.u_rank && send_base) {

				MPI_Request req;

				MPI_Isend(send_base + (tr.v_start - old_vs[(size_t)tr.old_rank]) * (long)esz, byte_count, MPI_BYTE, tr.new_rank, vi, g.comm.universe, &req);
				reqs.push_back(req);

			}

				if (tr.new_rank == g.comm.u_rank && t.gathered.ptr) {

					char* dst = static_cast<char*>(t.gathered.ptr) + (tr.v_start - my_new_vs_) * (long)esz;
					MPI_Request req;

				MPI_Irecv(dst, byte_count, MPI_BYTE, tr.old_rank, vi, g.comm.universe, &req);
				reqs.push_back(req);

			}

		}

	}

	if (!reqs.empty()) {

		MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

	}

}

void Resizer::redistribute_vecs() {

	int nvecs = g.loop ? (int)g.loop->vecs.size() : 0;
	int n = nvecs;

	MPI_Bcast(&n, 1, MPI_INT, 0, g.comm.universe);

	if (n == 0) {

		return;

	}

	init_vec_meta(nvecs, n);

	std::vector<long> old_vs(g.comm.u_size + 1, 0);
	std::inclusive_scan(rem_per_rank_.begin(), rem_per_rank_.end(), old_vs.begin() + 1);
	target_cuts_.clear();

	if (target_ > 0) {

		target_cuts_ = build_partition_cuts(total_rem_, target_);

	}

	const auto plan = build_transfer_plan(old_vs);

	bool was_active = (g.comm.active != MPI_COMM_NULL);
	init_vec_tasks(n, nvecs, was_active);

	my_new_vs_ = 0;
	my_new_count_ = 0;
	bool am_receiver = (g.comm.u_rank < target_);

	if (am_receiver) {

		long my_nv_e = target_cuts_[(size_t)g.comm.u_rank + 1];
		my_new_vs_ = target_cuts_[(size_t)g.comm.u_rank];
		my_new_count_ = my_nv_e - my_new_vs_;

	}

	scratch_assigned_.clear();

	if (am_receiver && my_new_count_ > 0) {

		scratch_assigned_ = slice_remaining(remaining_, remaining_offsets_, my_new_vs_, my_new_vs_ + my_new_count_);

	}

	scratch_reuse_flags_.assign((size_t)n, 0);

	if (am_receiver && my_new_count_ > 0) {

		for (int vi = 0; vi < n; vi++) {

			if (vmeta_[vi].shared_active || !vtasks_[vi].v) {

				continue;

			}

			if (vec_can_reuse_assigned_ranges(*vtasks_[vi].v, scratch_assigned_)) {

				scratch_reuse_flags_[(size_t)vi] = 1;

			}

		}

	}

	scratch_all_reuse_flags_.assign((size_t)g.comm.u_size * (size_t)n, 0);

	MPI_Allgather(scratch_reuse_flags_.data(), n, MPI_INT, scratch_all_reuse_flags_.data(), n, MPI_INT, g.comm.universe);

	reserve_receiver_buffers(n, am_receiver, plan, my_new_count_, scratch_all_reuse_flags_);
	exchange_vec_data(n, was_active, old_vs, plan, scratch_all_reuse_flags_);

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
		[naccs](int k) -> MalAcc* {

			return k < naccs ? g.loop->accs[k] : nullptr;

		},
		[this, naccs](int k, const char* r, int esz) {

			new_epoch_bufs_[k].assign(r, r + esz);

			if (k >= naccs) {

				return;

			}

			MalAcc* a = g.loop->accs[k];

			a->epoch_buf.assign(r, r + esz);
			a->fn_reset(a->ptr);

		}
	);

}

void Resizer::apply_active() {

	const std::vector<long>& active_cuts = (g.comm.a_size == target_ && target_cuts_.size() == (size_t)g.comm.a_size + 1) ? target_cuts_ : (target_cuts_ = build_partition_cuts(total_rem_, g.comm.a_size));
	long vstart = active_cuts[(size_t)g.comm.a_rank];
	long vend = active_cuts[(size_t)g.comm.a_rank + 1];
	scratch_assigned_ = slice_remaining(remaining_, remaining_offsets_, vstart, vend);
	auto& assigned = scratch_assigned_;

	long new_asgn = vend - vstart;

	const bool waiting_for_activation = g.loop && g.loop->phase.load(std::memory_order_relaxed) == MAL_LOOP_WAITING_ACTIVATION;
	bool publish_pending_after_broadcast = false;
	std::vector<std::pair<long,long>> deferred_pending_ranges;

	MAL_LOG_L(MAL_LOG_INFO, "RESIZE", "a_rank=%d assigned %zu range(s) (%ld iters, weight=%.4f)", g.comm.a_rank, assigned.size(), new_asgn, g.comm.a_rank < (int)g.lb.weights.size() ? g.lb.weights[g.comm.a_rank] : 1.0 / g.comm.a_size);

	g.lb.epoch_assigned = new_asgn;
	g.lb.epoch_start_time = MPI_Wtime();

	if (g.loop && !waiting_for_activation) {

		for (size_t ti = 0; ti < vtasks_.size(); ti++) {

			auto& t = vtasks_[ti];

			if (!t.v) {

				continue;

			}

			if (t.v->attach_policy == MAL_ATTACH_SHARED_ACTIVE) {

				configure_shared_active_vec(*t.v, (size_t)std::max(1L, t.v->total_N) * vmeta_[ti].esz);

				continue;

			}

			long new_local = t.v->done_n + new_asgn;
			bool reused_local = false;

				if (new_asgn > 0 && !assigned.empty() && !t.gathered.ptr) {

				reused_local = vec_reuse_local_copy(*t.v, assigned, t.v->done_n);

			}

			size_t buf_need = (size_t)std::max(1L, new_local) * vmeta_[ti].esz;
			pool_reserve(t.v->buf, t.v->buf_bytes, buf_need);

				if (new_asgn > 0 && !reused_local && t.gathered.ptr && !assigned.empty()) {

				long buf_off = t.v->done_n;
				long gathered_off = 0;

				for (auto [g_start, g_end] : assigned) {

					long len = g_end - g_start;

					if (len > 0 && gathered_off + len <= my_new_count_) {

							std::memcpy(static_cast<char*>(t.v->buf) + buf_off * vmeta_[ti].esz, static_cast<char*>(t.gathered.ptr) + gathered_off * vmeta_[ti].esz, len * vmeta_[ti].esz);

					}

					buf_off += len;
					gathered_off += len;

				}

			}

			long new_buf_global_start = assigned.empty() ? t.v->buf_global_start : (assigned[0].first - t.v->done_n);
			set_partitioned_layout(*t.v, new_local, t.v->done_n, new_buf_global_start);

			if (!set_read_only_cache_from_ranges(*t.v, assigned, t.v->done_n) &&
				t.v->access_mode != MAL_ACCESS_READ_ONLY) {

				t.v->cache_valid = false;

			}

			t.v->sync_user_ptr();

		}

		if (!assigned.empty()) {

			install_loop_plan(*g.loop, assigned);
			set_limit(*g.loop, g.loop->end);
			set_iter (*g.loop, g.loop->start);

		} else {

			freeze_loop_at_current(*g.loop);

		}

	} else {

		auto pa = std::make_unique<PendingActivation>();

		deferred_pending_ranges = std::move(assigned);
		pa->ranges.clear();
		pa->vec_slices.resize(vtasks_.size());

		for (int vi = 0; vi < (int)vtasks_.size(); vi++) {

			auto& t = vtasks_[vi];

				if (new_asgn > 0 && t.gathered.ptr && vmeta_[vi].esz > 0) {

					pa->vec_slices[(size_t)vi] = t.gathered;
					t.gathered = {};

			}

		}

		pa->acc_epoch_bufs = std::move(new_epoch_bufs_);
		g.pending = std::move(pa);
		publish_pending_after_broadcast = true;

	}

	broadcast_shared_vecs();
	broadcast_shared_mats();

	if (publish_pending_after_broadcast && g.pending) {

		g.pending->ranges = std::move(deferred_pending_ranges);
		g.sync.notify();

	}

}

void Resizer::broadcast_shared_mats() {

	bool any_new = (target_ > old_a_size_);

	if (!any_new || g.comm.active == MPI_COMM_NULL) {

		return;

	}

	int n_shared = (int)g.shared.size();

	MPI_Bcast(&n_shared, 1, MPI_INT, 0, g.comm.active);

	if (n_shared == 0) {

		return;

	}

	bool is_new = (g.comm.u_rank >= old_a_size_ && g.comm.u_rank < target_);
	PendingActivation* pa = is_new ? &ensure_pending_activation() : nullptr;

	std::vector<size_t> tots(n_shared, 0);

	if (!is_new) {

		for (int si = 0; si < n_shared; si++) {

			tots[si] = get_shared_mat_or_abort(si)->total_bytes;

		}

	}

	mpi_bcast_bytes(tots.data(), (size_t)n_shared * sizeof(size_t), 0, g.comm.active);

	if (pa) pa->shared_mats.reserve(n_shared);

	for (int si = 0; si < n_shared; si++) {

		size_t tot = tots[si];
		void* buf = nullptr;

		if (is_new) {

			buf = g_buffer_pool.acquire(tot > 0 ? tot : 1);

		} else {

			SharedMat* sm = get_shared_mat_or_abort(si);

			if (!sm->buf || sm->total_bytes != tot) {

				if (!sm->user_owned && sm->buf) {

					g_buffer_pool.release(sm->buf, sm->total_bytes > 0 ? sm->total_bytes : 1);

				}

				sm->buf = g_buffer_pool.acquire(tot > 0 ? tot : 1);
				sm->user_owned = false;

			}

			sm->total_bytes = tot;
			buf = sm->buf;

			if (sm->user_ptr) {

				*sm->user_ptr = sm->buf;

			}

		}

		mpi_bcast_bytes(buf, tot, 0, g.comm.active);

		if (pa) pa->shared_mats.push_back({buf, tot > 0 ? tot : 1});

	}

}

void Resizer::broadcast_shared_vecs() {

	bool any_new = (target_ > old_a_size_);

	if (!any_new || g.comm.active == MPI_COMM_NULL) {

		return;

	}

	struct SharedVecBroadcast {
		int index;
		int bytes;
	};

	std::vector<SharedVecBroadcast> shared_meta;

	if (g.comm.a_rank == 0) {

		shared_meta.reserve(g.vecs.size());

		for (int i = 0; i < (int)g.vecs.size(); i++) {

			MalVec* v = g.vecs[i].get();

			if (!v || v->attach_policy != MAL_ATTACH_SHARED_ACTIVE) {

				continue;

			}

			size_t b = (size_t)std::max(0L, v->total_N) * v->elem_size;

			if (MAL_UNLIKELY(b > (size_t)INT_MAX)) {

				MAL_LOG_L(MAL_LOG_ERROR, "RESIZE", "Shared-active vector size overflow (%zu bytes)", b);
				MPI_Abort(g.comm.universe, 1);

			}

			shared_meta.push_back({i, (int)b});

		}

	}

	int n = (int)shared_meta.size();
	MPI_Bcast(&n, 1, MPI_INT, 0, g.comm.active);

	if (n == 0) {

		return;

	}

	shared_meta.resize(n);
	MPI_Bcast(shared_meta.data(), n * (int)sizeof(SharedVecBroadcast), MPI_BYTE, 0, g.comm.active);

	for (const auto& meta : shared_meta) {

		int vi = meta.index;
		int nbytes = meta.bytes;

		if (MAL_UNLIKELY(vi < 0 || vi >= (int)g.vecs.size())) {

			MAL_LOG_L(MAL_LOG_ERROR, "RESIZE", "Shared-active vector index out of range: %d", vi);
			MPI_Abort(g.comm.universe, 1);

		}

		MalVec* v = g.vecs[vi].get();

		if (MAL_UNLIKELY(!v)) {

			MAL_LOG_L(MAL_LOG_ERROR, "RESIZE", "Shared-active vector missing at index %d", vi);
			MPI_Abort(g.comm.universe, 1);

		}

		configure_shared_active_vec(*v, (size_t)std::max(1L, v->total_N) * v->elem_size);

		if (nbytes > 0) {

			mpi_bcast_bytes(v->buf, (size_t)nbytes, 0, g.comm.active);

		}

	}

}

void Resizer::apply_inactive() {

	g.comm.a_rank = -1;
	g.comm.a_size = 0;

	g.lb.epoch_assigned = 0;
	g.lb.epoch_start_time = 0.0;

	for (auto& t : vtasks_) {

		if (!t.v) {

			continue;

		}

		if (t.v->attach_policy == MAL_ATTACH_SHARED_ACTIVE) {

			release_shared_active_vec(*t.v);

			continue;

		}

		refresh_inactive_read_only_cache(*t.v);

		if (t.v->access_mode != MAL_ACCESS_READ_ONLY) {

			size_t buf_need = (size_t)std::max(1L, t.v->done_n) * t.v->elem_size;
			pool_reserve(t.v->buf, t.v->buf_bytes, buf_need);
			t.v->local_n = t.v->done_n;

		}

		t.v->sync_user_ptr();

	}

	if (g.loop) {

		freeze_loop_at_current(*g.loop);

	}

	g.sync.compute_ready = true;

}

void Resizer::stash_gather_cache() {

	if (g.gather_cache.size() < vtasks_.size()) {

		g.gather_cache.resize(vtasks_.size());

	}

	for (size_t i = 0; i < vtasks_.size(); i++) {

			auto& t = vtasks_[i];
			auto& c = g.gather_cache[i];

		if (c.ptr) {

			g_buffer_pool.release(c.ptr, c.bytes);

		}

			c = t.gathered;
			t.gathered = {};

	}

}

void Resizer::prepare_phase() {

	if (target_ == g.comm.a_size) {

		return;

	}

	const double t0 = MPI_Wtime();
	MAL_LOG_L(MAL_LOG_INFO, "RESIZE", "Prepare phase start target=%d (active=%d)", target_, g.comm.a_size);

	old_a_size_ = g.comm.a_size;

	collect_ranges();
	redistribute_vecs();
	reduce_accs();

	MAL_LOG_L(MAL_LOG_INFO, "RESIZE", "Prepare phase done target=%d in %.4f s", target_, MPI_Wtime() - t0);

}

void Resizer::commit_phase() {

	if (target_ == g.comm.a_size) {

		return;

	}

	g.sync.wait_for_compute();

	MAL_LOG_L(MAL_LOG_INFO, "RESIZE", "Commit phase start target=%d (current=%d)", target_, g.comm.a_size);

	double t0 = MPI_Wtime();

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

	stash_gather_cache();

	MAL_LOG_L(MAL_LOG_INFO, "RESIZE", "Resize to %d done in %.4f s", target_, MPI_Wtime() - t0);

}

static ResizeDecisionContext make_resize_decision_context() {

	ResizeDecisionContext ctx;

	ctx.universe_size = g.comm.u_size;
	ctx.active_size = g.comm.a_size;
	ctx.compute_epoch = g.sync.compute_epoch.load(std::memory_order_relaxed);

 	return ctx;

}

static ResizeDecision decide_resize_fixed_sequence(const ResizeDecisionContext& ctx) {

	ResizeDecision out;

	if (!g.cfg.enabled.load(std::memory_order_relaxed)) {

		return out;

	}

	size_t seq_idx = g.cfg.seq_idx.load(std::memory_order_relaxed);

	if (seq_idx >= g.cfg.sequence.size()) {

		return out;

	}

	int target = g.cfg.sequence[seq_idx];

	if (target <= 0 || target > ctx.universe_size || target == ctx.active_size) {

		return out;

	}

	out.should_resize = true;
	out.target_active_size = target;

	return out;

}

static ResizeDecision decide_resize_auto(const ResizeDecisionContext& ctx) {

	ResizeDecision out;

	if (!g.cfg.enabled.load(std::memory_order_relaxed)) {

		return out;

	}

	long local_rem = 0;

	if (g.loop && g.comm.active != MPI_COMM_NULL) {

		long cur = *g.loop->user_iter;

		if (cur + 1 < g.loop->end) {

			local_rem += g.loop->end - (cur + 1);

		}

		for (size_t ri = g.loop->plan_idx + 1; ri < g.loop->plan_ranges.size(); ri++) {

			local_rem += g.loop->plan_ranges[ri].second - g.loop->plan_ranges[ri].first;

		}

	}

	double my_elapsed = MPI_Wtime() - g.lb.epoch_start_time;
	long my_done = std::max(0L, g.lb.epoch_assigned - local_rem);
	double my_thr = (my_elapsed > 1e-6 && my_done > 0) ? (double)my_done / my_elapsed : 0.0;

	double send_sum[2] = {(double)local_rem, my_thr};
	double recv_sum[2] = {0.0, 0.0};
	MPI_Allreduce(send_sum, recv_sum, 2, MPI_DOUBLE, MPI_SUM, g.comm.universe);

	const double global_remaining = recv_sum[0];
	const double global_throughput_inst = recv_sum[1];

	const double thr_ewma_alpha = std::clamp(g.cfg.auto_thr_ewma_alpha, 1e-6, 1.0);
	double global_throughput = global_throughput_inst;

	if (global_throughput_inst > 1e-12) {

		if (g.lb.auto_thr_ewma <= 1e-12) {

			g.lb.auto_thr_ewma = global_throughput_inst;

		} else {

			g.lb.auto_thr_ewma = thr_ewma_alpha * global_throughput_inst + (1.0 - thr_ewma_alpha) * g.lb.auto_thr_ewma;

		}

		global_throughput = g.lb.auto_thr_ewma;

	}

	double my_data_bytes = 0.0;

	if (g.loop) {

		for (MalVec* v : g.loop->vecs) {

			if (v && v->attach_policy == MAL_ATTACH_PARTITIONED) {

				my_data_bytes += (double)v->total_N * (double)v->elem_size;

			}

		}

	}

	double send_max[2] = {my_data_bytes, (double)g.comm.a_size};
	double recv_max[2] = {0.0, 0.0};

	MPI_Allreduce(send_max, recv_max, 2, MPI_DOUBLE, MPI_MAX, g.comm.universe);

	const double max_data_bytes = recv_max[0];
	const int N = (int)recv_max[1];

	if (global_throughput < 1e-9 || global_remaining < 1.0 || N <= 0) {

		if (global_remaining < 1.0 && global_throughput > 1e-9) {

			g.sync.stop = true;
			g.sync.notify();

		}

		return out;

	}

	const int U = ctx.universe_size;
	const double bw = (g.lb.auto_bw_est_bps > 1e-12) ? g.lb.auto_bw_est_bps : g.cfg.auto_bandwidth_bps;
	const double epoch_secs = g.cfg.epoch_ms.load() / 1000.0;
	const double T_current = global_remaining / global_throughput;

	const double alpha_sync = std::max(0.0, g.lb.auto_alpha_est_sec);
	const double base_sync = std::max(alpha_sync, epoch_secs * g.cfg.auto_sync_overhead_frac);

	auto sync_cost_for = [&](int n) -> double {
		return base_sync * std::log2(std::max(2.0, (double)n));
	};

	const auto& w = g.lb.weights;

	const bool active_weights_ok = ((int)w.size() >= N);

	double sum_w_active = active_weights_ok ? 0.0 : ((double)N / (double)U);
	double w_min_active = active_weights_ok ? 1.0 : (1.0 / U);

	if (active_weights_ok) {

		for (int k = 0; k < N; k++) {

			sum_w_active += w[k];
			w_min_active = std::min(w_min_active, w[k]);

		}

	}

	if ((double)N > global_remaining && N > 1) {

		const int target = std::max(1, std::min(N - 1, (int)global_remaining));
		const double sc = sync_cost_for(N);

		const double resize_cost = (bw > 0.0 ? max_data_bytes * (double)(N - target) / ((double)N * bw) : 0.0) + sc;

		if (T_current > resize_cost * 2.0) {

			MAL_LOG_L(MAL_LOG_INFO, "AUTO", "Scale down (starvation): rem=%.0f < active=%d → target=%d (resize_cost=%.3fs T=%.3fs)", global_remaining, N, target, resize_cost, T_current);

			out.should_resize = true;
			out.target_active_size = target;
			return out;

		}

		MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "Scale down skipped: rem=%.0f < active=%d but T=%.3fs < 2*cost=%.3fs", global_remaining, N, T_current, resize_cost * 2.0);

	}

	if (N >= U) {

		return out;

	}

	double sum_w_cand = sum_w_active;
	double w_min_cand = active_weights_ok ? w_min_active : (1.0 / U);
	double sum_w_at_best = sum_w_active;
	int candidate = -1;

	for (int c = N + 1; c <= U; c++) {

		const bool cand_w_ok = ((int)w.size() >= c);
		const double w_new = (active_weights_ok && cand_w_ok) ? w[c - 1] : (1.0 / U);

		sum_w_cand += w_new;
		w_min_cand = std::min(w_min_cand, w_new);

		const double sc_c = sync_cost_for(c);

		if (sc_c > 1e-9) {

			const double speedup_c = (sum_w_active > 1e-9) ? sum_w_cand / sum_w_active : (double)c / (double)N;
			const double T_new_c = T_current / speedup_c;
			const double min_t_rank = (sum_w_cand > 1e-9) ? T_new_c * w_min_cand / sum_w_cand : T_new_c / c;

			if (min_t_rank < sc_c) {

				break;

			}

		}

		candidate = c;
		sum_w_at_best = sum_w_cand;

	}

	if (candidate == -1) {

		return out;

	}

	const double speedup = (active_weights_ok && sum_w_active > 1e-9 && sum_w_at_best > 1e-9) ? sum_w_at_best / sum_w_active : (double)candidate / (double)N;
	const double T_new = T_current / speedup;

	const int k_new = candidate - N;
	const double data_moved = (bw > 0.0) ? max_data_bytes * (double)k_new / (double)candidate / bw : 0.0;
	const double transfer_cost = data_moved + sync_cost_for(candidate);
	const double net_gain = T_current - T_new - transfer_cost;

	if (net_gain > 0.0) {

		MAL_LOG_L(MAL_LOG_INFO, "AUTO", "Scale up: rem=%.0f T_curr=%.3fs T_new=%.3fs speedup=%.2fx gain=%.3fs cost=%.3fs → target=%d", global_remaining, T_current, T_new, speedup, net_gain, transfer_cost, candidate);

		out.should_resize = true;
		out.target_active_size = candidate;

	}

	return out;

}

static ResizeDecision decide_resize_hw_counters(const ResizeDecisionContext&) {

	MAL_LOG_L(MAL_LOG_ERROR, "EPOCH", "MAL_RESIZE_POLICY_HW_COUNTERS is not implemented yet");
	std::abort();

}

static ResizeDecision decide_resize_remaining_iters(const ResizeDecisionContext&) {

	MAL_LOG_L(MAL_LOG_ERROR, "EPOCH", "MAL_RESIZE_POLICY_REMAINING_ITERS is not implemented yet");
	std::abort();

}

static ResizeDecision run_local_resize_decision(ResizeDecisionContext& ctx) {

	ctx = make_resize_decision_context();
	ResizeDecision decision;

	switch (g.cfg.resize_policy) {

		case MAL_RESIZE_POLICY_AUTO:
			decision = decide_resize_auto(ctx);
			break;

		case MAL_RESIZE_POLICY_FIXED_SEQUENCE:
			decision = decide_resize_fixed_sequence(ctx);
			break;

		case MAL_RESIZE_POLICY_HW_COUNTERS:
			decision = decide_resize_hw_counters(ctx);
			break;

		case MAL_RESIZE_POLICY_REMAINING_ITERS:
			decision = decide_resize_remaining_iters(ctx);
			break;

		default:
			decision = decide_resize_auto(ctx);

	}

	if (!decision.should_resize) {

		decision.target_active_size = -1;
		return decision;

	}

	if (decision.target_active_size <= 0 || decision.target_active_size > g.comm.u_size) {

		MAL_LOG_L(MAL_LOG_WARN, "EPOCH", "Decision returned invalid target=%d (valid range 1..%d)", decision.target_active_size, g.comm.u_size);
		decision.should_resize = false;
		decision.target_active_size = -1;

	}

	return decision;

}

struct ResizeConsensus {

	bool unanimous{false};
	bool should_resize{false};
	int target{-1};
	int active_size{-1};
	unsigned long long decision_epoch{0};

};

static ResizeConsensus unanimous_resize_decision() {

	ResizeConsensus out;
	ResizeDecisionContext local_ctx;
	ResizeDecision local_decision = run_local_resize_decision(local_ctx);

	const long long sr = local_decision.should_resize ? 1LL : 0LL;
	const long long tgt = local_decision.should_resize ? (long long)local_decision.target_active_size : (long long)INT_MAX;
	const long long asiz = (long long)g.comm.a_size;

	long long send[6] = { sr, tgt, asiz, -sr, -tgt, -asiz };

	long long recv[6];
	MPI_Allreduce(send, recv, 6, MPI_LONG_LONG, MPI_MIN, g.comm.universe);

	const long long min_sr = recv[0], max_sr = -recv[3];
	const long long min_tgt = recv[1], max_tgt = -recv[4];
	const long long max_asiz = -recv[5];

	if (min_sr != max_sr) {

		MAL_LOG_L(MAL_LOG_WARN, "EPOCH", "Resize decision is not unanimous (yes_min=%lld yes_max=%lld)", min_sr, max_sr);

		return out;

	}

	out.unanimous = true;
	out.should_resize = (min_sr != 0);
	out.decision_epoch = local_ctx.compute_epoch;
	out.active_size = (int)max_asiz;

	if (!out.should_resize) return out;

	if (min_tgt != max_tgt) {

		MAL_LOG_L(MAL_LOG_WARN, "EPOCH", "Resize target is not unanimous (min=%lld max=%lld)", min_tgt, max_tgt);

		out.unanimous = false;
		out.should_resize = false;
		out.target = -1;

		return out;

	}

	out.target = (int)min_tgt;

	return out;

}

static void advance_default_sequence_after_commit() {

	if (!g.cfg.enabled.load(std::memory_order_relaxed)) {

		return;

	}

	size_t seq_idx = g.cfg.seq_idx.load(std::memory_order_relaxed);

	if (seq_idx < g.cfg.sequence.size()) {

		seq_idx++;
		g.cfg.seq_idx.store(seq_idx, std::memory_order_relaxed);

	}

	if (seq_idx >= g.cfg.sequence.size()) {

		g.cfg.enabled.store(false, std::memory_order_relaxed);

	}

}

static bool prepare_resize_if_needed() {

	{

		std::lock_guard lk(g.resize_mu);

		if (g.prepared_resize.ready()) {

			return false;

		}

	}

	ResizeConsensus consensus = unanimous_resize_decision();

	const bool seq_policy = (g.cfg.resize_policy == MAL_RESIZE_POLICY_FIXED_SEQUENCE);

	if (seq_policy && !consensus.unanimous) {

		advance_default_sequence_after_commit();
		MAL_LOG_L(MAL_LOG_WARN, "EPOCH", "Sequence divergence detected (non-unanimous decision); advancing sequence index to seek next common point");

		return false;

	}

	if (seq_policy && consensus.unanimous && !consensus.should_resize) {

		size_t seq_idx = g.cfg.seq_idx.load(std::memory_order_relaxed);
		bool skipped = g.cfg.enabled.load(std::memory_order_relaxed) &&
			seq_idx < g.cfg.sequence.size() &&
			g.cfg.sequence[seq_idx] == consensus.active_size;

		if (skipped) {

			advance_default_sequence_after_commit();

			MAL_LOG_L(MAL_LOG_INFO, "EPOCH", "Skipping no-op resize target=%d", consensus.active_size);

		}

		return false;

	}

	if (!consensus.unanimous || !consensus.should_resize || consensus.target == g.comm.a_size) {

		return false;

	}

	auto prepared = std::make_unique<Resizer>(consensus.target);
	prepared->prepare_phase();

	{

		std::lock_guard lk(g.resize_mu);

		if (g.prepared_resize.ready()) {

			return false;

		}

		g.prepared_resize.target = consensus.target;
		g.prepared_resize.decision_epoch = consensus.decision_epoch;

		g.prepared_resize.work = std::move(prepared);

	}

	MAL_LOG_L(MAL_LOG_INFO, "EPOCH", "Prepared resize candidate target=%d (epoch=%llu)", consensus.target, consensus.decision_epoch);

	return true;

}

static void clear_prepared_resize() {

	std::lock_guard lk(g.resize_mu);
	g.prepared_resize.reset();

}

static bool commit_prepared_resize_if_ready() {

	int prep_target = -1;
	unsigned long long prep_epoch = 0;
	std::unique_ptr<Resizer> prepared_work;

	{

		std::lock_guard lk(g.resize_mu);

		if (!g.prepared_resize.ready()) {

			return false;

		}

		prep_target = g.prepared_resize.target;
		prep_epoch = g.prepared_resize.decision_epoch;
		prepared_work = std::move(g.prepared_resize.work);
		g.prepared_resize.target = -1;
		g.prepared_resize.decision_epoch = 0;

	}

	unsigned long long epoch_now = g.sync.compute_epoch.load(std::memory_order_relaxed);
	int local_changed = (epoch_now != prep_epoch) ? 1 : 0;
	int any_changed = 0;

	MPI_Allreduce(&local_changed, &any_changed, 1, MPI_INT, MPI_MAX, g.comm.universe);

	if (any_changed != 0) {

		const int mode = g.cfg.epoch_change_mode.load(std::memory_order_relaxed);

		if (mode == MAL_EPOCH_CHANGE_USE_LAST_DECISION) {

			MAL_LOG_L(MAL_LOG_INFO, "EPOCH", "Epoch changed; reusing prepared decision target=%d with existing data (mode=1)", prep_target);

		} else {

			MAL_LOG_L(MAL_LOG_INFO, "EPOCH", "Epoch changed; discarding prepared data and recalculating (mode=0)");

			prepared_work.reset();

			ResizeConsensus refreshed = unanimous_resize_decision();

			if (!refreshed.unanimous || !refreshed.should_resize || refreshed.target == g.comm.a_size) {

				MAL_LOG_L(MAL_LOG_INFO, "EPOCH", "Recalculated decision: no valid resize needed (old_target=%d)", prep_target);
				return false;

			}

			prepared_work = std::make_unique<Resizer>(refreshed.target);
			prepared_work->prepare_phase();

			prep_target = refreshed.target;
			prep_epoch = refreshed.decision_epoch;

		}

	}

	double my_data_bytes = 0.0;

	if (g.loop) {

		for (MalVec* v : g.loop->vecs) {

			if (v && v->attach_policy == MAL_ATTACH_PARTITIONED) {

				my_data_bytes += (double)v->total_N * (double)v->elem_size;

			}

		}

	}

	double max_data_bytes = 0.0;
	MPI_Allreduce(&my_data_bytes, &max_data_bytes, 1, MPI_DOUBLE, MPI_MAX, g.comm.universe);

	const int old_n = std::max(1, g.comm.a_size);
	const int new_n = std::max(1, prep_target);
	const int denom_n = std::max(old_n, new_n);
	const double model_moved_bytes = max_data_bytes * (double)std::abs(new_n - old_n) / (double)denom_n;
	const double model_logp = std::log2(std::max(2.0, (double)new_n));

	g.sync.resize_pending = true;
	g.sync.notify();

	MAL_LOG_L(MAL_LOG_INFO, "EPOCH", "Committing resize target=%d", prep_target);

	const double commit_t0 = MPI_Wtime();
	prepared_work->commit_phase();
	const double commit_elapsed_local = MPI_Wtime() - commit_t0;

	double commit_elapsed = 0.0;
	MPI_Allreduce(&commit_elapsed_local, &commit_elapsed, 1, MPI_DOUBLE, MPI_MAX, g.comm.universe);

	if (model_logp > 1e-9) {

		const double cal_alpha = std::clamp(g.cfg.auto_calibration_alpha, 1e-6, 1.0);
		const double bw_ref = (g.lb.auto_bw_est_bps > 1e-12) ? g.lb.auto_bw_est_bps : g.cfg.auto_bandwidth_bps;
		const double data_term = (bw_ref > 1e-12) ? (model_moved_bytes / bw_ref) : 0.0;

		double alpha_sample = (commit_elapsed - data_term) / model_logp;
		if (!std::isfinite(alpha_sample) || alpha_sample < 0.0) {

			alpha_sample = 0.0;

		}

		g.lb.auto_alpha_est_sec = cal_alpha * alpha_sample + (1.0 - cal_alpha) * g.lb.auto_alpha_est_sec;

		if (model_moved_bytes > 1e-6 && commit_elapsed > 1e-9) {

			double beta_sample = (commit_elapsed - g.lb.auto_alpha_est_sec * model_logp) / model_moved_bytes;

			if (std::isfinite(beta_sample) && beta_sample > 0.0) {

				const double bw_sample = 1.0 / beta_sample;
				g.lb.auto_bw_est_bps = cal_alpha * bw_sample + (1.0 - cal_alpha) * g.lb.auto_bw_est_bps;

			}

		}

		MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "Calibrated comm model: alpha=%.6fs bw=%.3e B/s (elapsed=%.4fs moved=%.3eB logp=%.3f)", g.lb.auto_alpha_est_sec, g.lb.auto_bw_est_bps, commit_elapsed, model_moved_bytes, model_logp);

	}

	g.sync.resize_pending = false;
	clear_prepared_resize();

	if (g.cfg.resize_policy == MAL_RESIZE_POLICY_FIXED_SEQUENCE) {

		advance_default_sequence_after_commit();

	}

	MAL_LOG_L(MAL_LOG_INFO, "EPOCH", "Commit complete (active=%d)", g.comm.a_size);

	g.sync.notify();

	return true;

}

static void progress_thread() {

	#ifdef __APPLE__

		if (g.cfg.affinity_enabled) {

			#if defined(__arm64__) || defined(__aarch64__)

				pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0);
				MAL_LOG_L(MAL_LOG_INFO, "AFFINITY", "worker: QoS set to E-Core");

			#elif defined(__x86_64__) || defined(__i386__)

				mach_port_t self = mach_thread_self();
				thread_affinity_policy_data_t policy = { 1 };
				thread_policy_set(self, THREAD_AFFINITY_POLICY, (thread_policy_t)&policy, THREAD_AFFINITY_POLICY_COUNT);
				mach_port_deallocate(mach_task_self(), self);
				MAL_LOG_L(MAL_LOG_INFO, "AFFINITY", "worker: affinity hint set to E-Core");

			#endif

		}

	#endif

	while (!g.sync.stop) {

		for (;;) {

			std::deque<std::function<void()>> batch;

			{

				std::lock_guard lk(g.attach_mu);

				if (g.attach_tasks.empty()) {

					g.sync.attach_pending = false;
					g.sync.notify();
					break;

				}

				batch.swap(g.attach_tasks);

			}

			for (auto& fn : batch) {

				if (fn) fn();

			}

		}

		const int wait_ms = g.cfg.epoch_ms.load(std::memory_order_relaxed);

		{

			std::unique_lock lk(g.sync.mu);

			g.sync.cv.wait_for(lk, std::chrono::milliseconds(wait_ms > 0 ? wait_ms : MAL_EPOCH_INTERVAL_MS), [] { return g.sync.stop.load() || g.sync.attach_pending.load(); });

		}

		if (g.sync.stop) {

			break;

		}

		if (g.sync.attach_pending.load()) {

			continue;

		}

		prepare_resize_if_needed();
		commit_prepared_resize_if_ready();

		g.sync.notify();

	}

	g.sync.notify();

}

#endif

