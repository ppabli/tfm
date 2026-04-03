#include "malleable_resizer.cpp"

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

	if (!b) {

		clear_prepared_resize();

	}

	g.sync.notify();

}

void mal_set_attach_exec_mode(MalAttachExecMode mode) {

	if (mode != MAL_ATTACH_SYNC && mode != MAL_ATTACH_ASYNC) {

		MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring invalid attach execution mode=%d", (int)mode);

		return;

	}

	g.cfg.attach_mode.store(mode);
	g.sync.notify();

}

MalAttachExecMode mal_get_attach_exec_mode() {

	return g.cfg.attach_mode.load();

}

static bool apply_resize_sequence(const std::vector<int>& seq, const char* source) {

	g.cfg.sequence.clear();
	g.cfg.sequence.reserve(seq.size());

	for (size_t i = 0; i < seq.size(); i++) {

		const int target = seq[i];

		if (target <= 0) {

			MAL_LOG_L(MAL_LOG_ERROR, "CONFIG", "%s: invalid resize target seq[%zu]=%d (must be > 0)", source ? source : "CONFIG", i, target);
			return false;

		}

		g.cfg.sequence.push_back(target);

	}

	if (g.cfg.sequence.empty()) {

		MAL_LOG_L(MAL_LOG_ERROR, "CONFIG", "%s: resize sequence is empty", source ? source : "CONFIG");
		return false;

	}

	g.cfg.seq_idx.store(0, std::memory_order_relaxed);
	return true;

}

static bool parse_resize_sequence(const char* text, std::vector<int>& seq_out, bool& found_invalid) {

	seq_out.clear();
	found_invalid = false;

	if (!text || !*text) {

		return false;

	}

	const char* p = text;

	while (*p) {

		char* end = nullptr;
		long n = std::strtol(p, &end, 10);

		if (end == p) {

			found_invalid = true;
			break;

		}

		if (n > 0) {

			seq_out.push_back((int)n);

		} else {

			found_invalid = true;

		}

		p = end;

		while (*p == ',' || *p == ' ') {

			p++;

		}

	}

	return !seq_out.empty();

}

static void load_resize_sequence_or_abort() {

	const char* source_name = nullptr;
	const char* source_value = nullptr;

	if (const char* v = std::getenv("MAL_RESIZE_SEQ")) {

		if (*v) {

			source_name = "MAL_RESIZE_SEQ";
			source_value = v;

		}

	}

	if (!source_value) {

		#ifdef MAL_DEFAULT_RESIZE_SEQ
			source_name = "MAL_DEFAULT_RESIZE_SEQ";
			source_value = MAL_DEFAULT_RESIZE_SEQ;
		#endif

	}

	if (!source_name || !source_value) {

		MAL_LOG_L(MAL_LOG_ERROR, "CONFIG", "Missing resize sequence: set MAL_RESIZE_SEQ or define MAL_DEFAULT_RESIZE_SEQ at build time");
		std::abort();

	}

	std::vector<int> seq;
	bool found_invalid = false;

	if (!parse_resize_sequence(source_value, seq, found_invalid) || found_invalid) {

		MAL_LOG_L(MAL_LOG_ERROR, "CONFIG", "%s is invalid; expected comma-separated positive integers", source_name);
		std::abort();

	}

	if (!apply_resize_sequence(seq, source_name)) {

		std::abort();

	}

	MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "%s loaded (%zu resize points)", source_name, seq.size());

}

static void validate_resize_sequence_against_universe_or_abort() {

	for (size_t i = 0; i < g.cfg.sequence.size(); i++) {

		const int target = g.cfg.sequence[i];

		if (target > g.comm.u_size) {

			MAL_LOG_L(MAL_LOG_ERROR, "CONFIG", "Resize target seq[%zu]=%d exceeds universe size=%d", i, target, g.comm.u_size);
			std::abort();

		}

	}

}

static void load_env_config() {

	if (g.cfg.resize_policy == MAL_RESIZE_POLICY_FIXED_SEQUENCE) {

		load_resize_sequence_or_abort();

	}

	if (const char* v = std::getenv("MAL_EPOCH_INTERVAL_MS")) {

		long ms = std::strtol(v, nullptr, 10);

		if (ms > 0) {

			g.cfg.epoch_ms.store((int)ms);
			MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "MAL_EPOCH_INTERVAL_MS=%ld", ms);

		} else {

			MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring MAL_EPOCH_INTERVAL_MS='%s' (must be > 0)", v);

		}

	}

	if (const char* v = std::getenv("MAL_EPOCH_CHANGE_MODE")) {

		char* end = nullptr;
		long mode = std::strtol(v, &end, 10);

		if (end == v || (mode != MAL_EPOCH_CHANGE_RECALCULATE && mode != MAL_EPOCH_CHANGE_USE_LAST_DECISION)) {

			MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring MAL_EPOCH_CHANGE_MODE='%s' (valid: 0=recalculate, 1=use last decision)", v);

		} else {

			g.cfg.epoch_change_mode.store((int)mode);
			MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "MAL_EPOCH_CHANGE_MODE=%ld", mode);

		}

	}

	if (const char* v = std::getenv("MAL_RESIZE_ENABLED")) {

		bool val = std::strtol(v, nullptr, 10) != 0;
		g.cfg.enabled.store(val);
		MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "MAL_RESIZE_ENABLED=%d", (int)val);

	}

	if (const char* v = std::getenv("MAL_AFFINITY")) {

		char* end = nullptr;
		long val = std::strtol(v, &end, 10);

		if (end == v) {

			MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring MAL_AFFINITY='%s' (invalid), using compile-time default (%d)", v, MAL_AFFINITY_ENABLED);

		} else {

			g.cfg.affinity_enabled = (val != 0);
			MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "MAL_AFFINITY=%ld to affinity %s", val, g.cfg.affinity_enabled ? "enabled" : "disabled");

		}

	}

	if (const char* v = std::getenv("MAL_MAIN_CORE")) {

		char* end = nullptr;
		long val = std::strtol(v, &end, 10);

		if (end == v || val < 0) {

			MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring MAL_MAIN_CORE='%s' (must be >= 0), using compile-time default (%d)", v, MAL_MAIN_CORE_DEFAULT);

		} else {

			g.cfg.main_core = (int)val;
			MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "MAL_MAIN_CORE=%ld", val);

		}

	}

	if (const char* v = std::getenv("MAL_WORKER_CORE")) {

		char* end = nullptr;
		long val = std::strtol(v, &end, 10);

		if (end == v || val < 0) {

			MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring MAL_WORKER_CORE='%s' (must be >= 0), using compile-time default (%d)", v, MAL_WORKER_CORE_DEFAULT);

		} else {

			g.cfg.worker_core = (int)val;
			MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "MAL_WORKER_CORE=%ld", val);

		}

	}

	if (const char* v = std::getenv("MAL_AUTO_BANDWIDTH_BPS")) {

		char* end = nullptr;
		double val = std::strtod(v, &end);

		if (end == v || val <= 0.0) {

			MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring MAL_AUTO_BANDWIDTH_BPS='%s' (must be > 0)", v);

		} else {

			g.cfg.auto_bandwidth_bps = val;
			MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "MAL_AUTO_BANDWIDTH_BPS=%.3e", val);

		}

	}

	if (const char* v = std::getenv("MAL_AUTO_SYNC_OVERHEAD_FRAC")) {

		char* end = nullptr;
		double val = std::strtod(v, &end);

		if (end == v || val < 0.0 || val >= 1.0) {

			MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring MAL_AUTO_SYNC_OVERHEAD_FRAC='%s' (must be in [0, 1))", v);

		} else {

			g.cfg.auto_sync_overhead_frac = val;
			MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "MAL_AUTO_SYNC_OVERHEAD_FRAC=%.4f", val);

		}

	}

	if (const char* v = std::getenv("MAL_AUTO_THR_EWMA_ALPHA")) {

		char* end = nullptr;
		double val = std::strtod(v, &end);

		if (end == v || val <= 0.0 || val > 1.0) {

			MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring MAL_AUTO_THR_EWMA_ALPHA='%s' (must be in (0, 1])", v);

		} else {

			g.cfg.auto_thr_ewma_alpha = val;
			MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "MAL_AUTO_THR_EWMA_ALPHA=%.4f", val);

		}

	}

	if (const char* v = std::getenv("MAL_AUTO_CALIBRATION_ALPHA")) {

		char* end = nullptr;
		double val = std::strtod(v, &end);

		if (end == v || val <= 0.0 || val > 1.0) {

			MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring MAL_AUTO_CALIBRATION_ALPHA='%s' (must be in (0, 1])", v);

		} else {

			g.cfg.auto_calibration_alpha = val;
			MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "MAL_AUTO_CALIBRATION_ALPHA=%.4f", val);

		}

	}

}

void mal_init(MalResizePolicy policy) {

	g.cfg.resize_policy = policy;

	load_env_config();
	g.lb.auto_bw_est_bps = g.cfg.auto_bandwidth_bps;

	MPI_Session_init(MPI_INFO_NULL, MPI_ERRORS_RETURN, &g.comm.session);
	MPI_Group_from_session_pset(g.comm.session, "mpi://WORLD", &g.comm.world_group);
	MPI_Comm_create_from_group(g.comm.world_group, "malleable.universe", MPI_INFO_NULL, MPI_ERRORS_RETURN, &g.comm.universe);
	MPI_Comm_rank(g.comm.universe, &g.comm.u_rank);
	MPI_Comm_size(g.comm.universe, &g.comm.u_size);

	if (policy == MAL_RESIZE_POLICY_FIXED_SEQUENCE) {

		validate_resize_sequence_against_universe_or_abort();

	}

	#if defined(__linux__) || defined(__APPLE__)

		pin_main_thread_to_pcore();

	#endif

	if (MAL_INITIAL_SIZE <= 0 || MAL_INITIAL_SIZE > g.comm.u_size) {

		if (g.comm.u_rank == 0) {

			MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "MAL_INITIAL_SIZE=%d is invalid for universe size=%d", MAL_INITIAL_SIZE, g.comm.u_size);

		}

	}

	int color = (g.comm.u_rank < MAL_INITIAL_SIZE) ? 0 : MPI_UNDEFINED;
	MPI_Comm_split(g.comm.universe, color, g.comm.u_rank, &g.comm.active);

	g.worker = std::thread(progress_thread);

	#if defined(__linux__) || defined(__APPLE__)

		pin_worker_thread_to_ecore(g.worker);

	#endif

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

		for (int k = 0; k < g.comm.a_size; k++) {

			long ks, ke;
			distribute(v.total_N, g.comm.a_size, k, ks, ke);
			sc[k] = (int)((ke - ks) * (long)v.elem_size);

		}

		sd = make_displs(sc);

	}

	long rc_bytes = v.local_n * (long)v.elem_size;

	if (MAL_UNLIKELY(rc_bytes > INT_MAX)) {

		MAL_LOG_L(MAL_LOG_ERROR, "SCATTER", "Local receive size overflow (%ld bytes) in vec_scatter", rc_bytes);
		MPI_Abort(g.comm.universe, 1);

	}

	int rc = (int)rc_bytes;

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

		my_seg_flat.push_back(s);
		my_seg_flat.push_back(c);

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

	void* recv_raw = nullptr;
	size_t recv_cap = 0;

	if (root) {

		data_counts.resize(usiz);

		for (int k = 0; k < usiz; k++) {

			long total = 0;

			for (int s = 0; s < seg_counts[k] / 2; s++) {

				total += all_segs[seg_displs[k] + s * 2 + 1];

			}

			data_counts[k] = (int)(total * (long)v.elem_size);

		}

		data_displs = make_displs(data_counts);

		size_t total_recv = (size_t)(data_displs.back() + data_counts.back());
		pool_reserve(recv_raw, recv_cap, total_recv > 0 ? total_recv : 1, /*preserve_data=*/false);

	}

	MPI_Gatherv(my_data_bytes > 0 ? v.buf : nullptr, (int)my_data_bytes, MPI_BYTE, recv_raw, root ? data_counts.data() : nullptr, root ? data_displs.data() : nullptr, MPI_BYTE, v.gather_root, g.comm.universe);

	if (root && v.result_buf) {

		char* recv_buf = static_cast<char*>(recv_raw);
		long data_off = 0;

		for (int k = 0; k < usiz; k++) {

			for (int s = 0; s < seg_counts[k] / 2; s++) {

				long gs = all_segs[seg_displs[k] + s * 2];
				long cnt = all_segs[seg_displs[k] + s * 2 + 1];

				std::memcpy(static_cast<char*>(v.result_buf) + gs * (long)v.elem_size, recv_buf + data_off, cnt * (long)v.elem_size);

				data_off += cnt * (long)v.elem_size;

			}

		}

	}

	if (recv_raw) {

		g_buffer_pool.release(recv_raw, recv_cap);

	}

}

void mal_finalize() {

	mal_wait_attach_tasks();

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

		if (v.result_buf && !v.user_ptr) {

			std::free(v.result_buf);
			v.result_buf = nullptr;

		}

		v.free_resources();

	}

	g.vecs.clear();

	int naccs = (int)g.accs.size();

	MPI_Bcast(&naccs, 1, MPI_INT, 0, g.comm.universe);

	batched_allreduce(naccs,
		[](int k) -> MalAcc* {

			return g.accs[k].get();

		},
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

	for (auto& e : g.gather_cache) {

		g_buffer_pool.release(e.ptr, e.bytes);

	}

	g.gather_cache.clear();

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

		load_pending_ranges_into_loop(f);
		f.phase.store(MAL_LOOP_ATTACHING, std::memory_order_relaxed);

	} else if (g.comm.a_size > 0 && !g.sync.stop) {

		weighted_distribute(total_iters, g.comm.a_size, g.comm.a_rank, f.start, f.end);
		install_loop_plan(f, {{f.start, f.end}});

		g.lb.epoch_assigned = f.end - f.start;
		g.lb.epoch_start_time = MPI_Wtime();
		f.phase.store(MAL_LOOP_ATTACHING, std::memory_order_relaxed);

	} else {

		f.start = f.end = 0;
		f.plan_ranges.clear();
		f.plan_local_bases.clear();
		f.phase.store(MAL_LOOP_WAITING_ACTIVATION, std::memory_order_relaxed);

	}

	set_iter(f, f.start);
	limit = f.end;

	g.sync.compute_ready = false;
	g.loop = &f;

	while (f.start == f.end && !g.sync.stop) {

		f.current = f.end;
		f.phase.store(MAL_LOOP_WAITING_ACTIVATION, std::memory_order_relaxed);
		g.sync.compute_wait(has_work_or_stop);

		if (f.start == f.end && g.pending && !g.pending->ranges.empty()) {

			load_pending_ranges_into_loop(f);
			f.phase.store(MAL_LOOP_ATTACHING, std::memory_order_relaxed);

		}

	}

	if (!g.sync.stop.load(std::memory_order_relaxed) &&
		f.start < f.end &&
		!g.sync.attach_pending.load(std::memory_order_relaxed)) {

		f.phase.store(MAL_LOOP_RUNNING, std::memory_order_relaxed);

	}

	return f;

}

static void advance_next_range(MalFor& f) {

	f.plan_idx++;
	auto [a, b] = f.plan_ranges[f.plan_idx];

	f.start = a;

	sync_vec_mapping_for_current_range(f);

	set_limit(f, b);

	prime_range_start(f);

	MAL_LOG_L(MAL_LOG_INFO, "RANGE", "Next range [%ld, %ld) (base=%ld)", a, b, current_range_local_base(f));

}

void mal_check_for(MalFor& f) {

	g.sync.compute_epoch.fetch_add(1, std::memory_order_relaxed);

	if (g.sync.attach_pending) {

		f.phase.store(MAL_LOOP_ATTACHING, std::memory_order_relaxed);

		g.sync.compute_wait([] {

			return !g.sync.attach_pending.load() || g.sync.stop.load();

		});

	}

	if (!g.sync.stop.load(std::memory_order_relaxed) && f.start < f.end) {

		f.phase.store(MAL_LOOP_RUNNING, std::memory_order_relaxed);

	}

	f.current = *f.user_iter;

	if (MAL_UNLIKELY(g.sync.resize_pending)) {

		g.sync.compute_wait([] {

			return !g.sync.resize_pending.load() || g.sync.stop.load();

		});

		if (g.comm.active != MPI_COMM_NULL && f.start == f.end && g.pending && !g.pending->ranges.empty()) {

			load_pending_ranges_into_loop(f);

		}

		if (g.comm.active != MPI_COMM_NULL && f.start < f.end) {

			prime_range_start(f);

			return;

		}

	}

	if (MAL_LIKELY(*f.user_iter + 1 < f.end)) {

		return;

	}

	if (f.plan_idx + 1 < f.plan_ranges.size()) {

		advance_next_range(f);

		return;

	}

	if (g.sync.stop) {

		return;

	}

	f.current = f.end;
	g.sync.compute_wait(has_work_or_stop);

	if (!g.sync.stop) {

		prime_range_start(f);

	}

}

void mal_attach_vec(MalFor& f, void** user_ptr, size_t elem_size, long total_N, int result_rank, MalAttachPolicy policy, MalAttachExecMode exec_mode, MalDataAccessMode access_mode) {

	f.phase.store(MAL_LOOP_ATTACHING, std::memory_order_relaxed);

	auto vp = std::make_unique<MalVec>();
	MalVec* v = vp.get();

	void* orig = user_ptr ? *user_ptr : nullptr;
	long n = f.end - f.start;
	long planned_total = total_range_iters(f.plan_ranges);

	if (planned_total > 0) {

		n = planned_total;

	}

	v->elem_size = elem_size;
	v->local_n = n;
	v->buf_global_start = f.start;
	v->total_N = total_N;
	v->user_ptr = user_ptr;
	v->gather_root = result_rank;
	v->attach_policy = policy;
	v->access_mode = access_mode;
	v->cache_valid = false;

	if (result_rank >= 0 && g.comm.u_rank == result_rank) {

		v->result_buf = orig ? orig : checked_realloc(nullptr, total_N > 0 ? (size_t)total_N * elem_size : 1, "mal_attach_vec.result_buf");

	}

	const bool shared_active = (policy == MAL_ATTACH_SHARED_ACTIVE);
	bool once_all = (policy == MAL_ATTACH_SHARED_ALL);
	const bool async_attach = use_async_attach_mode(exec_mode);

	if (MAL_UNLIKELY(elem_size == 0)) {

		MAL_LOG_L(MAL_LOG_WARN, "ATTACH", "mal_attach_vec called with elem_size=0");

	}

	if (MAL_UNLIKELY(total_N < 0)) {

		MAL_LOG_L(MAL_LOG_WARN, "ATTACH", "mal_attach_vec called with negative total_N=%ld", total_N);

	}

	if ((once_all || shared_active) && result_rank >= 0) {

		MAL_LOG_L(MAL_LOG_WARN, "ATTACH", "Shared vector policy ignores gather result_rank=%d", result_rank);

		if (v->result_buf != nullptr && v->result_buf != orig) {

			std::free(v->result_buf);

		}

		v->result_buf = nullptr;
		v->gather_root = -1;
		once_all = false;
		result_rank = -1;

	}

	if (once_all || shared_active) {

		n = total_N;
		v->local_n = n;
		v->done_n = 0;
		v->buf_global_start = 0;

	}

	v->buf = static_cast<char*>(g_buffer_pool.acquire((v->local_n > 0 ? (size_t)v->local_n : 1) * elem_size));
	v->buf_bytes = (v->local_n > 0 ? (size_t)v->local_n : 1) * elem_size;
	v->plan_origin_n = v->done_n;

	int idx = (int)f.vecs.size();

	if (g.pending) {

		StagedBuffer stash = take_pending_vec_slice(idx);

		if (stash.ptr && n > 0) {

			std::memcpy(v->buf, stash.ptr, (size_t)n * elem_size);
			g_buffer_pool.release(stash.ptr, stash.bytes);

		}

	}

	v->sync_user_ptr();

	f.vecs.push_back(v);
	g.vecs.push_back(std::move(vp));

	if (!g.pending && once_all) {

		size_t total_bytes = (size_t)std::max(0L, total_N) * elem_size;
		run_shared_all_attach_bcast(v->buf, orig, total_bytes, result_rank, exec_mode, "MAL_ATTACH_SHARED_ALL vector has null root pointer; broadcasting zero-initialized data");

	} else if (!g.pending && shared_active && g.comm.active != MPI_COMM_NULL) {

		size_t total_bytes = (size_t)std::max(0L, total_N) * elem_size;
		run_shared_active_attach_bcast(*v, orig, total_bytes, exec_mode);

	} else if (!g.pending && g.comm.active != MPI_COMM_NULL) {

		size_t orig_bytes = (size_t)std::max(0L, total_N) * elem_size;
		run_partitioned_attach_scatter(*v, orig, result_rank, orig_bytes, exec_mode);

	}

	if (g.pending && idx + 1 == (int)g.pending->vec_slices.size()) {

		g.pending->vec_slices.clear();

	}

	if (!g.sync.stop.load(std::memory_order_relaxed) && f.start < f.end && (!async_attach || !g.sync.attach_pending.load(std::memory_order_relaxed))) {

		f.phase.store(MAL_LOOP_RUNNING, std::memory_order_relaxed);

	}

}

void mal_attach_vec(MalForND& f, void** user_ptr, size_t elem_size, long total_N, int result_rank, MalAttachPolicy policy, MalAttachExecMode exec_mode, MalDataAccessMode access_mode) {

	mal_attach_vec(mal_for_nd_base(f), user_ptr, elem_size, total_N, result_rank, policy, exec_mode, access_mode);

}

void detail::acc_register(MalFor& f, detail::AccDesc d, int result_rank) {

	f.phase.store(MAL_LOOP_ATTACHING, std::memory_order_relaxed);

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

	a->epoch_buf = take_pending_acc_epoch_buf(a->esz);

	a->fn_reset(a->ptr);

	f.accs.push_back(a);
	g.accs.push_back(std::move(ap));

	if (!g.sync.stop.load(std::memory_order_relaxed) && f.start < f.end) {

		f.phase.store(MAL_LOOP_RUNNING, std::memory_order_relaxed);

	}

}

void mal_attach_mat(MalFor& f, void** user_ptr, size_t elem_size, long primary_n, long secondary_n, int result_rank, MalAttachPolicy policy, MalAttachExecMode exec_mode, MalDataAccessMode access_mode) {

	f.phase.store(MAL_LOOP_ATTACHING, std::memory_order_relaxed);

	if (MAL_UNLIKELY(elem_size == 0 || primary_n < 0 || secondary_n < 0)) {

		MAL_LOG_L(MAL_LOG_WARN, "ATTACH", "mal_attach_mat called with invalid shape/size elem_size=%zu primary_n=%ld secondary_n=%ld", elem_size, primary_n, secondary_n);

	}

	if (policy == MAL_ATTACH_PARTITIONED) {

		mal_attach_vec(f, user_ptr, elem_size * (size_t)secondary_n, primary_n, result_rank, MAL_ATTACH_PARTITIONED, exec_mode, access_mode);

		return;

	}

	const size_t total_bytes = (size_t)primary_n * (size_t)secondary_n * elem_size;
	const bool shared_all = (policy == MAL_ATTACH_SHARED_ALL);
	const bool async_attach = use_async_attach_mode(exec_mode);

	if (!shared_all && g.comm.active == MPI_COMM_NULL) {

		if (user_ptr) {

			*user_ptr = nullptr;

		}

		auto sp = std::make_unique<SharedMat>();
		sp->buf = nullptr;
		sp->total_bytes = total_bytes;
		sp->user_owned = false;
		sp->user_ptr = user_ptr;

		g.shared.push_back(std::move(sp));

		if (!g.sync.stop.load(std::memory_order_relaxed) && f.start < f.end && (!async_attach || !g.sync.attach_pending.load(std::memory_order_relaxed))) {

			f.phase.store(MAL_LOOP_RUNNING, std::memory_order_relaxed);

		}

		return;

	}

	if (result_rank >= 0) {

		MAL_LOG_L(MAL_LOG_WARN, "ATTACH", "result_rank=%d ignored for shared matrix policies", result_rank);

	}

	void* orig = user_ptr ? *user_ptr : nullptr;
	void* buf;

	if (g.pending && !shared_all) {

		StagedBuffer staged = take_pending_shared_mat();

		if (staged.ptr) {

			buf = staged.ptr;

		} else {

			buf = acquire_or_broadcast_active_shared_mat(orig, total_bytes, exec_mode);

		}

	} else {

		if (shared_all) {

			buf = g_buffer_pool.acquire(total_bytes > 0 ? total_bytes : 1);
			run_shared_all_attach_bcast(buf, orig, total_bytes, -1, exec_mode, "MAL_ATTACH_SHARED_ALL matrix has null root pointer; broadcasting zero-initialized data");

		} else {

			buf = acquire_or_broadcast_active_shared_mat(orig, total_bytes, exec_mode);

		}

	}

	if (user_ptr) {

		*user_ptr = buf;

	}

	auto sp = std::make_unique<SharedMat>();
	sp->buf = buf;
	sp->total_bytes = total_bytes;
	sp->user_owned = shared_all ? (g.comm.u_rank == 0 && orig != nullptr) : (g.comm.a_rank == 0 && orig != nullptr);
	sp->user_ptr = user_ptr;

	g.shared.push_back(std::move(sp));

	if (!g.sync.stop.load(std::memory_order_relaxed) &&
		f.start < f.end &&
		(!async_attach || !g.sync.attach_pending.load(std::memory_order_relaxed))) {

		f.phase.store(MAL_LOOP_RUNNING, std::memory_order_relaxed);

	}

}

void mal_attach_mat(MalForND& f, void** user_ptr, size_t elem_size, long primary_n, long secondary_n, int result_rank, MalAttachPolicy policy, MalAttachExecMode exec_mode, MalDataAccessMode access_mode) {

	mal_attach_mat(mal_for_nd_base(f), user_ptr, elem_size, primary_n, secondary_n, result_rank, policy, exec_mode, access_mode);

}
