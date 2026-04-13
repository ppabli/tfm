#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <vector>
#include <mpi.h>
#include "example_utils.hpp"

#define SPARSE_ROWS 3600
#define SPARSE_COLS 4096
#define SPARSE_MAX_ROW_NNZ 240

struct BlockRange {
	long start;
	long count;
};

BlockRange block_range(long total, int rank, int size) {

	const long base = total / size;
	const long rem = total % size;
	const long count = base + (rank < rem ? 1 : 0);
	const long start = rank * base + std::min<long>(rank, rem);

	return {start, count};

}

void build_gatherv_layout(long total, int world_size, std::vector<int>& counts, std::vector<int>& displs) {

	counts.resize(world_size);
	displs.resize(world_size);

	for (int r = 0; r < world_size; r++) {

		const BlockRange rr = block_range(total, r, world_size);
		counts[r] = static_cast<int>(rr.count);
		displs[r] = static_cast<int>(rr.start);

	}

}

double x_val_for_col(int col) {

	return 1.0 + static_cast<double>(col % 9) * 0.075;

}

int nnz_for_row(long row) {

	const long block = (row / 36) % 5;

	if (block == 0) {

		return 6 + static_cast<int>(row % 7);

	}

	if (block == 1) {

		return 24 + static_cast<int>(row % 17);

	}

	if (block == 2) {

		return 80 + static_cast<int>(row % 29);

	}

	if (block == 3) {

		return 140 + static_cast<int>((row % 6) * 12);

	}

	if ((row % 4) == 0) {

		return 220;

	}

	return 42 + static_cast<int>(row % 23);

}

int clamped_nnz_for_row(long row) {

	return std::min(nnz_for_row(row), SPARSE_MAX_ROW_NNZ);

}

int stride_for_row(long row) {

	int stride = 17 + static_cast<int>(row % 31);
	if ((stride % 2) == 0) {

		stride++;

	}

	return stride;

}

int seed_for_row(long row) {

	return static_cast<int>((row * 59 + (row / 9) * 83 + 7) % SPARSE_COLS);

}

int col_for(long row, int k) {

	return (seed_for_row(row) + k * stride_for_row(row) + (k % 5) * 19) % SPARSE_COLS;

}

double val_for(long row, int k) {

	return 0.4 + static_cast<double>((row + k) % 21) * 0.11;

}

long build_sparse_problem(int* row_nnz, int* col_idx, double* values, double* x) {

	for (int c = 0; c < SPARSE_COLS; c++) {

		x[c] = x_val_for_col(c);

	}

	long total_nnz = 0;
	for (long row = 0; row < SPARSE_ROWS; row++) {

		const int clamped_nnz = clamped_nnz_for_row(row);
		row_nnz[row] = clamped_nnz;
		total_nnz += clamped_nnz;

		const long base = row * SPARSE_MAX_ROW_NNZ;
		for (int k = 0; k < SPARSE_MAX_ROW_NNZ; k++) {

			if (k < clamped_nnz) {

				col_idx[base + k] = col_for(row, k);
				values[base + k] = val_for(row, k);

			} else {

				col_idx[base + k] = 0;
				values[base + k] = 0.0;

			}

		}

	}

	return total_nnz;

}

int main(int argc, char* argv[]) {

	MPI_Init(&argc, &argv);

	int world_rank = 0;
	int world_size = 1;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	int* row_nnz = static_cast<int*>(std::malloc(SPARSE_ROWS * sizeof(int)));
	int* col_idx = static_cast<int*>(std::malloc(SPARSE_ROWS * SPARSE_MAX_ROW_NNZ * sizeof(int)));
	double* values = static_cast<double*>(std::malloc(SPARSE_ROWS * SPARSE_MAX_ROW_NNZ * sizeof(double)));
	double* x = static_cast<double*>(std::malloc(SPARSE_COLS * sizeof(double)));
	double* y = (world_rank == 0) ? static_cast<double*>(std::malloc(SPARSE_ROWS * sizeof(double))) : nullptr;

	long total_nnz = 0;
	if (world_rank == 0) {

		total_nnz = build_sparse_problem(row_nnz, col_idx, values, x);

	}

	MPI_Barrier(MPI_COMM_WORLD);
	const double t0 = MPI_Wtime();

	MPI_Bcast(row_nnz, SPARSE_ROWS, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(col_idx, SPARSE_ROWS * SPARSE_MAX_ROW_NNZ, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(values, SPARSE_ROWS * SPARSE_MAX_ROW_NNZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(x, SPARSE_COLS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&total_nnz, 1, MPI_LONG, 0, MPI_COMM_WORLD);

	const BlockRange range = block_range(SPARSE_ROWS, world_rank, world_size);
	std::vector<double> local_y(static_cast<size_t>(range.count));
	const useconds_t delay_scale_percent = sparse_delay_scale_percent();

	for (long local_i = 0; local_i < range.count; local_i++) {

		const long row = range.start + local_i;
		double acc = 0.0;
		const int nnz = row_nnz[row];
		const long base = row * SPARSE_MAX_ROW_NNZ;

		for (int k = 0; k < nnz; k++) {

			acc += values[base + k] * x[col_idx[base + k]];

		}

		const int delay_ms = 4 + nnz / 2;
		const useconds_t delay_us = static_cast<useconds_t>(delay_ms) * 1000u * delay_scale_percent / 100u;
		local_y[static_cast<size_t>(local_i)] = acc;
		usleep(delay_us);

	}

	std::vector<int> recv_counts;
	std::vector<int> recv_displs;
	if (world_rank == 0) {

		build_gatherv_layout(SPARSE_ROWS, world_size, recv_counts, recv_displs);

	}

	MPI_Gatherv(local_y.data(), static_cast<int>(range.count), MPI_DOUBLE, y, (world_rank == 0) ? recv_counts.data() : nullptr, (world_rank == 0) ? recv_displs.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	const double t1 = MPI_Wtime();

	if (world_rank == 0) {

		int errors = 0;
		double max_abs_err = 0.0;

		for (long r = 0; r < SPARSE_ROWS; r++) {

			double expected = 0.0;
			const int nnz = clamped_nnz_for_row(r);

			for (int k = 0; k < nnz; k++) {

				expected += val_for(r, k) * x_val_for_col(col_for(r, k));

			}

			const double err = std::fabs(y[r] - expected);
			max_abs_err = std::max(max_abs_err, err);

			if (err > 1e-9) {

				errors++;
				if (errors <= 3) {

					std::printf("[CHECK] row=%ld y=%.12f expected=%.12f err=%.3e\n", r, y[r], expected, err);

				}

			}

		}

		std::printf("[RESULT] sparse mat-vec %s (rows=%d cols=%d nnz=%ld errors=%d max_abs_err=%.3e)\n", errors == 0 ? "OK" : "WRONG", SPARSE_ROWS, SPARSE_COLS, total_nnz, errors, max_abs_err);
		std::printf("[TIME] sparse normal mpi np=%d seconds=%.6f\n", world_size, t1 - t0);
		std::free(y);

	}

	std::free(row_nnz);
	std::free(col_idx);
	std::free(values);
	std::free(x);

	MPI_Finalize();
	return EXIT_SUCCESS;

}
