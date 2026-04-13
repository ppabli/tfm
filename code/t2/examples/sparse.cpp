#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <mpi.h>
#include "malleable.hpp"
#include "example_utils.hpp"

#define SPARSE_ROWS 3600
#define SPARSE_COLS 4096
#define SPARSE_MAX_ROW_NNZ 240

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

int main(int /*argc*/, char* /*argv*/[]) {

	mal_init(MAL_RESIZE_POLICY_AUTO);

	int* row_nnz = nullptr;
	int* col_idx = nullptr;
	double* values = nullptr;
	double* x = nullptr;
	double* y = nullptr;

	long total_nnz = 0;

	if (mal_rank() == 0) {

		row_nnz = static_cast<int*>(std::malloc(SPARSE_ROWS * sizeof(int)));
		col_idx = static_cast<int*>(std::malloc(SPARSE_ROWS * SPARSE_MAX_ROW_NNZ * sizeof(int)));
		values = static_cast<double*>(std::malloc(SPARSE_ROWS * SPARSE_MAX_ROW_NNZ * sizeof(double)));
		x = static_cast<double*>(std::malloc(SPARSE_COLS * sizeof(double)));
		y = static_cast<double*>(std::malloc(SPARSE_ROWS * sizeof(double)));

		total_nnz = build_sparse_problem(row_nnz, col_idx, values, x);

		MAL_LOG(MAL_LOG_INFO, "[SETUP] sparse rows=%d cols=%d max_row_nnz=%d nnz=%ld mode=auto", SPARSE_ROWS, SPARSE_COLS, SPARSE_MAX_ROW_NNZ, total_nnz);

	}

	long row, limit;
	MalFor f = mal_for(SPARSE_ROWS, row, limit);
	const useconds_t delay_scale_percent = sparse_delay_scale_percent();

	mal_attach_mat(f, (void**)&row_nnz, sizeof(int), SPARSE_ROWS, 1, -1, MAL_ATTACH_PARTITIONED, MAL_ATTACH_INHERIT, MAL_ACCESS_READ_ONLY);
	mal_attach_mat(f, (void**)&col_idx, sizeof(int), SPARSE_ROWS, SPARSE_MAX_ROW_NNZ, -1, MAL_ATTACH_PARTITIONED, MAL_ATTACH_INHERIT, MAL_ACCESS_READ_ONLY);
	mal_attach_mat(f, (void**)&values, sizeof(double), SPARSE_ROWS, SPARSE_MAX_ROW_NNZ, -1, MAL_ATTACH_PARTITIONED, MAL_ATTACH_INHERIT, MAL_ACCESS_READ_ONLY);
	mal_attach_mat(f, (void**)&x, sizeof(double), 1, SPARSE_COLS, -1, MAL_ATTACH_SHARED_ACTIVE, MAL_ATTACH_INHERIT, MAL_ACCESS_READ_ONLY);
	mal_attach_mat(f, (void**)&y, sizeof(double), SPARSE_ROWS, 1, 0, MAL_ATTACH_PARTITIONED);

	for (; row < limit; row++) {

		double acc = 0.0;
		const int nnz = row_nnz[row];
		const long base = row * SPARSE_MAX_ROW_NNZ;

		for (int k = 0; k < nnz; k++) {

			acc += values[base + k] * x[col_idx[base + k]];

		}

		const int delay_ms = 4 + nnz / 2;
		const useconds_t delay_us = (useconds_t)delay_ms * 1000u * delay_scale_percent / 100u;
		y[row] = acc;

		MAL_LOG(MAL_LOG_INFO, "[ITER] row=%ld nnz=%d delay=%dms y=%.6f", row, nnz, delay_ms, acc);

		usleep(delay_us);

		mal_check_for(f);

	}

	mal_finalize();

	if (mal_rank() == 0) {

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

					MAL_LOG(MAL_LOG_ERROR, "[CHECK] row=%ld y=%.12f expected=%.12f err=%.3e", r, y[r], expected, err);

				}

			}

		}

		MAL_LOG(MAL_LOG_INFO, "[RESULT] sparse mat-vec %s (rows=%d cols=%d nnz=%ld errors=%d max_abs_err=%.3e)", errors == 0 ? "OK" : "WRONG", SPARSE_ROWS, SPARSE_COLS, total_nnz, errors, max_abs_err);

	}

	if (row_nnz) {

		std::free(row_nnz);

	}

	if (col_idx) {

		std::free(col_idx);

	}

	if (values) {

		std::free(values);

	}

	if (x) {

		std::free(x);

	}

	if (y) {

		std::free(y);

	}

	return EXIT_SUCCESS;

}
