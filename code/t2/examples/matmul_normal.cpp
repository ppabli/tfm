#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include "example_utils.hpp"

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

void build_gatherv_layout(long total_rows, int row_width, int world_size, std::vector<int>& counts, std::vector<int>& displs) {

	counts.resize(world_size);
	displs.resize(world_size);

	for (int r = 0; r < world_size; r++) {

		const BlockRange rr = block_range(total_rows, r, world_size);
		counts[r] = static_cast<int>(rr.count * row_width);
		displs[r] = static_cast<int>(rr.start * row_width);

	}

}

int run_matvec(int world_rank, int world_size, int M, int K, double* elapsed_out) {

	double* A = static_cast<double*>(std::malloc(static_cast<size_t>(M * K) * sizeof(double)));
	double* x = static_cast<double*>(std::malloc(static_cast<size_t>(K) * sizeof(double)));
	double* y = (world_rank == 0) ? static_cast<double*>(std::malloc(static_cast<size_t>(M) * sizeof(double))) : nullptr;

	if (world_rank == 0) {

		for (int r = 0; r < M; r++) {

			for (int c = 0; c < K; c++) {

				A[r * K + c] = static_cast<double>(r * K + c + 1);

			}

		}

		for (int c = 0; c < K; c++) {

			x[c] = 1.0;

		}

	}

	MPI_Barrier(MPI_COMM_WORLD);
	const double t0 = MPI_Wtime();

	MPI_Bcast(A, M * K, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(x, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	const BlockRange range = block_range(M, world_rank, world_size);
	std::vector<double> local_y(static_cast<size_t>(range.count));
	const useconds_t delay_us = example_delay_us(100000);

	for (long local_i = 0; local_i < range.count; local_i++) {

		const long row = range.start + local_i;
		double acc = 0.0;

		for (long k = 0; k < K; k++) {

			acc += A[row * K + k] * x[k];

		}

		local_y[static_cast<size_t>(local_i)] = acc;
		usleep(delay_us);

	}

	std::vector<int> recv_counts;
	std::vector<int> recv_displs;
	if (world_rank == 0) {

		build_gatherv_layout(M, 1, world_size, recv_counts, recv_displs);

	}

	MPI_Gatherv(local_y.data(), static_cast<int>(range.count), MPI_DOUBLE, y, (world_rank == 0) ? recv_counts.data() : nullptr, (world_rank == 0) ? recv_displs.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	const double t1 = MPI_Wtime();

	if (world_rank == 0) {

		*elapsed_out = t1 - t0;

	}

	int errors = 0;

	if (world_rank == 0) {

		for (int r = 0; r < M; r++) {

			double expected = 0.0;
			for (int c = 0; c < K; c++) {

				expected += static_cast<double>(r * K + c + 1);

			}

			if (std::fabs(y[r] - expected) > 1e-9) {

				errors = 1;
				break;

			}

		}

		std::printf("[RESULT] mat-vec %s (%d errors)\n", errors == 0 ? "OK" : "WRONG", errors);
		std::free(y);

	}

	std::free(A);
	std::free(x);

	return errors;

}

int run_matmul(int world_rank, int world_size, int M, int K, int N, double* elapsed_out) {

	double* A = static_cast<double*>(std::malloc(static_cast<size_t>(M * K) * sizeof(double)));
	double* B = static_cast<double*>(std::malloc(static_cast<size_t>(K * N) * sizeof(double)));
	double* C = (world_rank == 0) ? static_cast<double*>(std::malloc(static_cast<size_t>(M * N) * sizeof(double))) : nullptr;

	if (world_rank == 0) {

		for (int r = 0; r < M; r++) {

			for (int c = 0; c < K; c++) {

				A[r * K + c] = static_cast<double>(r + 1);

			}

		}

		for (int r = 0; r < K; r++) {

			for (int c = 0; c < N; c++) {

				B[r * N + c] = 1.0;

			}

		}

	}

	MPI_Barrier(MPI_COMM_WORLD);
	const double t0 = MPI_Wtime();

	MPI_Bcast(A, M * K, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(B, K * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	const BlockRange range = block_range(M, world_rank, world_size);
	std::vector<double> local_c(static_cast<size_t>(range.count * N));
	const useconds_t delay_us = example_delay_us(100000);

	for (long local_i = 0; local_i < range.count; local_i++) {

		const long row = range.start + local_i;

		for (long j = 0; j < N; j++) {

			double acc = 0.0;

			for (long k = 0; k < K; k++) {

				acc += A[row * K + k] * B[k * N + j];

			}

			local_c[static_cast<size_t>(local_i * N + j)] = acc;

		}

		usleep(delay_us);

	}

	std::vector<int> recv_counts;
	std::vector<int> recv_displs;
	if (world_rank == 0) {

		build_gatherv_layout(M, N, world_size, recv_counts, recv_displs);

	}

	MPI_Gatherv(local_c.data(), static_cast<int>(range.count * N), MPI_DOUBLE, C, (world_rank == 0) ? recv_counts.data() : nullptr, (world_rank == 0) ? recv_displs.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	const double t1 = MPI_Wtime();

	if (world_rank == 0) {

		*elapsed_out = t1 - t0;

	}

	int errors = 0;

	if (world_rank == 0) {

		for (int r = 0; r < M && errors == 0; r++) {

			for (int c = 0; c < N; c++) {

				if (std::fabs(C[r * N + c] - static_cast<double>(r + 1) * K) > 1e-9) {

					errors = 1;
					break;

				}

			}

		}

		std::printf("[RESULT] mat-mat %s (%d errors)\n", errors == 0 ? "OK" : "WRONG", errors);
		std::free(C);

	}

	std::free(A);
	std::free(B);

	return errors;

}

int main(int argc, char* argv[]) {

	MPI_Init(&argc, &argv);

	int world_rank = 0;
	int world_size = 1;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	const int M = static_cast<int>(parse_arg_long(argc, argv, "m", 12));
	const int K = static_cast<int>(parse_arg_long(argc, argv, "k", 6));
	const int N = static_cast<int>(parse_arg_long(argc, argv, "n", 4));

	const bool do_mv = (argc > 1 && std::strcmp(argv[1], "mv") == 0);
	double elapsed = 0.0;

	if (do_mv) {

		run_matvec(world_rank, world_size, M, K, &elapsed);

	} else {

		run_matmul(world_rank, world_size, M, K, N, &elapsed);

	}

	if (world_rank == 0) {

		std::printf("[TIME] %s normal mpi np=%d seconds=%.6f\n", do_mv ? "mat-vec" : "mat-mat", world_size, elapsed);

	}

	MPI_Finalize();
	return EXIT_SUCCESS;

}
