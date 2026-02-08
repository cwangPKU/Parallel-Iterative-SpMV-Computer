#include <stdio.h>
#include <stdlib.h>
#include <time.h>   // For timing
#include <string.h> // For strcmp, strtok
#include <math.h>   // For fabs (for potential future error checking in C)
#include <vector>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <immintrin.h>  // For AVX2 intrinsics
#include <utility>
#include <cstring>
#include <Eigen/Dense>

// --- CSR Matrix Structure ---
typedef struct {
    int num_rows;
    int num_cols;
    int num_nonzeros;
    double* values;
    int* col_indices;
    int* row_ptr;
} CSRMatrix;

// --- Function to free CSRMatrix memory ---
void free_csr_matrix(CSRMatrix* A) {
    if (A) {
        free(A->values);
        free(A->col_indices);
        free(A->row_ptr);
    }
}

// --- Function to read CSR matrix from file ---
int read_csr_matrix(const char* filename, CSRMatrix* A) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Cannot open matrix file '%s'\n", filename);
        return 0;
    }

    if (fscanf(file, "%d %d %d\n", &(A->num_rows), &(A->num_cols), &(A->num_nonzeros)) != 3) {
        fprintf(stderr, "Error reading matrix header from '%s'.\n", filename);
        fclose(file);
        return 0;
    }

    if (A->num_rows < 0 || A->num_cols < 0 || A->num_nonzeros < 0) {
        fprintf(stderr, "Error: Invalid matrix dimensions or non-zero count in header of '%s'.\n", filename);
        fclose(file);
        return 0;
    }

    A->row_ptr = (int*)malloc((A->num_rows + 1) * sizeof(int));
    if (!A->row_ptr) {
        perror("Memory allocation failed for row_ptr");
        fclose(file);
        return 0;
    }

    for (int i = 0; i <= A->num_rows; ++i) {
        if (fscanf(file, "%d", &(A->row_ptr[i])) != 1) {
            fprintf(stderr, "Error reading row_ptr element %d from '%s'.\n", i, filename);
            free(A->row_ptr); A->row_ptr = NULL;
            fclose(file);
            return 0;
        }
    }
    char ch;
    while ((ch = fgetc(file)) != '\n' && ch != EOF);

    if (A->num_nonzeros > 0) {
        A->col_indices = (int*)malloc(A->num_nonzeros * sizeof(int));
        A->values = (double*)malloc(A->num_nonzeros * sizeof(double));
        if (!A->col_indices || !A->values) {
            perror("Memory allocation failed for col_indices or values");
            free(A->row_ptr); A->row_ptr = NULL;
            free(A->col_indices); A->col_indices = NULL;
            free(A->values); A->values = NULL;
            fclose(file);
            return 0;
        }

        for (int i = 0; i < A->num_nonzeros; ++i) {
            if (fscanf(file, "%d", &(A->col_indices[i])) != 1) {
                fprintf(stderr, "Error reading col_indices element %d from '%s'.\n", i, filename);
                free_csr_matrix(A); A->row_ptr = NULL; A->col_indices = NULL; A->values = NULL;
                fclose(file);
                return 0;
            }
        }
        while ((ch = fgetc(file)) != '\n' && ch != EOF);

        for (int i = 0; i < A->num_nonzeros; ++i) {
            if (fscanf(file, "%lf", &(A->values[i])) != 1) {
                fprintf(stderr, "Error reading values element %d from '%s'.\n", i, filename);
                free_csr_matrix(A); A->row_ptr = NULL; A->col_indices = NULL; A->values = NULL;
                fclose(file);
                return 0;
            }
        }
    }
    else {
        A->col_indices = NULL;
        A->values = NULL;
    }

    fclose(file);
    return 1;
}

// --- Serial SpMV: y = Ax (CSR format) ---
void spmv_csr_serial(const CSRMatrix* A, const double* x, double* y) {
    for (int i = 0; i < A->num_rows; ++i) {
        y[i] = 0.0;
        for (int k = A->row_ptr[i]; k < A->row_ptr[i + 1]; ++k) {
            y[i] += A->values[k] * x[A->col_indices[k]];
        }
    }
}

// --- Function to write vector to file ---
int write_vector_to_file(const char* filename, const double* vec, int size) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error: Cannot open output file '%s' for writing vector.\n", filename);
        return 0;
    }
    for (int i = 0; i < size; ++i) {
        fprintf(file, "%.15e\n", vec[i]); // High precision output
    }
    fclose(file);
    printf("Final vector written to %s\n", filename);
    return 1;
}



typedef struct {
    int num_block_rows;
    int num_block_cols;
    int num_blocks;
    int block_size;
    int* block_row_ptr;
    int* block_col_indices;
    double* block_values;
} BCSRMatrix;

void allocate_bcsr_memory(const CSRMatrix& A, int block_size, BCSRMatrix& B, int*& tmp_flags, int*& block_col_counts) {
    B.block_size = block_size;
    B.num_block_rows = (A.num_rows + block_size - 1) / block_size;
    B.num_block_cols = (A.num_cols + block_size - 1) / block_size;

    B.block_row_ptr = new int[B.num_block_rows + 1]();
    tmp_flags = new int[B.num_block_cols]();
    block_col_counts = new int[B.num_block_rows]();
}

void csr_to_bcsr_parallel(const CSRMatrix& A, BCSRMatrix& B, int* tmp_flags, int* block_col_counts) {
    int block_size = B.block_size;
    int num_block_rows = B.num_block_rows;
    int num_block_cols = B.num_block_cols;
    
    // Count blocks per block-row
    #pragma omp parallel for
    for (int br = 0; br < num_block_rows; ++br) {
        std::vector<int> local_flags(num_block_cols, 0);
        int row_start = br * block_size;
        int row_end = std::min(row_start + block_size, A.num_rows);

        for (int r = row_start; r < row_end; ++r) {
            for (int i = A.row_ptr[r]; i < A.row_ptr[r + 1]; ++i) {
                int bc = A.col_indices[i] / block_size;
                local_flags[bc] = 1;
            }
        }

        int count = std::count(local_flags.begin(), local_flags.end(), 1);
        block_col_counts[br] = count;
    }

    // Compute block_row_ptr
    B.block_row_ptr[0] = 0;
    for (int i = 0; i < num_block_rows; ++i) {
        B.block_row_ptr[i + 1] = B.block_row_ptr[i] + block_col_counts[i];
    }
    B.num_blocks = B.block_row_ptr[num_block_rows];

    // Allocate blocks
    B.block_col_indices = new int[B.num_blocks]();
    B.block_values = new double[B.num_blocks * block_size * block_size]();

    // Fill block_col_indices
    #pragma omp parallel for
    for (int br = 0; br < num_block_rows; ++br) {
        std::vector<int> local_flags(num_block_cols, 0);
        int row_start = br * block_size;
        int row_end = std::min(row_start + block_size, A.num_rows);
        int ptr = B.block_row_ptr[br];

        for (int r = row_start; r < row_end; ++r) {
            for (int i = A.row_ptr[r]; i < A.row_ptr[r + 1]; ++i) {
                int bc = A.col_indices[i] / block_size;
                if (!local_flags[bc]) {
                    local_flags[bc] = 1;
                    B.block_col_indices[ptr++] = bc;
                }
            }
        }
    }

    // Fill block values
    #pragma omp parallel for
    for (int br = 0; br < num_block_rows; ++br) {
        int* block_col_to_idx = new int[B.num_block_cols]();
        std::fill(block_col_to_idx, block_col_to_idx + num_block_cols, -1);
        int start = B.block_row_ptr[br];
        int end = B.block_row_ptr[br + 1];
        for (int i = start; i < end; ++i) {
            block_col_to_idx[B.block_col_indices[i]] = i;
        }

        int row_start = br * block_size;
        int row_end = std::min(row_start + block_size, A.num_rows);
        for (int r = row_start; r < row_end; ++r) {
            int local_row = r % block_size;
            for (int i = A.row_ptr[r]; i < A.row_ptr[r + 1]; ++i) {
                int c = A.col_indices[i];
                int bc = c / block_size;
                int local_col = c % block_size;
                int block_idx = block_col_to_idx[bc];
                int offset = block_idx * block_size * block_size + local_row * block_size + local_col;
                B.block_values[offset] = A.values[i];
            }
        }
    }
}

void free_bcsr(BCSRMatrix& B) {
    delete[] B.block_row_ptr;
    delete[] B.block_col_indices;
    delete[] B.block_values;
    B.block_row_ptr = nullptr;
    B.block_col_indices = nullptr;
    B.block_values = nullptr;
}



// --- Main function ---
int main(int argc, char* argv[]) {
    const char* matrix_filename = NULL;
    int num_iterations = 0;
    const char* output_vec_filename = NULL;

    // Parse command line arguments
    if (argc < 2) { // At least program_name, matrix_file, num_iterations
        fprintf(stderr, "Usage: %s <matrix_file.csr> [-o <output_vector_file>]\n", argv[0]);
        return 1;
    }

    matrix_filename = argv[1];
    num_iterations = 3000;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "-o") == 0) {
            if (i + 1 < argc) {
                output_vec_filename = argv[i + 1];
                i++; // Consume the filename argument
            }
            else {
                fprintf(stderr, "Error: -o option requires a filename.\n");
                return 1;
            }
        }
        else {
            fprintf(stderr, "Error: Unknown option %s\n", argv[i]);
            return 1;
        }
    }

    CSRMatrix A;
    printf("Loading matrix from %s...\n", matrix_filename);
    if (!read_csr_matrix(matrix_filename, &A)) {
        fprintf(stderr, "Failed to load CSR matrix.\n");
        return 1;
    }
    printf("Matrix loaded: %d rows, %d cols, %d non-zeros.\n", A.num_rows, A.num_cols, A.num_nonzeros);

    if (A.num_nonzeros > 0 && A.num_rows != A.num_cols) {
        fprintf(stderr, "Error: For x_{i+1} = A x_i, matrix A must be square (num_rows == num_cols).\n");
        fprintf(stderr, "Matrix dimensions are %d rows and %d cols.\n", A.num_rows, A.num_cols);
        free_csr_matrix(&A);
        return 1;
    }

    double* x_0 = (double*)malloc(A.num_cols * sizeof(double));
    double* x_current = (double*)malloc(A.num_cols * sizeof(double));
    double* x_next = (double*)malloc(A.num_rows * sizeof(double));


    const int total_threads = omp_get_max_threads();



    if (!x_current || !x_next) {
        perror("Memory allocation failed for vectors");
        free_csr_matrix(&A); free(x_current); free(x_next);
        return 1;
    }

    for (int i = 0; i < A.num_cols; ++i) {
        x_0[i] = i % 10; // Initialize x_0
        x_current[i] = i % 10; // Initialize x_current
    }


    printf("Starting %d iterations of SpMV (x_new = A * x_old + x_0)...\n", num_iterations);
    
    int block_size = 6;
    BCSRMatrix B;
    int* tmp_flags = nullptr, * block_col_counts = nullptr;
    allocate_bcsr_memory(A, block_size, B, tmp_flags, block_col_counts);
    csr_to_bcsr_parallel(A, B, tmp_flags, block_col_counts);


    double start_time = omp_get_wtime();
    
    // start the iterations
    /*
    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();

        int chunk = (B.num_block_rows + num_threads - 1) / num_threads;

        double* arr = new double[B.block_size]();

        for (int iter = 0; iter < num_iterations; ++iter)
        {
            #pragma omp for schedule(static, chunk)
            for (int br = 0; br < B.num_block_rows; ++br)
            {
                int row_start = br * B.block_size;
                int row_end = std::min(row_start + B.block_size, A.num_rows);
                //double* arr = new double[row_end - row_start]();
                //memset(arr, 0, B.block_size * sizeof(double));
                for (int bi = B.block_row_ptr[br]; bi < B.block_row_ptr[br + 1]; ++bi)
                {
                    
                    int bc = B.block_col_indices[bi];
                    int block_offset = bi * B.block_size * B.block_size;
                    int col_start = bc * B.block_size;
                    int col_end = std::min(col_start + B.block_size, A.num_cols);
                    int col_offset = bc * B.block_size;

                    
                    for (int i = 0; i < row_end - row_start; i++)
                    {
                        
                        for (int j = 0; j < col_end - col_start; j++)
                        {
                            arr[i] += B.block_values[block_offset + i * B.block_size + j] * x_current[col_offset + j];
                        }
                    }
                    
                }
                
                for (int i = 0; i < row_end - row_start; ++i)
                {
                    x_next[row_start + i] = x_0[i]; + arr[i];
                    arr[i] = 0;
                }
                
            }
            #pragma omp single
            {
                std::swap(x_current, x_next);
            }
        }
    }
    */
    
    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int chunk = (B.num_block_rows + num_threads - 1) / num_threads;

        double* arr = new double[B.block_size]();

        for (int iter = 0; iter < num_iterations; ++iter)
        {
            #pragma omp for schedule(static, chunk)
            for (int br = 0; br < B.num_block_rows; ++br)
            {
                int row_start = br * B.block_size;
                int row_end = std::min(row_start + B.block_size, A.num_rows);
                int block_row_len = row_end - row_start;


                Eigen::Index block_row_len_e = static_cast<Eigen::Index>(block_row_len);

                for (int bi = B.block_row_ptr[br]; bi < B.block_row_ptr[br + 1]; ++bi)
                {
                    int bc = B.block_col_indices[bi];
                    int col_start = bc * B.block_size;
                    int col_end = std::min(col_start + B.block_size, A.num_cols);
                    int block_col_len = col_end - col_start;

                    
                    Eigen::Index block_col_len_e = static_cast<Eigen::Index>(block_col_len);

                    int block_offset = bi * B.block_size * B.block_size;
                    int col_offset = bc * B.block_size;

                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                        block_mat(&B.block_values[block_offset], block_row_len_e, block_col_len_e);

                    Eigen::Map<const Eigen::VectorXd> x_vec(&x_current[col_offset], block_col_len_e);

                    Eigen::Map<Eigen::VectorXd> arr_vec(arr, block_row_len_e);
                    arr_vec.noalias() += block_mat * x_vec;
                }

                for (int i = 0; i < block_row_len; ++i)
                {
                    x_next[row_start + i] = x_0[i] + arr[i];
                    arr[i] = 0; 
                }
            }

            #pragma omp single
            {
                std::swap(x_current, x_next);
            }
        }

        delete[] arr;
    }
    


    double end_time = omp_get_wtime();
    double cpu_time_used = end_time - start_time;

    std::cout << B.num_blocks << " " << B.num_block_rows << std::endl;
    std::cout << "Converted CSR to BCSR." << std::endl;


    printf("Iterations completed.\n");
    printf("Time taken for %d iterations: %f seconds\n", num_iterations, cpu_time_used);

    // Output final vector if filename is provided
    if (output_vec_filename) {
        if (!write_vector_to_file(output_vec_filename, x_current, A.num_cols)) {
            fprintf(stderr, "Failed to write output vector.\n");
            // Continue to cleanup, but indicate potential issue
        }
    }
    else {
        // If no output file, print some values to console for quick check
        printf("First 5 elements of final x_current (or fewer if N < 5):\n");
        for (int i = 0; i < A.num_cols && i < 5; ++i) {
            printf("x[%d] = %.6e\n", i, x_current[i]);
        }
        double checksum = 0.0;
        for (int i = 0; i < A.num_cols; ++i) checksum += x_current[i];
        printf("Checksum of final x_current: %.6e\n", checksum);
    }

    free_csr_matrix(&A);
    free(x_current);
    free(x_next);

    return 0;
}