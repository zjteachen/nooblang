/*
 * Hand-written CUDA kernels for fused quantized GEMM operations.
 * Scope: quantize * quantized, and quantized * unquantized.
 * Arguments:
 * t1, t2: tensors A and B to matmul
 * out: output target memory buffer
 */
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

typedef unsigned char uchar;

// Note that other configurations for this are untested.
// Avoid BLOCK_ROWS != BLOCK_COLS.
#define BLOCK_ROWS 16
#define BLOCK_COLS 16
#define SLICE_LEN 16

/*
 * A represents an MxK array, but it is quantized so that each pair of int4s is
 * in a single uchar. B is KxN. Given also is a tensor of start values/leap
 * values for dequantization. |start| = |leap| = (M*K)/pack_size.
 */
template <typename T>
__global__ void
fused_matmul_quant_unquant_INT4(const uchar *A, const T *B, T *C, T *starts,
                                T *leaps, int pack_size, int M, int N, int K) {
  // indices of C which this thread will calculate
  int true_r = blockDim.y * blockIdx.y + threadIdx.y;
  int true_c = blockDim.x * blockIdx.x + threadIdx.x;
  /*
  Each block calculates a BLOCK_ROWSxBLOCK_COLS tile of C.
  Requires BLOCK_ROWS rows of A starting at r (0<=r<M), and BLOCK_COLS
  columns of B starting at c (0<=c<N).
  To calculate C[i, j], where 0<=i<M and 0<=j<N, we need the dot product of
  row i of A and column j of B.
  For efficient memory usage, we can imagine that each 16x16 thread block
  calculates the BLOCK_ROWS*BLOCK_COLS equivalent row-col dot products from A
  and B; but instead of calculating each dot-product at a time, we calculate all
  of them in slices along K. So we calculate every dot prod simultaneously per
  slice.
  */
  // These shared memory store needed mem per slice
  // This kernel allocates threads with BLOCK_ROWSxBLOCK_COLS granularity,
  __shared__ float a[BLOCK_ROWS][SLICE_LEN]; // values of
  __shared__ float b[BLOCK_COLS][SLICE_LEN];
  float dot = 0;
  for (int k = 0; k < K; k += SLICE_LEN) {
    // I think in the old version of this kernel,
    // it was very confusing which thread corresponded to which memory
    // loading responsibility. Since usually BLOCK_ROWS=BLOCK_COLS=SLICE_LEN,
    // the semantic meaning of all 3 could be swapped interchangably
    // which allowed for some heavy abuse of notation.
    // For now let's assume this invariant holds,
    // so smem for a = smem for b and there is no asymmetry in weight loading.
    // So we just need to load 256 values for each.
    // a[i][j] = A value at virtual row i and K-virtual col value j, *1
    // b[i][j] = B value at virtual col i and K-virtual row value j. *2

    // load in A chunk, offset by col n.
    int a_row = true_r;
    int a_col = k + threadIdx.x;
    int a_true_idx = a_row * K + a_col;
    int dequant_idx = a_true_idx / pack_size;
    uchar dequant_val = (a_row < M && a_col < K) ? A[a_true_idx / 2] : 0;
    dequant_val = (dequant_val >> (4 * (a_true_idx % 2))) & 0xF;

    a[threadIdx.y][threadIdx.x] =
        (a_row < M && a_col < K)
            ? (starts[dequant_idx] + leaps[dequant_idx] * dequant_val)
            : static_cast<T>(0);
    // load in B chunk, offset by row n.
    int b_col = true_c;
    int b_row = k + threadIdx.y;
    b[threadIdx.x][threadIdx.y] =
        (b_row < K && b_col < N) ? B[b_row * N + b_col] : static_cast<T>(0);
    __syncthreads();

    // After the shared memory processing, the shared memory should satisfy *1
    // and *2. threadIdx.y is the virtual row, and threadIdx.x is the virtual
    // col. We iterate through every K-dim element and dot the needed values.
    for (int i = 0; i < SLICE_LEN; ++i) {
      dot += a[threadIdx.y][i] * b[threadIdx.x][i];
    }
    __syncthreads();
  }

  if (true_r < M && true_c < N) {
    C[true_r * N + true_c] = dot;
  }
}

__global__ void fused_matmul_quant_quant_INT4() {}

template <typename T>
void launch_fused_matmul_quant_unquant_INT4(const uchar *A, const T *B, T *C,
                                            T *starts, T *leaps, int pack_size,
                                            int M, int N, int K,
                                            cudaStream_t stream) {
  dim3 threadsPerBlock(BLOCK_ROWS, BLOCK_COLS);
  dim3 blocksPerGrid((N + BLOCK_COLS - 1) / BLOCK_COLS,
                     (M + BLOCK_ROWS - 1) / BLOCK_ROWS);

  fused_matmul_quant_unquant_INT4<T>
      <<<blocksPerGrid, threadsPerBlock, 0, stream>>>(A, B, C, starts, leaps,
                                                      pack_size, M, N, K);
}

torch::Tensor fused_matmul_quant_unquant(torch::Tensor A, torch::Tensor B,
                                         torch::Tensor starts,
                                         torch::Tensor leaps, int pack_size) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && starts.is_cuda() && leaps.is_cuda(),
              "all tensors must be on CUDA");
  TORCH_CHECK(A.is_contiguous() && B.is_contiguous(),
              "A and B must be contiguous");
  TORCH_CHECK(A.dtype() == torch::kUInt8, "A must be packed uint8");
  TORCH_CHECK(B.dtype() == starts.dtype() && B.dtype() == leaps.dtype(),
              "B, starts, and leaps must share one compute dtype");

  int M = A.size(0);
  int N = B.size(1);
  int K = A.size(1) * 2;
  TORCH_CHECK(B.size(0) == K,
              "B's leading dimension must match A's unpacked width");

  auto C = torch::empty({M, N}, B.options());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16, at::ScalarType::Half, B.scalar_type(),
      "fused_matmul_quant_unquant_INT4", [&] {
        launch_fused_matmul_quant_unquant_INT4<scalar_t>(
            A.data_ptr<uchar>(), B.data_ptr<scalar_t>(), C.data_ptr<scalar_t>(),
            starts.data_ptr<scalar_t>(), leaps.data_ptr<scalar_t>(), pack_size,
            M, N, K, at::cuda::getCurrentCUDAStream());
      });

  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_matmul_quant_unquant", &fused_matmul_quant_unquant,
        "Fused INT4-dequant x unquantized matmul");
}
