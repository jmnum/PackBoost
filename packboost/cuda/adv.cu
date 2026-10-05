#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>

constexpr int WARP_SIZE = 32;


template <typename LeafT>
__global__ void advance_and_predict_kernel(
    int32_t* __restrict__ P,            // [N]
    const uint32_t* __restrict__ X,     // [R, M]
    const LeafT*  __restrict__ L_old,   // [K0, Dm, N]
    LeafT*        __restrict__ L_new,   // [K0, Dm, N]
    const int32_t* __restrict__ V,      // [rounds, K0, 2*nodes]
    const uint16_t* __restrict__ I,     // [rounds, K0, nodes]
    int N, int R, int M, int K0, int Dm, int nodes,
    int rounds, int tree_set,
    int stride)
{
    const int tree_fold = blockIdx.x;
    const int iblk      = blockIdx.z;
    const int wi        = threadIdx.x;

    if (wi >= WARP_SIZE || tree_fold >= K0)
        return;

    const size_t Vbase =
        ((size_t)tree_set * (size_t)K0 + (size_t)tree_fold)
        * (size_t)(2 * nodes);

    const size_t Ibase =
        ((size_t)tree_set * (size_t)K0 + (size_t)tree_fold)
        * (size_t)nodes;

    /*
     * IMPORTANT:
     * Keep the exact same depth definition as the original kernel.
     */
    const int depths = min(tree_set + 1, Dm + 1);

    /*
     * One local accumulator for all depths.
     *
     * Previously:
     *
     *     atomicAdd(&P[k], add);
     *
     * was executed once per depth.
     *
     * Now all contributions for this sample are accumulated locally
     * and only one atomicAdd is performed.
     */
    int p_add = 0;

    for (int j = 0; j < stride; ++j) {

        const int k = 32 * (stride * iblk + j) + wi;

        if (k >= N)
            continue;

        /*
         * Reset accumulator for this sample.
         */
        p_add = 0;

        /*
         * Preserve the original depth traversal exactly.
         */
        for (int depth = 0; depth < depths; ++depth) {

            uint16_t leaf_prev = 0;

            if (depth > 0) {
                const size_t off_old =
                    (((size_t)tree_fold * (size_t)Dm)
                     + (size_t)(depth - 1))
                    * (size_t)N
                    + (size_t)k;

                leaf_prev = (uint16_t)L_old[off_old];
            }

            const int lo =
                (int)leaf_prev + ((1 << depth) - 1);

            const uint16_t li =
                I[Ibase + (size_t)lo];

            const uint32_t word =
                X[(size_t)li * (size_t)M + (size_t)(k >> 5)];

            const uint32_t bit =
                (word >> (k & 31)) & 1u;

            const uint16_t leaf_new =
                (uint16_t)((leaf_prev << 1) | (uint16_t)bit);

            if (depth < Dm) {
                const size_t off_new =
                    (((size_t)tree_fold * (size_t)Dm)
                     + (size_t)depth)
                    * (size_t)N
                    + (size_t)k;

                L_new[off_new] = (LeafT)leaf_new;
            }

            const size_t idx =
                (size_t)(2 * lo + 1 - (int)bit);

            /*
             * Accumulate locally instead of doing an atomicAdd here.
             */
            p_add += V[Vbase + idx];
        }

        /*
         * ONE atomic operation per sample instead of
         * one atomic operation per depth.
         */
        if (p_add != 0) {
            atomicAdd(&P[k], p_add);
        }
    }
}


static void launch_advpred(
    torch::Tensor P,
    torch::Tensor X,
    torch::Tensor L_old,
    torch::Tensor L_new,
    torch::Tensor V,
    torch::Tensor I,
    int tree_set)
{
    const int N = (int)P.size(0);
    const int R = (int)X.size(0);
    const int M = (int)X.size(1);

    const int rounds = (int)V.size(0);
    const int K0     = (int)V.size(1);
    const int nodes2 = (int)V.size(2);
    const int nodes  = nodes2 / 2;

    const int Dm = (int)L_old.size(1);

    /*
     * Same depth count as the original implementation.
     */
    const int depths =
        std::min(tree_set + 1, Dm + 1);

    /*
     * Keep the existing z decomposition.
     *
     * Each warp processes a contiguous set of samples.
     */
    const int zblocks = 512;

    int stride =
        (N + (zblocks * 32) - 1)
        / (zblocks * 32);

    if (stride < 1)
        stride = 1;

    const int gz = std::max(1, zblocks);

    /*
     * IMPORTANT:
     *
     * depth is now handled inside the kernel.
     *
     * Before:
     *     grid = [K0, depths, gz]
     *
     * Now:
     *     grid = [K0, 1, gz]
     *
     * This allows one thread to accumulate all depth
     * contributions before the single atomicAdd.
     */
    const dim3 grid(
        (unsigned)K0,
        1,
        (unsigned)gz
    );

    const dim3 block(WARP_SIZE);

    auto stream =
        at::cuda::getCurrentCUDAStream();

    if (L_old.scalar_type() == at::kByte &&
        L_new.scalar_type() == at::kByte) {

        advance_and_predict_kernel<uint8_t>
            <<<grid, block, 0, stream.stream()>>>(
                P.data_ptr<int32_t>(),
                reinterpret_cast<const uint32_t*>(
                    X.data_ptr()),
                L_old.data_ptr<uint8_t>(),
                L_new.data_ptr<uint8_t>(),
                V.data_ptr<int32_t>(),
                reinterpret_cast<const uint16_t*>(
                    I.data_ptr<uint16_t>()),
                N, R, M, K0, Dm, nodes,
                rounds, tree_set, stride);

    }
    else if (L_old.scalar_type() == at::kShort &&
             L_new.scalar_type() == at::kShort) {

        advance_and_predict_kernel<uint16_t>
            <<<grid, block, 0, stream.stream()>>>(
                P.data_ptr<int32_t>(),
                reinterpret_cast<const uint32_t*>(
                    X.data_ptr()),
                L_old.data_ptr<uint16_t>(),
                L_new.data_ptr<uint16_t>(),
                V.data_ptr<int32_t>(),
                reinterpret_cast<const uint16_t*>(
                    I.data_ptr<uint16_t>()),
                N, R, M, K0, Dm, nodes,
                rounds, tree_set, stride);

    }
    else {

        TORCH_CHECK(
            false,
            "L_old/L_new must both be uint8 or both be "
            "uint16 (torch.int16 payload)"
        );
    }
}


void advance_and_predict_launcher(
    torch::Tensor P,
    torch::Tensor X,
    torch::Tensor L_old,
    torch::Tensor L_new,
    torch::Tensor V,
    torch::Tensor I,
    int tree_set)
{
    launch_advpred(
        P.contiguous(),
        X.contiguous(),
        L_old.contiguous(),
        L_new.contiguous(),
        V.contiguous(),
        I.contiguous(),
        tree_set
    );
}
