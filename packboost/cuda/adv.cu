#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>

constexpr int WARP_SIZE = 32;


template <typename LeafT>
__global__ __launch_bounds__(32)
void advance_and_predict_kernel(
    int32_t* __restrict__ P,            // [N]
    const uint32_t* __restrict__ X,     // [R, M]
    const LeafT* __restrict__ L_old,    // [K0, Dm, N]
    LeafT* __restrict__ L_new,          // [K0, Dm, N]
    const int32_t* __restrict__ V,      // [rounds, K0, 2*nodes]
    const uint16_t* __restrict__ I,     // [rounds, K0, nodes]

    int N,
    int M,
    int K0,
    int Dm,
    int nodes,
    int tree_set,
    int stride)
{
    const int tree_fold = blockIdx.x;
    const int depth     = blockIdx.y;
    const int iblk      = blockIdx.z;
    const int lane      = threadIdx.x;

    if (tree_fold >= K0 || lane >= 32)
        return;

    /*
     * Number of active depths is exactly the same
     * as in the original implementation.
     *
     * For a given launch:
     *     depth = 0 ... depths-1
     */
    const int depths = min(tree_set + 1, Dm + 1);

    if (depth >= depths)
        return;

    /*
     * Base pointers for this tree/fold.
     *
     * V and I are tiny compared with X/L and are accessed
     * repeatedly by all threads of the warp.
     */
    const size_t vbase =
        ((size_t)tree_set * (size_t)K0 +
         (size_t)tree_fold) *
        (size_t)(2 * nodes);

    const size_t ibase =
        ((size_t)tree_set * (size_t)K0 +
         (size_t)tree_fold) *
        (size_t)nodes;

    const int lold_depth = depth - 1;

    /*
     * L_old/L_new are laid out:
     *
     * [tree_fold][depth][sample]
     *
     * Precompute the base addresses for this fold/depth.
     */
    const size_t lold_base =
        (lold_depth >= 0)
        ? (((size_t)tree_fold * (size_t)Dm +
            (size_t)lold_depth) * (size_t)N)
        : 0;

    const size_t lnew_base =
        (depth < Dm)
        ? (((size_t)tree_fold * (size_t)Dm +
            (size_t)depth) * (size_t)N)
        : 0;

    /*
     * Each warp processes consecutive groups of 32 samples.
     */
    for (int j = 0; j < stride; ++j) {

        const int k =
            (iblk * stride + j) * 32 + lane;

        if (k >= N)
            continue;

        /*
         * At depth 0 the previous leaf is always zero.
         */
        uint16_t leaf_prev = 0;

        if (depth > 0) {
            leaf_prev =
                (uint16_t)L_old[lold_base + (size_t)k];
        }

        /*
         * Tree node corresponding to this sample.
         */
        const int lo =
            leaf_prev + ((1 << depth) - 1);

        /*
         * I selects the bitplane.
         */
        const uint16_t li =
            I[ibase + (size_t)lo];

        /*
         * X is [R, M].
         *
         * Calculate the packed-word position once.
         */
        const int word_idx = k >> 5;
        const int bit_idx  = k & 31;

        const uint32_t word =
            X[(size_t)li * (size_t)M +
              (size_t)word_idx];

        const uint32_t bit =
            (word >> bit_idx) & 1u;

        /*
         * New leaf.
         */
        const uint16_t leaf_new =
            (uint16_t)((leaf_prev << 1) |
                       (uint16_t)bit);

        /*
         * Store leaf for the next boosting round.
         */
        if (depth < Dm) {
            L_new[lnew_base + (size_t)k] =
                (LeafT)leaf_new;
        }

        /*
         * Prediction contribution.
         */
        const size_t idx =
            (size_t)(2 * lo + 1 - (int)bit);

        const int32_t add =
            V[vbase + idx];

        /*
         * Exactly the same atomic behavior as the
         * original kernel.
         */
        atomicAdd(&P[k], add);
    }
}


template <typename LeafT>
static void launch_advpred_typed(
    torch::Tensor P,
    torch::Tensor X,
    torch::Tensor L_old,
    torch::Tensor L_new,
    torch::Tensor V,
    torch::Tensor I,
    int tree_set)
{
    const int N =
        (int)P.size(0);

    const int M =
        (int)X.size(1);

    const int K0 =
        (int)V.size(1);

    const int nodes2 =
        (int)V.size(2);

    const int nodes =
        nodes2 / 2;

    const int Dm =
        (int)L_old.size(1);

    const int depths =
        std::min(tree_set + 1, Dm + 1);

    /*
     * T4:
     *
     * One warp/block.
     *
     * 512 warps gives approximately one resident warp
     * per SM on a 40-SM T4, while keeping the launch
     * structure of the original fast kernel.
     */
    constexpr int zblocks = 512;

    const int samples_per_z =
        zblocks * WARP_SIZE;

    const int stride =
        std::max(
            1,
            (N + samples_per_z - 1) /
            samples_per_z
        );

    const dim3 grid(
        (unsigned)K0,
        (unsigned)depths,
        (unsigned)zblocks
    );

    const dim3 block(WARP_SIZE);

    auto stream =
        at::cuda::getCurrentCUDAStream();

    advance_and_predict_kernel<LeafT>
        <<<grid, block, 0, stream.stream()>>>(
            P.data_ptr<int32_t>(),
            reinterpret_cast<const uint32_t*>(
                X.data_ptr()),
            L_old.data_ptr<LeafT>(),
            L_new.data_ptr<LeafT>(),
            V.data_ptr<int32_t>(),
            I.data_ptr<uint16_t>(),
            N,
            M,
            K0,
            Dm,
            nodes,
            tree_set,
            stride
        );
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
    if (L_old.scalar_type() == at::kByte &&
        L_new.scalar_type() == at::kByte) {

        launch_advpred_typed<uint8_t>(
            P,
            X,
            L_old,
            L_new,
            V,
            I,
            tree_set
        );

    }
    else if (L_old.scalar_type() == at::kShort &&
             L_new.scalar_type() == at::kShort) {

        launch_advpred_typed<uint16_t>(
            P,
            X,
            L_old,
            L_new,
            V,
            I,
            tree_set
        );

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
