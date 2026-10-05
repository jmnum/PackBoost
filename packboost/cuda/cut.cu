#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <ATen/cuda/CUDAContext.h>
#include <math.h>

constexpr int WARP_SIZE = 32;
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

#ifndef CUTCUDA_MAX_FOLDS
#define CUTCUDA_MAX_FOLDS 32
#endif


__device__ __forceinline__
int depth_from_leaf(int leaf)
{
    return 31 - __clz(
        static_cast<unsigned>(leaf + 1)
    );
}


extern "C" __global__
void cut_cuda_kernel(
    const uint16_t* __restrict__ F,
    const uint8_t*  __restrict__ FST,
    const int64_t*  __restrict__ H,
    const int64_t*  __restrict__ H0,
    int32_t*        __restrict__ V,
    uint16_t*       __restrict__ I,

    int treesets,
    int K0,
    int K1,
    int nodes,
    int D,

    int tree_set,

    float L2,
    float lr,
    int qgrad_bits,
    int max_depth,

    float min_child_weight,
    float min_split_gain)
{
    /*
     * ------------------------------------------------------------
     * One warp = one fold
     * ------------------------------------------------------------
     */

    const int warp_id =
        threadIdx.x >> 5;

    const int lane =
        threadIdx.x & 31;


    if (warp_id >= K0)
        return;


    const int fold =
        warp_id;


    /*
     * ------------------------------------------------------------
     * One block handles one tree node.
     * ------------------------------------------------------------
     */

    const int leaf =
        blockIdx.x;


    if (leaf >= nodes)
        return;


    if ((unsigned)tree_set >=
        (unsigned)treesets)
        return;


    const int depth =
        depth_from_leaf(leaf);


    if ((unsigned)depth >=
        (unsigned)D)
        return;


    /*
     * ------------------------------------------------------------
     * Parent histogram
     * ------------------------------------------------------------
     */

    const size_t h0_base =
        (
            static_cast<size_t>(fold) *
            static_cast<size_t>(nodes) +
            static_cast<size_t>(leaf)
        ) * 2u;


    const float parent_G =
        static_cast<float>(
            H0[h0_base + 0]
        );


    const float parent_N =
        static_cast<float>(
            H0[h0_base + 1]
        );


    /*
     * ------------------------------------------------------------
     * Scaling
     * ------------------------------------------------------------
     */

    const float L2_eff =
        L2 *
        powf(
            2.0f,
            5.0f - static_cast<float>(depth)
        );


    const float qscale =
        lr *
        static_cast<float>(
            1u << (31 - qgrad_bits)
        ) *
        powf(
            2.0f,
            -static_cast<float>(
                max_depth - depth
            )
        );


    const bool use_min_child =
        min_child_weight > 0.0f;


    const bool use_min_gain =
        min_split_gain > 0.0f;


    /*
     * ------------------------------------------------------------
     * Best candidate for THIS lane.
     *
     * Lane = one of the 32 sampled features inside F[k].
     * ------------------------------------------------------------
     */

    float best_gain =
        -CUDART_INF_F;

    int best_left =
        0;

    int best_right =
        0;

    uint16_t best_k =
        0;


    /*
     * ------------------------------------------------------------
     * Search all feature sets.
     *
     * FST tells us which fold owns candidate k.
     * ------------------------------------------------------------
     */

    for (int k = 0;
         k < K1;
         ++k)
    {
        const size_t fst_idx =
            (
                (
                    static_cast<size_t>(tree_set) *
                    static_cast<size_t>(K1)
                )
                +
                static_cast<size_t>(k)
            )
            *
            static_cast<size_t>(D)
            +
            static_cast<size_t>(depth);


        const int candidate_fold =
            static_cast<int>(
                FST[fst_idx]
            );


        if (candidate_fold != fold)
            continue;


        /*
         * H[k, leaf, channel, lane]
         */

        const size_t h_base =
            (
                (
                    static_cast<size_t>(k) *
                    static_cast<size_t>(nodes)
                    +
                    static_cast<size_t>(leaf)
                )
                * 2u
            )
            * 32u
            +
            static_cast<size_t>(lane);


        const float G0 =
            static_cast<float>(
                H[h_base + 0 * 32u]
            );


        const float N0 =
            static_cast<float>(
                H[h_base + 1 * 32u]
            );


        const float G1 =
            parent_G - G0;


        const float N1 =
            parent_N - N0;


        /*
         * --------------------------------------------------------
         * Child constraints
         * --------------------------------------------------------
         */

        if (use_min_child)
        {
            if (N0 < min_child_weight ||
                N1 < min_child_weight)
            {
                continue;
            }
        }


        /*
         * --------------------------------------------------------
         * Leaf values
         * --------------------------------------------------------
         */

        const float V0f =
            G0 /
            (N0 + L2_eff);


        const float V1f =
            G1 /
            (N1 + L2_eff);


        /*
         * --------------------------------------------------------
         * Gain
         * --------------------------------------------------------
         */

        const float S0 =
            G0 * V0f;


        const float S1 =
            G1 * V1f;


        const float gain =
            S0 + S1;


        if (use_min_gain &&
            gain < min_split_gain)
        {
            continue;
        }


        /*
         * --------------------------------------------------------
         * Local best.
         *
         * IMPORTANT:
         * use float comparison, not __float_as_int(gain).
         * --------------------------------------------------------
         */

        if (gain > best_gain)
        {
            best_gain  = gain;
            best_left  =
                __float2int_rz(
                    qscale * V0f
                );

            best_right =
                __float2int_rz(
                    qscale * V1f
                );

            best_k =
                static_cast<uint16_t>(k);
        }
    }


    /*
     * ------------------------------------------------------------
     * Warp reduction: maximum gain
     * ------------------------------------------------------------
     */

    float warp_gain =
        best_gain;


    #pragma unroll
    for (int p = 0;
         p < 5;
         ++p)
    {
        const float other =
            __shfl_xor_sync(
                FULL_MASK,
                warp_gain,
                1 << p
            );

        warp_gain =
            (other > warp_gain)
            ? other
            : warp_gain;
    }


    /*
     * No valid split.
     */

    if (!isfinite(warp_gain))
        return;


    /*
     * ------------------------------------------------------------
     * Select the same winner lane as the old implementation:
     *
     * highest lane among equal best gains.
     * ------------------------------------------------------------
     */

    const unsigned winners =
        __ballot_sync(
            FULL_MASK,
            best_gain == warp_gain
        );


    if (winners == 0)
        return;


    const int winner_lane =
        31 - __clz(winners);


    /*
     * ------------------------------------------------------------
     * Only winner writes V/I.
     * ------------------------------------------------------------
     */

    if (lane == winner_lane)
    {
        const size_t v_base =
            (
                (
                    static_cast<size_t>(tree_set) *
                    static_cast<size_t>(K0)
                +
                    static_cast<size_t>(fold)
                )
                *
                static_cast<size_t>(2 * nodes)
            )
            +
            static_cast<size_t>(2 * leaf);


        V[v_base + 0] =
            best_left;


        V[v_base + 1] =
            best_right;


        /*
         * F is laid out:
         *
         * [tree_set, K1 * 32]
         *
         * each k owns 32 candidates.
         */

        const size_t f_idx =
            static_cast<size_t>(tree_set) *
            static_cast<size_t>(32 * K1)
            +
            static_cast<size_t>(best_k) *
            32u
            +
            static_cast<size_t>(winner_lane);


        const size_t i_idx =
            (
                (
                    static_cast<size_t>(tree_set) *
                    static_cast<size_t>(K0)
                +
                    static_cast<size_t>(fold)
                )
                *
                static_cast<size_t>(nodes)
            )
            +
            static_cast<size_t>(leaf);


        I[i_idx] =
            F[f_idx];
    }
}


/* ================================================================
 * Launcher
 * ================================================================ */

void cut_cuda_launcher(
    torch::Tensor F,
    torch::Tensor FST,
    torch::Tensor H,
    torch::Tensor H0,
    torch::Tensor V,
    torch::Tensor I,
    int tree_set,
    double L2,
    double lr,
    int qgrad_bits,
    int max_depth,
    double min_child_weight,
    double min_split_gain)
{
    const int treesets =
        static_cast<int>(F.size(0));

    const int K1 =
        static_cast<int>(H.size(0));

    const int nodes =
        static_cast<int>(H.size(1));

    const int K0 =
        static_cast<int>(H0.size(0));

    const int D =
        static_cast<int>(FST.size(2));


    TORCH_CHECK(
        K0 <= CUTCUDA_MAX_FOLDS,
        "K0=",
        K0,
        " exceeds CUTCUDA_MAX_FOLDS=",
        CUTCUDA_MAX_FOLDS
    );


    /*
     * One warp per fold.
     *
     * At K0=32:
     * 32 warps = 1024 threads.
     */

    TORCH_CHECK(
        K0 <= 32,
        "The new cut kernel supports at most 32 folds"
    );


    const dim3 grid(
        (1 << max_depth) - 1,
        1,
        1
    );


    const dim3 block(
        K0 * WARP_SIZE,
        1,
        1
    );


    auto stream =
        at::cuda::getCurrentCUDAStream();


    cut_cuda_kernel<<<
        grid,
        block,
        0,
        stream.stream()
    >>>(
        reinterpret_cast<const uint16_t*>(
            F.data_ptr<uint16_t>()
        ),

        FST.data_ptr<uint8_t>(),

        H.data_ptr<int64_t>(),

        H0.data_ptr<int64_t>(),

        V.data_ptr<int32_t>(),

        reinterpret_cast<uint16_t*>(
            I.data_ptr<uint16_t>()
        ),

        treesets,
        K0,
        K1,
        nodes,
        D,
        tree_set,

        static_cast<float>(L2),
        static_cast<float>(lr),
        qgrad_bits,
        max_depth,

        static_cast<float>(
            min_child_weight
        ),

        static_cast<float>(
            min_split_gain
        )
    );


    TORCH_CHECK(
        cudaGetLastError() == cudaSuccess,
        "cut_cuda_kernel launch failed: ",
        cudaGetErrorString(
            cudaGetLastError()
        )
    );
}
