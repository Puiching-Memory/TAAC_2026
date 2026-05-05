# pyright: reportInvalidTypeForm=false

"""TileLang kernel definitions for PCVR operators."""

import tilelang as tl
import tilelang.language as T


def build_flash_attention_forward_kernel(
    batch: int,
    heads: int,
    query_len: int,
    kv_len: int,
    dim: int,
    *,
    is_causal: bool,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
    tl_dtype,
    accum_dtype,
):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    pass_configs = {
        tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }

    @tl.jit(out_idx=[4], target="cuda", pass_configs=pass_configs)
    def flash_attention_forward_kernel(
        batch: int,
        heads: int,
        query_len: int,
        kv_len: int,
        dim: int,
        dtype: T.dtype = tl_dtype,
        acc_dtype: T.dtype = accum_dtype,
    ):
        @T.prim_func
        def main(
            q: T.Tensor((batch, query_len, heads, dim), dtype),
            k: T.Tensor((batch, kv_len, heads, dim), dtype),
            v: T.Tensor((batch, kv_len, heads, dim), dtype),
            key_lengths: T.Tensor((batch,), "int32"),
            out: T.Tensor((batch, query_len, heads, dim), dtype),
        ):
            with T.Kernel(T.ceildiv(query_len, block_m), heads, batch, threads=threads) as (bx, by, bz):
                q_shared = T.alloc_shared((block_m, dim), dtype)
                k_shared = T.alloc_shared((block_n, dim), dtype)
                v_shared = T.alloc_shared((block_n, dim), dtype)
                o_shared = T.alloc_shared((block_m, dim), dtype)

                acc_s = T.alloc_fragment((block_m, block_n), acc_dtype)
                acc_s_cast = T.alloc_fragment((block_m, block_n), dtype)
                acc_o = T.alloc_fragment((block_m, dim), acc_dtype)
                scores_max = T.alloc_fragment((block_m,), acc_dtype)
                scores_max_prev = T.alloc_fragment((block_m,), acc_dtype)
                scores_scale = T.alloc_fragment((block_m,), acc_dtype)
                scores_sum = T.alloc_fragment((block_m,), acc_dtype)
                logsum = T.alloc_fragment((block_m,), acc_dtype)

                T.copy(q[bz, bx * block_m : (bx + 1) * block_m, by, :], q_shared, disable_tma=True)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(acc_dtype))

                loop_range = (
                    T.min(T.ceildiv(kv_len, block_n), T.ceildiv((bx + 1) * block_m, block_n))
                    if is_causal
                    else T.ceildiv(kv_len, block_n)
                )

                for block_k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(k[bz, block_k * block_n : (block_k + 1) * block_n, by, :], k_shared, disable_tma=True)

                    if is_causal:
                        for i, j in T.Parallel(block_m, block_n):
                            query_index = bx * block_m + i
                            key_index = block_k * block_n + j
                            acc_s[i, j] = T.if_then_else(
                                (query_index < query_len) & (key_index < key_lengths[bz]) & (query_index >= key_index),
                                0,
                                -T.infinity(acc_s.dtype),
                            )
                    else:
                        for i, j in T.Parallel(block_m, block_n):
                            key_index = block_k * block_n + j
                            acc_s[i, j] = T.if_then_else(
                                key_index < key_lengths[bz],
                                0,
                                -T.infinity(acc_s.dtype),
                            )

                    T.gemm(q_shared, k_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(acc_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)

                    for i in T.Parallel(block_m):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)

                    for i, j in T.Parallel(block_m, block_n):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)

                    T.reduce_sum(acc_s, scores_sum, dim=1)

                    for i in T.Parallel(block_m):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                    T.copy(acc_s, acc_s_cast)

                    for i, j in T.Parallel(block_m, dim):
                        acc_o[i, j] *= scores_scale[i]

                    T.copy(v[bz, block_k * block_n : (block_k + 1) * block_n, by, :], v_shared, disable_tma=True)
                    T.gemm(acc_s_cast, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(block_m, dim):
                    query_index = bx * block_m + i
                    acc_o[i, j] = T.if_then_else(
                        (query_index < query_len) & (logsum[i] > 0),
                        acc_o[i, j] / logsum[i],
                        0,
                    )

                T.copy(acc_o, o_shared)
                T.copy(o_shared, out[bz, bx * block_m : (bx + 1) * block_m, by, :], disable_tma=True)

        return main

    return flash_attention_forward_kernel(batch, heads, query_len, kv_len, dim)


def build_rms_norm_forward_kernel(rows: int, cols: int, block_rows: int, eps: float, tl_dtype, accum_dtype):
    @tl.jit(out_idx=[2, 3], target="cuda")
    def rms_norm_forward_kernel(
        rows: int,
        cols: int,
        block_rows: int,
        eps: float,
        dtype: T.dtype = tl_dtype,
        acc_dtype: T.dtype = accum_dtype,
    ):
        @T.prim_func
        def main(
            x: T.Tensor((rows, cols), dtype),
            weight: T.Tensor((cols,), dtype),
            out: T.Tensor((rows, cols), dtype),
            inv_rms: T.Tensor((rows,), acc_dtype),
        ):
            with T.Kernel(T.ceildiv(rows, block_rows), threads=128) as bx:
                x_shared = T.alloc_shared((block_rows, cols), dtype)
                weight_shared = T.alloc_shared((cols,), dtype)
                x_local = T.alloc_fragment((block_rows, cols), dtype)
                weight_local = T.alloc_fragment((cols,), dtype)
                x_square = T.alloc_fragment((block_rows, cols), acc_dtype)
                row_scale = T.alloc_fragment((block_rows,), acc_dtype)

                T.copy(x[bx * block_rows : (bx + 1) * block_rows, :], x_shared)
                T.copy(weight, weight_shared)
                T.copy(x_shared, x_local)
                T.copy(weight_shared, weight_local)

                for i, j in T.Parallel(block_rows, cols):
                    x_square[i, j] = x_local[i, j].astype(acc_dtype) * x_local[i, j].astype(acc_dtype)

                T.reduce_sum(x_square, row_scale, dim=1)

                for i in T.Parallel(block_rows):
                    row_scale[i] = T.rsqrt(row_scale[i] / cols + eps)

                for i, j in T.Parallel(block_rows, cols):
                    out[bx * block_rows + i, j] = (
                        x_local[i, j].astype(acc_dtype)
                        * row_scale[i]
                        * weight_local[j].astype(acc_dtype)
                    ).astype(dtype)
                T.copy(row_scale, inv_rms[bx * block_rows : (bx + 1) * block_rows])

        return main

    return rms_norm_forward_kernel(rows, cols, block_rows, eps)


def build_rms_norm_backward_kernel(rows: int, cols: int, block_rows: int, tl_dtype, accum_dtype):
    @tl.jit(out_idx=[4, 5], target="cuda")
    def rms_norm_backward_kernel(
        rows: int,
        cols: int,
        block_rows: int,
        dtype: T.dtype = tl_dtype,
        acc_dtype: T.dtype = accum_dtype,
    ):
        @T.prim_func
        def main(
            x: T.Tensor((rows, cols), dtype),
            weight: T.Tensor((cols,), dtype),
            inv_rms: T.Tensor((rows,), acc_dtype),
            grad_out: T.Tensor((rows, cols), dtype),
            grad_x: T.Tensor((rows, cols), dtype),
            grad_weight_partial: T.Tensor((T.ceildiv(rows, block_rows), cols), acc_dtype),
        ):
            with T.Kernel(T.ceildiv(rows, block_rows), threads=128) as bx:
                x_shared = T.alloc_shared((block_rows, cols), dtype)
                grad_out_shared = T.alloc_shared((block_rows, cols), dtype)
                weight_shared = T.alloc_shared((cols,), dtype)
                inv_rms_shared = T.alloc_shared((block_rows,), acc_dtype)

                x_local = T.alloc_fragment((block_rows, cols), dtype)
                grad_out_local = T.alloc_fragment((block_rows, cols), dtype)
                weight_local = T.alloc_fragment((cols,), dtype)
                inv_rms_local = T.alloc_fragment((block_rows,), acc_dtype)

                weighted_grad = T.alloc_fragment((block_rows, cols), acc_dtype)
                grad_dot_input = T.alloc_fragment((block_rows, cols), acc_dtype)
                row_dot = T.alloc_fragment((block_rows,), acc_dtype)
                grad_x_local = T.alloc_fragment((block_rows, cols), acc_dtype)
                grad_weight_contrib = T.alloc_fragment((block_rows, cols), acc_dtype)
                grad_weight_block = T.alloc_fragment((cols,), acc_dtype)

                T.copy(x[bx * block_rows : (bx + 1) * block_rows, :], x_shared)
                T.copy(grad_out[bx * block_rows : (bx + 1) * block_rows, :], grad_out_shared)
                T.copy(weight, weight_shared)
                T.copy(inv_rms[bx * block_rows : (bx + 1) * block_rows], inv_rms_shared)

                T.copy(x_shared, x_local)
                T.copy(grad_out_shared, grad_out_local)
                T.copy(weight_shared, weight_local)
                T.copy(inv_rms_shared, inv_rms_local)

                for i, j in T.Parallel(block_rows, cols):
                    weighted_grad[i, j] = grad_out_local[i, j].astype(acc_dtype) * weight_local[j].astype(acc_dtype)
                    grad_dot_input[i, j] = weighted_grad[i, j] * x_local[i, j].astype(acc_dtype)

                T.reduce_sum(grad_dot_input, row_dot, dim=1)

                for i, j in T.Parallel(block_rows, cols):
                    inv_rms_value = inv_rms_local[i]
                    inv_rms_cubed = inv_rms_value * inv_rms_value * inv_rms_value
                    grad_x_local[i, j] = (
                        weighted_grad[i, j] * inv_rms_value
                        - x_local[i, j].astype(acc_dtype) * row_dot[i] * inv_rms_cubed / cols
                    )
                    grad_weight_contrib[i, j] = (
                        grad_out_local[i, j].astype(acc_dtype)
                        * x_local[i, j].astype(acc_dtype)
                        * inv_rms_value
                    )

                T.reduce_sum(grad_weight_contrib, grad_weight_block, dim=0)

                for i, j in T.Parallel(block_rows, cols):
                    grad_x[bx * block_rows + i, j] = grad_x_local[i, j].astype(dtype)
                T.copy(grad_weight_block, grad_weight_partial[bx, :])

        return main

    return rms_norm_backward_kernel(rows, cols, block_rows)


def build_rms_norm_kernel(rows: int, cols: int, block_rows: int, eps: float, tl_dtype, accum_dtype):
    compiled = build_rms_norm_forward_kernel(rows, cols, block_rows, eps, tl_dtype, accum_dtype)

    def runner(x, weight):
        out, _inv_rms = compiled(x, weight)
        return out

    return runner