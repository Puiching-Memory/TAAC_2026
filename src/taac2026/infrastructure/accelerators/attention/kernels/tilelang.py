# pyright: reportInvalidTypeForm=false

"""TileLang flash-attention kernel builders."""

from taac2026.infrastructure.accelerators.tilelang_runtime import T, tl


def _flash_attention_pass_configs(*, block_n: int, num_stages: int) -> dict[object, object]:
    if tl is None:
        raise RuntimeError("tilelang is not installed")
    del block_n, num_stages
    return {
        tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }


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
    pass_configs = _flash_attention_pass_configs(block_n=block_n, num_stages=num_stages)

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


def build_flash_attention_training_forward_kernel(
    batch: int,
    heads: int,
    query_len: int,
    kv_len: int,
    dim: int,
    *,
    is_causal: bool,
    use_dropout: bool,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
    tl_dtype,
    accum_dtype,
):
    sm_scale = (1.0 / dim) ** 0.5
    scale = sm_scale * 1.44269504  # log2(e)
    pass_configs = _flash_attention_pass_configs(block_n=block_n, num_stages=num_stages)

    if use_dropout:

        @tl.jit(out_idx=[6, 7], target="cuda", pass_configs=pass_configs)
        def flash_attention_training_forward_kernel(
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
                q: T.Tensor((batch, heads, query_len, dim), dtype),
                k: T.Tensor((batch, heads, kv_len, dim), dtype),
                v: T.Tensor((batch, heads, kv_len, dim), dtype),
                key_lengths: T.Tensor((batch,), "int32"),
                dropout_mask: T.Tensor((batch, heads, query_len, kv_len), "uint8"),
                dropout_scale: acc_dtype,
                out: T.Tensor((batch, heads, query_len, dim), dtype),
                lse: T.Tensor((batch, heads, query_len), acc_dtype),
            ):
                with T.Kernel(T.ceildiv(query_len, block_m), heads, batch, threads=threads) as (bx, by, bz):
                    q_shared = T.alloc_shared((block_m, dim), dtype)
                    k_shared = T.alloc_shared((block_n, dim), dtype)
                    v_shared = T.alloc_shared((block_n, dim), dtype)

                    acc_s = T.alloc_fragment((block_m, block_n), acc_dtype)
                    acc_s_cast = T.alloc_fragment((block_m, block_n), dtype)
                    acc_o = T.alloc_fragment((block_m, dim), acc_dtype)
                    scores_max = T.alloc_fragment((block_m,), acc_dtype)
                    scores_max_prev = T.alloc_fragment((block_m,), acc_dtype)
                    scores_scale = T.alloc_fragment((block_m,), acc_dtype)
                    scores_sum = T.alloc_fragment((block_m,), acc_dtype)
                    logsum = T.alloc_fragment((block_m,), acc_dtype)

                    inv_keep_prob = T.alloc_var(acc_dtype)
                    inv_keep_prob = dropout_scale

                    T.copy(q[bz, by, bx * block_m : (bx + 1) * block_m, :], q_shared)
                    T.fill(acc_o, 0)
                    T.fill(logsum, 0)
                    T.fill(scores_max, -T.infinity(acc_dtype))

                    loop_range = (
                        T.min(T.ceildiv(kv_len, block_n), T.ceildiv((bx + 1) * block_m, block_n))
                        if is_causal
                        else T.ceildiv(kv_len, block_n)
                    )

                    for block_k in T.Pipelined(loop_range, num_stages=num_stages):
                        T.copy(k[bz, by, block_k * block_n : (block_k + 1) * block_n, :], k_shared)
                        T.clear(acc_s)
                        T.gemm(q_shared, k_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                        for i, j in T.Parallel(block_m, block_n):
                            query_index = bx * block_m + i
                            key_index = block_k * block_n + j
                            acc_s[i, j] = T.if_then_else(
                                (query_index < query_len)
                                & (key_index < key_lengths[bz])
                                & (T.if_then_else(is_causal, query_index >= key_index, True)),
                                acc_s[i, j],
                                -T.infinity(acc_dtype),
                            )

                        T.copy(scores_max, scores_max_prev)
                        T.fill(scores_max, -T.infinity(acc_dtype))
                        T.reduce_max(acc_s, scores_max, dim=1, clear=False)

                        for i in T.Parallel(block_m):
                            scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                            logsum[i] *= scores_scale[i]

                        for i, j in T.Parallel(block_m, dim):
                            acc_o[i, j] *= scores_scale[i]

                        for i, j in T.Parallel(block_m, block_n):
                            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)

                        T.reduce_sum(acc_s, scores_sum, dim=1)

                        for i in T.Parallel(block_m):
                            logsum[i] += scores_sum[i]

                        for i, j in T.Parallel(block_m, block_n):
                            query_index = bx * block_m + i
                            key_index = block_k * block_n + j
                            acc_s_cast[i, j] = T.if_then_else(
                                (query_index < query_len)
                                & (key_index < key_lengths[bz])
                                & (dropout_mask[bz, by, query_index, key_index] != 0),
                                (acc_s[i, j] * inv_keep_prob).astype(dtype),
                                T.cast(0, dtype),
                            )

                        T.copy(v[bz, by, block_k * block_n : (block_k + 1) * block_n, :], v_shared)
                        T.gemm(acc_s_cast, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                    for i, j in T.Parallel(block_m, dim):
                        query_index = bx * block_m + i
                        acc_o[i, j] = T.if_then_else(
                            (query_index < query_len) & (logsum[i] > 0),
                            acc_o[i, j] / logsum[i],
                            0,
                        )
                        if query_index < query_len:
                            out[bz, by, query_index, j] = acc_o[i, j].astype(dtype)

                    for i in T.Parallel(block_m):
                        query_index = bx * block_m + i
                        if query_index < query_len:
                            lse[bz, by, query_index] = T.if_then_else(
                                logsum[i] > 0,
                                T.log2(logsum[i]) + scores_max[i] * scale,
                                -T.infinity(acc_dtype),
                            )

            return main

        return flash_attention_training_forward_kernel(batch, heads, query_len, kv_len, dim)

    @tl.jit(out_idx=[4, 5], target="cuda", pass_configs=pass_configs)
    def flash_attention_training_forward_kernel(
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
            q: T.Tensor((batch, heads, query_len, dim), dtype),
            k: T.Tensor((batch, heads, kv_len, dim), dtype),
            v: T.Tensor((batch, heads, kv_len, dim), dtype),
            key_lengths: T.Tensor((batch,), "int32"),
            out: T.Tensor((batch, heads, query_len, dim), dtype),
            lse: T.Tensor((batch, heads, query_len), acc_dtype),
        ):
            with T.Kernel(T.ceildiv(query_len, block_m), heads, batch, threads=threads) as (bx, by, bz):
                q_shared = T.alloc_shared((block_m, dim), dtype)
                k_shared = T.alloc_shared((block_n, dim), dtype)
                v_shared = T.alloc_shared((block_n, dim), dtype)

                acc_s = T.alloc_fragment((block_m, block_n), acc_dtype)
                acc_s_cast = T.alloc_fragment((block_m, block_n), dtype)
                acc_o = T.alloc_fragment((block_m, dim), acc_dtype)
                scores_max = T.alloc_fragment((block_m,), acc_dtype)
                scores_max_prev = T.alloc_fragment((block_m,), acc_dtype)
                scores_scale = T.alloc_fragment((block_m,), acc_dtype)
                scores_sum = T.alloc_fragment((block_m,), acc_dtype)
                logsum = T.alloc_fragment((block_m,), acc_dtype)

                T.copy(q[bz, by, bx * block_m : (bx + 1) * block_m, :], q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(acc_dtype))

                loop_range = (
                    T.min(T.ceildiv(kv_len, block_n), T.ceildiv((bx + 1) * block_m, block_n))
                    if is_causal
                    else T.ceildiv(kv_len, block_n)
                )

                for block_k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(k[bz, by, block_k * block_n : (block_k + 1) * block_n, :], k_shared)
                    T.clear(acc_s)
                    T.gemm(q_shared, k_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    for i, j in T.Parallel(block_m, block_n):
                        query_index = bx * block_m + i
                        key_index = block_k * block_n + j
                        acc_s[i, j] = T.if_then_else(
                            (query_index < query_len)
                            & (key_index < key_lengths[bz])
                            & (T.if_then_else(is_causal, query_index >= key_index, True)),
                            acc_s[i, j],
                            -T.infinity(acc_dtype),
                        )

                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(acc_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)

                    for i in T.Parallel(block_m):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                        logsum[i] *= scores_scale[i]

                    for i, j in T.Parallel(block_m, dim):
                        acc_o[i, j] *= scores_scale[i]

                    for i, j in T.Parallel(block_m, block_n):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)

                    T.reduce_sum(acc_s, scores_sum, dim=1)

                    for i in T.Parallel(block_m):
                        logsum[i] += scores_sum[i]

                    T.copy(acc_s, acc_s_cast)
                    T.copy(v[bz, by, block_k * block_n : (block_k + 1) * block_n, :], v_shared)
                    T.gemm(acc_s_cast, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(block_m, dim):
                    query_index = bx * block_m + i
                    acc_o[i, j] = T.if_then_else(
                        (query_index < query_len) & (logsum[i] > 0),
                        acc_o[i, j] / logsum[i],
                        0,
                    )
                    if query_index < query_len:
                        out[bz, by, query_index, j] = acc_o[i, j].astype(dtype)

                for i in T.Parallel(block_m):
                    query_index = bx * block_m + i
                    if query_index < query_len:
                        lse[bz, by, query_index] = T.if_then_else(
                            logsum[i] > 0,
                            T.log2(logsum[i]) + scores_max[i] * scale,
                            -T.infinity(acc_dtype),
                        )

        return main

    return flash_attention_training_forward_kernel(batch, heads, query_len, kv_len, dim)


def build_flash_attention_backward_preprocess_kernel(
    batch: int,
    heads: int,
    query_len: int,
    dim: int,
    tl_dtype,
    accum_dtype,
):
    blk = 32

    @tl.jit(out_idx=[2], target="cuda")
    def flash_attention_backward_preprocess_kernel(
        batch: int,
        heads: int,
        query_len: int,
        dim: int,
        dtype: T.dtype = tl_dtype,
        acc_dtype: T.dtype = accum_dtype,
    ):
        @T.prim_func
        def main(
            out: T.Tensor((batch, heads, query_len, dim), dtype),
            grad_out: T.Tensor((batch, heads, query_len, dim), dtype),
            delta: T.Tensor((batch, heads, query_len), acc_dtype),
        ):
            with T.Kernel(heads, T.ceildiv(query_len, blk), batch) as (bx, by, bz):
                out_fragment = T.alloc_fragment((blk, blk), dtype)
                grad_fragment = T.alloc_fragment((blk, blk), dtype)
                acc = T.alloc_fragment((blk, blk), acc_dtype)
                delta_fragment = T.alloc_fragment((blk,), acc_dtype)

                T.clear(acc)
                for k in range(T.ceildiv(dim, blk)):
                    T.copy(out[bz, bx, by * blk : (by + 1) * blk, k * blk : (k + 1) * blk], out_fragment)
                    T.copy(grad_out[bz, bx, by * blk : (by + 1) * blk, k * blk : (k + 1) * blk], grad_fragment)
                    for i, j in T.Parallel(blk, blk):
                        acc[i, j] += out_fragment[i, j].astype(acc_dtype) * grad_fragment[i, j].astype(acc_dtype)
                T.reduce_sum(acc, delta_fragment, dim=1)
                T.copy(delta_fragment, delta[bz, bx, by * blk : (by + 1) * blk])

        return main

    return flash_attention_backward_preprocess_kernel(batch, heads, query_len, dim)


def build_flash_attention_backward_kernel(
    batch: int,
    heads: int,
    query_len: int,
    kv_len: int,
    dim: int,
    *,
    is_causal: bool,
    use_dropout: bool,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
    tl_dtype,
    accum_dtype,
):
    sm_scale = (1.0 / dim) ** 0.5
    scale = sm_scale * 1.44269504  # log2(e)
    pass_configs = _flash_attention_pass_configs(block_n=block_n, num_stages=num_stages)

    if use_dropout:

        @tl.jit(target="cuda", pass_configs=pass_configs)
        def flash_attention_backward_kernel(
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
                q: T.Tensor((batch, heads, query_len, dim), dtype),
                k: T.Tensor((batch, heads, kv_len, dim), dtype),
                v: T.Tensor((batch, heads, kv_len, dim), dtype),
                grad_out: T.Tensor((batch, heads, query_len, dim), dtype),
                lse: T.Tensor((batch, heads, query_len), acc_dtype),
                delta: T.Tensor((batch, heads, query_len), acc_dtype),
                key_lengths: T.Tensor((batch,), "int32"),
                dropout_mask: T.Tensor((batch, heads, query_len, kv_len), "uint8"),
                dropout_scale: acc_dtype,
                grad_q: T.Tensor((batch, heads, query_len, dim), acc_dtype),
                grad_k: T.Tensor((batch, heads, kv_len, dim), acc_dtype),
                grad_v: T.Tensor((batch, heads, kv_len, dim), acc_dtype),
            ):
                with T.Kernel(heads, T.ceildiv(kv_len, block_m), batch, threads=threads) as (bx, by, bz):
                    k_shared = T.alloc_shared((block_m, dim), dtype)
                    v_shared = T.alloc_shared((block_m, dim), dtype)
                    q_shared = T.alloc_shared((block_n, dim), dtype)
                    grad_out_shared = T.alloc_shared((block_n, dim), dtype)
                    lse_shared = T.alloc_shared((block_n,), acc_dtype)
                    delta_shared = T.alloc_shared((block_n,), acc_dtype)
                    dscore_shared = T.alloc_shared((block_m, block_n), dtype)
                    grad_k_shared = T.alloc_shared((block_m, dim), acc_dtype)
                    grad_v_shared = T.alloc_shared((block_m, dim), acc_dtype)

                    qk = T.alloc_fragment((block_m, block_n), acc_dtype)
                    prob = T.alloc_fragment((block_m, block_n), acc_dtype)
                    grad_prob = T.alloc_fragment((block_m, block_n), acc_dtype)
                    prob_cast = T.alloc_fragment((block_m, block_n), dtype)
                    dscore_cast = T.alloc_fragment((block_m, block_n), dtype)
                    grad_v_fragment = T.alloc_fragment((block_m, dim), acc_dtype)
                    grad_k_fragment = T.alloc_fragment((block_m, dim), acc_dtype)
                    grad_q_fragment = T.alloc_fragment((block_n, dim), acc_dtype)

                    kv_offset = by * block_m
                    T.copy(k[bz, bx, kv_offset : (by + 1) * block_m, :], k_shared)
                    T.copy(v[bz, bx, kv_offset : (by + 1) * block_m, :], v_shared)
                    T.clear(grad_v_fragment)
                    T.clear(grad_k_fragment)

                    loop_start = T.floordiv(kv_offset, block_n) if is_causal else 0
                    loop_end = T.ceildiv(query_len, block_n)

                    for block_q in T.Pipelined(loop_end - loop_start, num_stages=num_stages):
                        q_block = loop_start + block_q
                        q_offset = q_block * block_n

                        T.copy(q[bz, bx, q_offset : (q_block + 1) * block_n, :], q_shared)
                        T.copy(grad_out[bz, bx, q_offset : (q_block + 1) * block_n, :], grad_out_shared)
                        T.copy(lse[bz, bx, q_offset : (q_block + 1) * block_n], lse_shared)
                        T.copy(delta[bz, bx, q_offset : (q_block + 1) * block_n], delta_shared)

                        T.clear(qk)
                        T.gemm(k_shared, q_shared, qk, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                        for i, j in T.Parallel(block_m, block_n):
                            key_index = kv_offset + i
                            query_index = q_offset + j
                            prob[i, j] = T.if_then_else(
                                (query_index < query_len)
                                & (key_index < key_lengths[bz])
                                & (T.if_then_else(is_causal, query_index >= key_index, True)),
                                T.exp2(qk[i, j] * scale - lse_shared[j]),
                                0,
                            )

                        T.clear(grad_prob)
                        T.gemm(v_shared, grad_out_shared, grad_prob, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                        for i, j in T.Parallel(block_m, block_n):
                            key_index = kv_offset + i
                            query_index = q_offset + j
                            prob_cast[i, j] = T.if_then_else(
                                (query_index < query_len)
                                & (key_index < key_lengths[bz])
                                & (dropout_mask[bz, bx, query_index, key_index] != 0),
                                (prob[i, j] * dropout_scale).astype(dtype),
                                T.cast(0, dtype),
                            )

                        T.gemm(prob_cast, grad_out_shared, grad_v_fragment, policy=T.GemmWarpPolicy.FullRow)

                        for i, j in T.Parallel(block_m, block_n):
                            key_index = kv_offset + i
                            query_index = q_offset + j
                            dscore_cast[i, j] = T.if_then_else(
                                (query_index < query_len)
                                & (key_index < key_lengths[bz]),
                                (
                                    prob[i, j]
                                    * (
                                        T.if_then_else(
                                            dropout_mask[bz, bx, query_index, key_index] != 0,
                                            grad_prob[i, j] * dropout_scale,
                                            0,
                                        )
                                        - delta_shared[j]
                                    )
                                    * sm_scale
                                ).astype(dtype),
                                T.cast(0, dtype),
                            )

                        T.gemm(dscore_cast, q_shared, grad_k_fragment, policy=T.GemmWarpPolicy.FullRow)
                        T.copy(dscore_cast, dscore_shared)
                        T.clear(grad_q_fragment)
                        T.gemm(dscore_shared, k_shared, grad_q_fragment, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

                        for i, j in T.Parallel(block_n, dim):
                            query_index = q_offset + i
                            if query_index < query_len:
                                T.atomic_add(grad_q[bz, bx, query_index, j], grad_q_fragment[i, j])

                    T.copy(grad_k_fragment, grad_k_shared)
                    T.copy(grad_v_fragment, grad_v_shared)
                    for i, j in T.Parallel(block_m, dim):
                        key_index = kv_offset + i
                        if key_index < kv_len:
                            grad_k[bz, bx, key_index, j] = grad_k_shared[i, j]
                            grad_v[bz, bx, key_index, j] = grad_v_shared[i, j]

            return main

        return flash_attention_backward_kernel(batch, heads, query_len, kv_len, dim)

    @tl.jit(target="cuda", pass_configs=pass_configs)
    def flash_attention_backward_kernel(
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
            q: T.Tensor((batch, heads, query_len, dim), dtype),
            k: T.Tensor((batch, heads, kv_len, dim), dtype),
            v: T.Tensor((batch, heads, kv_len, dim), dtype),
            grad_out: T.Tensor((batch, heads, query_len, dim), dtype),
            lse: T.Tensor((batch, heads, query_len), acc_dtype),
            delta: T.Tensor((batch, heads, query_len), acc_dtype),
            key_lengths: T.Tensor((batch,), "int32"),
            grad_q: T.Tensor((batch, heads, query_len, dim), acc_dtype),
            grad_k: T.Tensor((batch, heads, kv_len, dim), acc_dtype),
            grad_v: T.Tensor((batch, heads, kv_len, dim), acc_dtype),
        ):
            with T.Kernel(heads, T.ceildiv(kv_len, block_m), batch, threads=threads) as (bx, by, bz):
                k_shared = T.alloc_shared((block_m, dim), dtype)
                v_shared = T.alloc_shared((block_m, dim), dtype)
                q_shared = T.alloc_shared((block_n, dim), dtype)
                grad_out_shared = T.alloc_shared((block_n, dim), dtype)
                lse_shared = T.alloc_shared((block_n,), acc_dtype)
                delta_shared = T.alloc_shared((block_n,), acc_dtype)
                dscore_shared = T.alloc_shared((block_m, block_n), dtype)
                grad_k_shared = T.alloc_shared((block_m, dim), acc_dtype)
                grad_v_shared = T.alloc_shared((block_m, dim), acc_dtype)

                qk = T.alloc_fragment((block_m, block_n), acc_dtype)
                prob = T.alloc_fragment((block_m, block_n), acc_dtype)
                grad_prob = T.alloc_fragment((block_m, block_n), acc_dtype)
                prob_cast = T.alloc_fragment((block_m, block_n), dtype)
                dscore_cast = T.alloc_fragment((block_m, block_n), dtype)
                grad_v_fragment = T.alloc_fragment((block_m, dim), acc_dtype)
                grad_k_fragment = T.alloc_fragment((block_m, dim), acc_dtype)
                grad_q_fragment = T.alloc_fragment((block_n, dim), acc_dtype)

                kv_offset = by * block_m
                T.copy(k[bz, bx, kv_offset : (by + 1) * block_m, :], k_shared)
                T.copy(v[bz, bx, kv_offset : (by + 1) * block_m, :], v_shared)
                T.clear(grad_v_fragment)
                T.clear(grad_k_fragment)

                loop_start = T.floordiv(kv_offset, block_n) if is_causal else 0
                loop_end = T.ceildiv(query_len, block_n)

                for block_q in T.Pipelined(loop_end - loop_start, num_stages=num_stages):
                    q_block = loop_start + block_q
                    q_offset = q_block * block_n

                    T.copy(q[bz, bx, q_offset : (q_block + 1) * block_n, :], q_shared)
                    T.copy(grad_out[bz, bx, q_offset : (q_block + 1) * block_n, :], grad_out_shared)
                    T.copy(lse[bz, bx, q_offset : (q_block + 1) * block_n], lse_shared)
                    T.copy(delta[bz, bx, q_offset : (q_block + 1) * block_n], delta_shared)

                    T.clear(qk)
                    T.gemm(k_shared, q_shared, qk, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    for i, j in T.Parallel(block_m, block_n):
                        key_index = kv_offset + i
                        query_index = q_offset + j
                        prob[i, j] = T.if_then_else(
                            (query_index < query_len)
                            & (key_index < key_lengths[bz])
                            & (T.if_then_else(is_causal, query_index >= key_index, True)),
                            T.exp2(qk[i, j] * scale - lse_shared[j]),
                            0,
                        )

                    T.clear(grad_prob)
                    T.gemm(v_shared, grad_out_shared, grad_prob, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(prob, prob_cast)
                    T.gemm(prob_cast, grad_out_shared, grad_v_fragment, policy=T.GemmWarpPolicy.FullRow)

                    for i, j in T.Parallel(block_m, block_n):
                        dscore_cast[i, j] = (prob[i, j] * (grad_prob[i, j] - delta_shared[j]) * sm_scale).astype(dtype)

                    T.gemm(dscore_cast, q_shared, grad_k_fragment, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(dscore_cast, dscore_shared)
                    T.clear(grad_q_fragment)
                    T.gemm(dscore_shared, k_shared, grad_q_fragment, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

                    for i, j in T.Parallel(block_n, dim):
                        query_index = q_offset + i
                        if query_index < query_len:
                            T.atomic_add(grad_q[bz, bx, query_index, j], grad_q_fragment[i, j])

                T.copy(grad_k_fragment, grad_k_shared)
                T.copy(grad_v_fragment, grad_v_shared)
                for i, j in T.Parallel(block_m, dim):
                    key_index = kv_offset + i
                    if key_index < kv_len:
                        grad_k[bz, bx, key_index, j] = grad_k_shared[i, j]
                        grad_v[bz, bx, key_index, j] = grad_v_shared[i, j]

        return main

    return flash_attention_backward_kernel(batch, heads, query_len, kv_len, dim)


__all__ = [
    "build_flash_attention_backward_kernel",
    "build_flash_attention_backward_preprocess_kernel",
    "build_flash_attention_forward_kernel",
    "build_flash_attention_training_forward_kernel",
]
