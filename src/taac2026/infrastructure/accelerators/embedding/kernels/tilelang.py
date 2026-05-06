# pyright: reportInvalidTypeForm=false

"""TileLang embedding-bag kernel builders."""

from taac2026.infrastructure.accelerators.tilelang_runtime import T, tl


def _embedding_bag_mean_pass_configs() -> dict[object, object]:
    if tl is None:
        raise RuntimeError("tilelang is not installed")
    return {
        tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }


def build_embedding_bag_mean_forward_kernel(
    batch: int,
    bag_size: int,
    num_embeddings: int,
    emb_dim: int,
    *,
    block_rows: int,
    block_cols: int,
    tl_dtype,
    accum_dtype,
):
    if tl is None:
        raise RuntimeError("tilelang is not installed")

    @tl.jit(out_idx=[2], target="cuda", pass_configs=_embedding_bag_mean_pass_configs())
    def embedding_bag_mean_forward_kernel(
        batch: int,
        bag_size: int,
        num_embeddings: int,
        emb_dim: int,
        block_rows: int,
        block_cols: int,
        dtype: T.dtype = tl_dtype,
        acc_dtype: T.dtype = accum_dtype,
    ):
        @T.prim_func
        def main(
            weight: T.Tensor((num_embeddings, emb_dim), dtype),
            values: T.Tensor((batch, bag_size), "int32"),
            out: T.Tensor((batch, emb_dim), dtype),
        ):
            with T.Kernel(T.ceildiv(batch, block_rows), T.ceildiv(emb_dim, block_cols), threads=128) as (bx, by):
                token_ids = T.alloc_fragment((block_rows,), "int32")
                valid_counts = T.alloc_fragment((block_rows,), acc_dtype)
                accum = T.alloc_fragment((block_rows, block_cols), acc_dtype)

                T.clear(valid_counts)
                T.clear(accum)

                for position in T.serial(bag_size):
                    for i in T.Parallel(block_rows):
                        row = bx * block_rows + i
                        token_ids[i] = 0
                        if row < batch:
                            token_ids[i] = values[row, position]
                            if (token_ids[i] > 0) & (token_ids[i] < num_embeddings):
                                valid_counts[i] += 1.0

                    for i, j in T.Parallel(block_rows, block_cols):
                        col = by * block_cols + j
                        if (col < emb_dim) & (token_ids[i] > 0) & (token_ids[i] < num_embeddings):
                            accum[i, j] += weight[token_ids[i], col].astype(acc_dtype)

                for i, j in T.Parallel(block_rows, block_cols):
                    row = bx * block_rows + i
                    col = by * block_cols + j
                    if (row < batch) & (col < emb_dim):
                        denominator = T.max(valid_counts[i], 1.0)
                        out[row, col] = (accum[i, j] / denominator).astype(dtype)

        return main

    return embedding_bag_mean_forward_kernel(batch, bag_size, num_embeddings, emb_dim, block_rows, block_cols)


def build_embedding_bag_mean_backward_kernel(
    batch: int,
    bag_size: int,
    num_embeddings: int,
    emb_dim: int,
    *,
    block_rows: int,
    block_cols: int,
    tl_dtype,
    accum_dtype,
):
    if tl is None:
        raise RuntimeError("tilelang is not installed")

    @tl.jit(target="cuda", pass_configs=_embedding_bag_mean_pass_configs())
    def embedding_bag_mean_backward_kernel(
        batch: int,
        bag_size: int,
        num_embeddings: int,
        emb_dim: int,
        block_rows: int,
        block_cols: int,
        dtype: T.dtype = tl_dtype,
        acc_dtype: T.dtype = accum_dtype,
    ):
        @T.prim_func
        def main(
            values: T.Tensor((batch, bag_size), "int32"),
            grad_out: T.Tensor((batch, emb_dim), dtype),
            grad_weight: T.Tensor((num_embeddings, emb_dim), acc_dtype),
        ):
            with T.Kernel(T.ceildiv(batch, block_rows), T.ceildiv(emb_dim, block_cols), threads=128) as (bx, by):
                token_ids = T.alloc_fragment((block_rows,), "int32")
                valid_counts = T.alloc_fragment((block_rows,), acc_dtype)

                T.clear(valid_counts)

                for position in T.serial(bag_size):
                    for i in T.Parallel(block_rows):
                        row = bx * block_rows + i
                        token_ids[i] = 0
                        if row < batch:
                            token_ids[i] = values[row, position]
                            if (token_ids[i] > 0) & (token_ids[i] < num_embeddings):
                                valid_counts[i] += 1.0

                for position in T.serial(bag_size):
                    for i in T.Parallel(block_rows):
                        row = bx * block_rows + i
                        token_ids[i] = 0
                        if row < batch:
                            token_ids[i] = values[row, position]

                    for i, j in T.Parallel(block_rows, block_cols):
                        row = bx * block_rows + i
                        col = by * block_cols + j
                        if (
                            (row < batch)
                            & (col < emb_dim)
                            & (token_ids[i] > 0)
                            & (token_ids[i] < num_embeddings)
                            & (valid_counts[i] > 0)
                        ):
                            T.atomic_add(
                                grad_weight[token_ids[i], col],
                                grad_out[row, col].astype(acc_dtype) / valid_counts[i],
                            )

        return main

    return embedding_bag_mean_backward_kernel(batch, bag_size, num_embeddings, emb_dim, block_rows, block_cols)


__all__ = [
    "build_embedding_bag_mean_backward_kernel",
    "build_embedding_bag_mean_forward_kernel",
]
