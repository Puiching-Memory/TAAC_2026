# pyright: reportInvalidTypeForm=false

"""TileLang kernel definitions for PCVR operators."""

import tilelang as tl
import tilelang.language as T


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