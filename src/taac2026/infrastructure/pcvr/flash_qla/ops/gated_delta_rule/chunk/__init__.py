import torch
import tilelang

from ....utils import l2norm
from ...utils import chunk_local_cumsum, group_reduce_vector


def flash_qla_target_compute_version() -> str | None:
    try:
        return str(tilelang.contrib.nvcc.get_target_compute_version())
    except Exception:
        return None


def flash_qla_device_compute_version(device: torch.device | None = None) -> str | None:
    if not torch.cuda.is_available():
        return None
    resolved_device = device
    if resolved_device is None:
        resolved_device = torch.device("cuda", torch.cuda.current_device())
    if resolved_device.type != "cuda":
        return None
    try:
        major, minor = torch.cuda.get_device_capability(resolved_device)
    except Exception:
        return None
    return f"{major}.{minor}"


def flash_qla_available(device: torch.device | None = None) -> bool:
    return flash_qla_target_compute_version() == "9.0" and flash_qla_device_compute_version(device) == "9.0"


def _require_runtime_support(*tensors: torch.Tensor | None) -> None:
    concrete_tensors = tuple(tensor for tensor in tensors if tensor is not None)
    if not concrete_tensors:
        return
    devices = {tensor.device for tensor in concrete_tensors}
    if len(devices) != 1:
        raise RuntimeError("flash_qla chunk_gated_delta_rule requires all tensors to live on the same device")
    device = concrete_tensors[0].device
    if device.type != "cuda":
        raise RuntimeError("flash_qla chunk_gated_delta_rule requires CUDA tensors")
    target_version = flash_qla_target_compute_version()
    device_version = flash_qla_device_compute_version(device)
    if target_version != "9.0" or device_version != "9.0":
        raise RuntimeError(
            "flash_qla chunk_gated_delta_rule currently requires Hopper SM90 "
            f"(device={device_version or 'unknown'}, target={target_version or 'unknown'})"
        )


def _load_hopper_kernels():
    from .hopper import fused_gdr_bwd, fused_gdr_fwd, fused_gdr_h, kkt_solve

    return fused_gdr_bwd, fused_gdr_fwd, fused_gdr_h, kkt_solve


def _load_cp_preprocess():
    from .cp_context import intra_card_cp_preprocess

    return intra_card_cp_preprocess


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    output_final_state: bool = True,
    output_h: bool = False,
    auto_cp: bool = True,
):
    _require_runtime_support(q, k, v, g, beta, initial_state, cu_seqlens)
    fused_gdr_bwd, fused_gdr_fwd, fused_gdr_h, kkt_solve = _load_hopper_kernels()
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)
    A = kkt_solve(
        k=k,
        b=beta,
        cu_seqlens=cu_seqlens,
    )
    cp_seq_map = None
    raw_cu_seqlens = None
    if auto_cp:
        intra_card_cp_preprocess = _load_cp_preprocess()
        initial_state, cu_seqlens, cp_seq_map, raw_cu_seqlens = (
            intra_card_cp_preprocess(
                k=k,
                v=v,
                a=A,
                g=g,
                b=beta,
                raw_h0=initial_state,
                raw_cu_seqlens=cu_seqlens,
            )
        )
    o, h, final_state = fused_gdr_fwd(
        q=q,
        k=k,
        v=v,
        a=A,
        g=g,
        b=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        output_h=output_h,
        output_o=True,
        cu_seqlens=cu_seqlens,
        cp_seq_map=cp_seq_map,
        raw_cu_seqlens=raw_cu_seqlens,
    )
    return g, A, o, h, final_state


def chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
):
    _require_runtime_support(q, k, v, g, beta, A, do, dht, initial_state, cu_seqlens)
    fused_gdr_bwd, fused_gdr_fwd, fused_gdr_h, kkt_solve = _load_hopper_kernels()
    h, _, _ = fused_gdr_h(
        k=k,
        v=v,
        a=A,
        g=g,
        b=beta,
        initial_state=initial_state,
        output_final_state=False,
        output_h=True,
        cu_seqlens=cu_seqlens,
    )
    dq, dk, dv, dg, db, dh0 = fused_gdr_bwd(
        q=q,
        k=k,
        v=v,
        a=A,
        g=g,
        b=beta,
        do=do,
        dht=dht,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    Hg, H = k.shape[-2], v.shape[-2]
    if Hg < H:
        dq = group_reduce_vector(dq, Hg)
        dk = group_reduce_vector(dk, Hg)
    assert dg.dtype == torch.float32, "dg should be fp32"
    dg = chunk_local_cumsum(dg, chunk_size=64, reverse=True, cu_seqlens=cu_seqlens)
    return dq, dk, dv, db, dg, dh0


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
    ):
        q_orig = q
        k_orig = k

        g, A, o, _, final_state = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            output_h=False,
            cu_seqlens=cu_seqlens,
        )

        ctx.save_for_backward(q_orig, k_orig, v, g, beta, A, initial_state, cu_seqlens)
        ctx.scale = scale
        return o.to(q.dtype), final_state

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor):
        q_orig, k_orig, v, g, beta, A, initial_state, cu_seqlens = ctx.saved_tensors

        dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd(
            q=q_orig,
            k=k_orig,
            v=v,
            g=g,
            beta=beta,
            A=A,
            do=do,
            dht=dht,
            scale=ctx.scale,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
        )

        return (
            dq.to(q_orig),
            dk.to(k_orig),
            dv.to(v),
            dg.to(g),
            db.to(beta),
            None,
            dh0,
            None,
            None,
            None,
        )


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    head_first: bool = False,
):
    _require_runtime_support(q, k, v, g, beta, initial_state, cu_seqlens)
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, (
        "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16 or float16."
    )
    assert not head_first, "head_first=True is not supported."
    assert v.shape[2] % k.shape[2] == 0, (
        "num_qk_heads must be divisible to num_v_heads."
    )

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )

    if scale is None:
        scale = k.shape[-1] ** -0.5

    if use_qk_l2norm_in_kernel:
        q = l2norm(q)
        k = l2norm(k)

    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
    )

    return o, final_state
