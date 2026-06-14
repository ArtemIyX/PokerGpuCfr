from __future__ import annotations

try:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:

    @triton.jit  # type: ignore[untyped-decorator]
    def regret_matching_accum_kernel(  # type: ignore[no-untyped-def]
        regrets_ptr,
        out_ptr,
        action_infoset_ptr,
        action_slot_ptr,
        row_sums_ptr,
        num_actions,
        max_actions,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < num_actions
        infoset = tl.load(action_infoset_ptr + offs, mask=mask, other=0).to(tl.int32)
        slot = tl.load(action_slot_ptr + offs, mask=mask, other=0).to(tl.int32)
        regret = tl.load(regrets_ptr + offs, mask=mask, other=0.0)
        valid = mask & (infoset >= 0) & (slot >= 0) & (slot < max_actions)
        positive = tl.where(valid, tl.maximum(regret, 0.0), 0.0)
        tl.store(out_ptr + offs, positive, mask=mask)
        tl.atomic_add(row_sums_ptr + infoset, positive, mask=valid)

    @triton.jit  # type: ignore[untyped-decorator]
    def regret_matching_normalize_kernel(  # type: ignore[no-untyped-def]
        out_ptr,
        row_sums_ptr,
        action_counts_ptr,
        action_infoset_ptr,
        action_slot_ptr,
        num_actions,
        max_actions,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < num_actions
        infoset = tl.load(action_infoset_ptr + offs, mask=mask, other=0).to(tl.int32)
        slot = tl.load(action_slot_ptr + offs, mask=mask, other=0).to(tl.int32)
        valid = mask & (infoset >= 0) & (slot >= 0) & (slot < max_actions)
        counts = tl.load(action_counts_ptr + infoset, mask=valid, other=1).to(tl.float32)
        sums = tl.load(row_sums_ptr + infoset, mask=valid, other=0.0)
        values = tl.load(out_ptr + offs, mask=mask, other=0.0)
        normalized = tl.where(sums > 0.0, values / sums, tl.where(slot < counts, 1.0 / counts, 0.0))
        tl.store(out_ptr + offs, normalized, mask=mask)

    @triton.jit  # type: ignore[untyped-decorator]
    def forward_compact_kernel(  # type: ignore[no-untyped-def]
        edge_src_ptr,
        edge_dst_ptr,
        edge_prob_ptr,
        edge_flat_ptr,
        strategy_ptr,
        range0_ptr,
        range1_ptr,
        out0_ptr,
        out1_ptr,
        edge_count,
        range_stride,
        out_stride,
        hand_count,
        BLOCK: tl.constexpr,
    ):
        edge_pid = tl.program_id(0)
        hand_pid = tl.program_id(1)
        edge_offs = edge_pid * BLOCK + tl.arange(0, BLOCK)
        edge_mask = edge_offs < edge_count
        hand_mask = hand_pid < hand_count
        mask = edge_mask & hand_mask
        src = tl.load(edge_src_ptr + edge_offs, mask=edge_mask, other=0).to(tl.int32)
        dst = tl.load(edge_dst_ptr + edge_offs, mask=edge_mask, other=0).to(tl.int32)
        flat = tl.load(edge_flat_ptr + edge_offs, mask=edge_mask, other=0).to(tl.int32)
        prob = tl.load(edge_prob_ptr + edge_offs, mask=edge_mask, other=0.0)
        valid = mask & (flat >= 0)
        src = tl.where(valid, src, 0)
        dst = tl.where(valid, dst, 0)
        flat = tl.where(valid, flat, 0)
        prob = tl.where(valid, prob, 0.0)
        strat = tl.load(strategy_ptr + flat, mask=valid, other=0.0)
        src_off = src * range_stride + tl.full([BLOCK], hand_pid, tl.int32)
        dst_off = dst * out_stride + tl.full([BLOCK], hand_pid, tl.int32)
        range0 = tl.load(range0_ptr + src_off, mask=valid, other=0.0)
        range1 = tl.load(range1_ptr + src_off, mask=valid, other=0.0)
        tl.atomic_add(out0_ptr + dst_off, range0 * prob * strat, mask=valid)
        tl.atomic_add(out1_ptr + dst_off, range1 * prob * strat, mask=valid)


    @triton.jit  # type: ignore[untyped-decorator]
    def backward_compact_kernel(  # type: ignore[no-untyped-def]
        edge_src_ptr,
        edge_dst_ptr,
        edge_prob_ptr,
        edge_flat_ptr,
        strategy_ptr,
        out0_ptr,
        out1_ptr,
        node_value0_ptr,
        node_value1_ptr,
        edge_count,
        out_stride,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < edge_count
        src = tl.load(edge_src_ptr + offs, mask=mask, other=0).to(tl.int32)
        dst = tl.load(edge_dst_ptr + offs, mask=mask, other=0).to(tl.int32)
        flat = tl.load(edge_flat_ptr + offs, mask=mask, other=0).to(tl.int32)
        prob = tl.load(edge_prob_ptr + offs, mask=mask, other=0.0)
        valid = mask & (flat >= 0)
        child0 = tl.load(node_value0_ptr + dst, mask=valid, other=0.0)
        child1 = tl.load(node_value1_ptr + dst, mask=valid, other=0.0)
        strat = tl.load(strategy_ptr + flat, mask=valid, other=0.0)
        tl.atomic_add(out0_ptr + src, prob * strat * child0, mask=valid)
        tl.atomic_add(out1_ptr + src, prob * strat * child1, mask=valid)
