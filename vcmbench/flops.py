"""Rough MAC / FLOP estimate for an ONNX model (laptop-side analysis).

flops = 2 * macs + elementwise_ops + dft_flops. Conv/Gemm bias adds are not counted.
Dynamic or unknown dims are treated as 1. Data-movement ops cost nothing; ops with
no cost model are listed in ``notes``.
"""
from __future__ import annotations

import math
import os
from collections import Counter
from typing import Any, Callable

try:
    import onnx
    from onnx import helper, shape_inference
except ImportError:  # pragma: no cover - exercised only without onnx
    onnx = None  # type: ignore[assignment]

Shape = list[int]
Cost = tuple[int, int, int]  # (macs, elementwise ops, dft flops)

_ELEMENTWISE = {
    "Add", "Sub", "Mul", "Div", "Neg", "Abs", "Relu", "LeakyRelu", "PRelu", "Clip", "Sigmoid",
    "Tanh", "Exp", "Log", "Sqrt", "Pow", "Reciprocal", "Erf", "Gelu", "Elu", "Selu", "Celu",
    "HardSigmoid", "HardSwish", "Softplus", "Softsign", "Min", "Max", "Sum", "Mean", "Floor",
    "Ceil", "Round", "Sin", "Cos", "Where", "Equal", "Greater", "Less", "GreaterOrEqual",
    "LessOrEqual", "Not", "And", "Or", "Xor", "Sign", "Mod", "QuantizeLinear", "DequantizeLinear",
}
_NORM_COST = {  # ops per output element
    "BatchNormalization": 2, "Softmax": 5, "LogSoftmax": 5,
    "LayerNormalization": 5, "InstanceNormalization": 5,
}
_REDUCE = {"ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceL1", "ReduceL2"}
_FREE = {
    "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Flatten", "Concat", "Slice", "Gather",
    "GatherElements", "GatherND", "Shape", "Size", "Constant", "ConstantOfShape", "Identity",
    "Cast", "Pad", "Split", "Expand", "Tile", "Dropout", "Range", "ArgMax", "ArgMin", "TopK",
    "DepthToSpace", "SpaceToDepth", "NonZero", "ScatterND", "ScatterElements",
    "CastLike", "Trilu", "OneHot",
}
_UNKNOWN = object()


def _prod(xs: list[int] | tuple[int, ...]) -> int:
    p = 1
    for x in xs:
        p *= int(x)
    return p


def format_si(n: float) -> str:
    """Format a count with an SI suffix, e.g. 1234567 -> '1.23 M', 4.5e9 -> '4.5 G'."""
    sign = "-" if n < 0 else ""
    x = abs(float(n))
    units = ["", "K", "M", "G", "T"]
    idx = 0
    while idx < len(units) - 1 and float(f"{x:.3g}") >= 1000:
        x /= 1000
        idx += 1
    text = f"{x:.3g}"
    return f"{sign}{text} {units[idx]}".rstrip()


def _dims(vi: Any, flags: list[str]) -> Shape | None:
    t = vi.type
    if not t.HasField("tensor_type") or not t.tensor_type.HasField("shape"):
        return None
    out = []
    for d in t.tensor_type.shape.dim:
        if d.WhichOneof("value") == "dim_value" and d.dim_value > 0:
            out.append(int(d.dim_value))
        else:
            flags.append(vi.name)
            out.append(1)
    return out


def _fix_input_dims(graph: Any, init_names: set[str], notes: list[str]) -> None:
    """Replace dynamic/unknown input dims by 1 (in place) and record a note."""
    for vi in graph.input:
        if vi.name in init_names or not vi.type.HasField("tensor_type"):
            continue
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            notes.append(f"input '{vi.name}' has no shape; downstream costs may be missing")
            continue
        for k, d in enumerate(tt.shape.dim):
            if d.WhichOneof("value") != "dim_value" or d.dim_value <= 0:
                label = d.dim_param or "unknown"
                d.ClearField("dim_param")
                d.dim_value = 1
                notes.append(f"input '{vi.name}' dim {k} ({label}) dynamic, assumed 1")


def _cost_conv(node: Any, a: dict, shape: Callable[[str], Shape | None]) -> Cost | None:
    wi = 3 if node.op_type == "QLinearConv" else 1
    out, w = shape(node.output[0]), shape(node.input[wi]) if len(node.input) > wi else None
    if out is None:
        return None
    if w and len(w) >= 3:
        return _prod(out) * w[1] * _prod(w[2:]), 0, 0
    x, k = shape(node.input[0]), a.get("kernel_shape")
    if x and k and len(x) >= 2:
        return _prod(out) * (x[1] // int(a.get("group", 1))) * _prod(k), 0, 0
    return None


def _cost_conv_transpose(node: Any, a: dict, shape: Callable[[str], Shape | None]) -> Cost | None:
    x, w = shape(node.input[0]), shape(node.input[1])
    if not x or not w or len(w) < 3:
        return None
    return _prod(x) * w[1] * _prod(w[2:]), 0, 0  # each input elem hits Cout/g * kernel weights


def _cost_gemm(node: Any, a: dict, shape: Callable[[str], Shape | None]) -> Cost | None:
    x, out = shape(node.input[0]), shape(node.output[0])
    if not x or len(x) != 2 or out is None:
        return None
    k = x[0] if a.get("transA", 0) else x[1]
    return _prod(out) * k, 0, 0


def _cost_matmul(node: Any, a: dict, shape: Callable[[str], Shape | None]) -> Cost | None:
    x, out = shape(node.input[0]), shape(node.output[0])
    if not x or out is None:
        return None
    return _prod(out) * x[-1], 0, 0


def _cost_rnn(node: Any, a: dict, shape: Callable[[str], Shape | None]) -> Cost | None:
    x, w, r = shape(node.input[0]), shape(node.input[1]), shape(node.input[2])
    if not x or not w or not r or len(x) < 3 or len(w) != 3 or len(r) != 3:
        return None
    seq, batch = (x[1], x[0]) if a.get("layout", 0) == 1 else (x[0], x[1])
    # LSTM: D*4H*(I+H) per step per batch; GRU 3H; RNN H. Gate nonlinearities not counted.
    return seq * batch * (w[0] * w[1] * w[2] + r[0] * r[1] * r[2]), 0, 0


def _fft_flops(n: int, count: int) -> int:
    return int(count * 5 * n * math.log2(n)) if n >= 2 else 0


def _cost_stft(node: Any, a: dict, shape: Callable[[str], Shape | None]) -> Cost | None:
    out = shape(node.output[0])  # (batch, frames, bins, 2)
    if not out or len(out) != 4:
        return None
    win = shape(node.input[2]) if len(node.input) > 2 and node.input[2] else None
    if win:
        n = win[0]
    else:
        n = (out[2] - 1) * 2 if a.get("onesided", 1) else out[2]
    return 0, 0, _fft_flops(n, out[0] * out[1])


def _cost_dft(node: Any, a: dict, shape: Callable[[str], Shape | None]) -> Cost | None:
    out = shape(node.output[0])
    if not out or len(out) < 3:
        return None
    axis = int(a.get("axis", -2))
    n = out[axis]
    if a.get("onesided", 0):
        n = (n - 1) * 2
    return 0, 0, _fft_flops(n, _prod(out) // max(n * out[-1], 1))


def _node_cost(node: Any, a: dict, shape: Callable[[str], Shape | None]) -> Cost | None | object:
    """Cost of one node; None if shapes are missing, _UNKNOWN if no cost model exists."""
    op = node.op_type
    if node.domain not in ("", "ai.onnx"):
        return _UNKNOWN
    if op in ("Conv", "ConvInteger", "QLinearConv"):
        return _cost_conv(node, a, shape)
    if op == "ConvTranspose":
        return _cost_conv_transpose(node, a, shape)
    if op == "Gemm":
        return _cost_gemm(node, a, shape)
    if op in ("MatMul", "MatMulInteger", "QLinearMatMul"):
        return _cost_matmul(node, a, shape)
    if op in ("LSTM", "GRU", "RNN"):
        return _cost_rnn(node, a, shape)
    if op == "STFT":
        return _cost_stft(node, a, shape)
    if op == "DFT":
        return _cost_dft(node, a, shape)
    out = shape(node.output[0]) if node.output else None
    x = shape(node.input[0]) if node.input else None
    if op in _ELEMENTWISE:
        return (0, _prod(out), 0) if out is not None else None
    if op in _NORM_COST:
        return (0, _NORM_COST[op] * _prod(out), 0) if out is not None else None
    if op in ("MaxPool", "AveragePool", "LpPool"):
        k = a.get("kernel_shape")
        return (0, _prod(out) * _prod(k), 0) if out is not None and k else None
    if op in ("GlobalAveragePool", "GlobalMaxPool") or op in _REDUCE:
        return (0, _prod(x), 0) if x is not None else None
    return _UNKNOWN


def onnx_profile(path: str) -> dict[str, Any]:
    """Profile an ONNX file: params, size, opset, op counts, MACs and FLOPs estimate."""
    if onnx is None:
        raise ImportError("vcmbench.flops.onnx_profile needs the 'onnx' package (pip install onnx)")
    notes: list[str] = []
    model = onnx.load(path, load_external_data=False)
    graph = model.graph
    init_names = {t.name for t in graph.initializer}
    params = sum(_prod(list(t.dims)) for t in graph.initializer)
    opset = next((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), None)

    _fix_input_dims(graph, init_names, notes)
    del graph.value_info[:]
    try:
        model = shape_inference.infer_shapes(model)
        graph = model.graph
    except Exception as exc:  # noqa: BLE001 - inference is best effort
        notes.append(f"shape inference failed ({exc}); only declared shapes are used")

    flags: list[str] = []
    shapes: dict[str, Shape] = {t.name: list(t.dims) for t in graph.initializer}
    for vi in [*graph.input, *graph.value_info, *graph.output]:
        if vi.name in init_names:
            continue
        dims = _dims(vi, flags)
        if dims is not None:
            shapes[vi.name] = dims
    if flags:
        notes.append("unresolved dims in intermediate tensors assumed 1: " + ", ".join(sorted(set(flags))[:5]))
    input_shapes = {vi.name: shapes[vi.name] for vi in graph.input if vi.name not in init_names and vi.name in shapes}

    def shape(name: str) -> Shape | None:
        return shapes.get(name) if name else None

    counts: Counter[str] = Counter()
    unknown: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    macs = elem = dft = 0
    for node in graph.node:
        counts[node.op_type] += 1
        if node.op_type in _FREE and node.domain in ("", "ai.onnx"):
            continue
        attrs = {x.name: helper.get_attribute_value(x) for x in node.attribute}
        cost = _node_cost(node, attrs, shape)
        if cost is _UNKNOWN:
            unknown[node.op_type] += 1
        elif cost is None:
            skipped[node.op_type] += 1
        else:
            m, e, d = cost  # type: ignore[misc]
            macs, elem, dft = macs + m, elem + e, dft + d
    if unknown:
        notes.append("no cost model, not counted: " + ", ".join(f"{k} x{v}" for k, v in sorted(unknown.items())))
    if skipped:
        notes.append("missing shape info, not counted: " + ", ".join(f"{k} x{v}" for k, v in sorted(skipped.items())))
    if dft:
        notes.append("STFT/DFT estimated as 5*N*log2(N) flops per transform")
    return {
        "params": params,
        "size_mb": os.path.getsize(path) / (1024 * 1024),
        "opset": opset,
        "n_nodes": len(graph.node),
        "op_counts": dict(counts),
        "macs": macs,
        "flops": 2 * macs + elem + dft,
        "elementwise_ops": elem,
        "dft_flops": dft,
        "input_shapes": input_shapes,
        "notes": notes,
    }
