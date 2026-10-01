import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

onnx = pytest.importorskip("onnx")
from onnx import TensorProto as TP  # noqa: E402
from onnx import helper as h  # noqa: E402
from onnx import numpy_helper as nh  # noqa: E402

from vcmbench.flops import format_si, onnx_profile  # noqa: E402


def _init(name, shape):
    return nh.from_array(np.zeros(shape, dtype=np.float32), name)


def _save(tmp_path, nodes, inputs, outputs, inits=(), opset=17):
    graph = h.make_graph(nodes, "g", inputs, outputs, list(inits))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", opset)])
    path = tmp_path / "m.onnx"
    onnx.save(model, str(path))
    return str(path)


def _vi(name, shape, dtype=TP.FLOAT):
    return h.make_tensor_value_info(name, dtype, shape)


def test_format_si():
    assert format_si(0) == "0"
    assert format_si(999) == "999"
    assert format_si(1234567) == "1.23 M"
    assert format_si(4.5e9) == "4.5 G"
    assert format_si(1500) == "1.5 K"
    assert format_si(999999) == "1 M"
    assert format_si(2e12) == "2 T"


def test_conv_1x1(tmp_path):
    path = _save(
        tmp_path,
        [h.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[1, 1])],
        [_vi("x", [1, 3, 4, 4])], [_vi("y", [1, 8, 4, 4])], [_init("w", [8, 3, 1, 1])],
    )
    p = onnx_profile(path)
    assert p["macs"] == 8 * 16 * 3  # out elems * Cin * 1
    assert p["flops"] == 2 * p["macs"]
    assert p["params"] == 24
    assert p["opset"] == 17 and p["n_nodes"] == 1
    assert p["op_counts"] == {"Conv": 1}
    assert p["input_shapes"] == {"x": [1, 3, 4, 4]}
    assert p["size_mb"] > 0
    assert p["notes"] == []


def test_conv_3x3_grouped(tmp_path):
    # depthwise: Cin=4, group=4, W (4,1,3,3), out 1x4x2x2
    path = _save(
        tmp_path,
        [h.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], group=4)],
        [_vi("x", [1, 4, 4, 4])], [_vi("y", [1, 4, 2, 2])], [_init("w", [4, 1, 3, 3])],
    )
    assert onnx_profile(path)["macs"] == 16 * 1 * 9


def test_gemm(tmp_path):
    path = _save(
        tmp_path,
        [h.make_node("Gemm", ["a", "b", "c"], ["y"])],
        [_vi("a", [2, 5])], [_vi("y", [2, 7])], [_init("b", [5, 7]), _init("c", [7])],
    )
    p = onnx_profile(path)
    assert p["macs"] == 2 * 7 * 5
    assert p["flops"] == 140
    assert p["params"] == 35 + 7


def test_gemm_transA(tmp_path):
    path = _save(
        tmp_path,
        [h.make_node("Gemm", ["a", "b"], ["y"], transA=1)],
        [_vi("a", [5, 2])], [_vi("y", [2, 7])], [_init("b", [5, 7])],
    )
    assert onnx_profile(path)["macs"] == 70


def test_matmul_batched(tmp_path):
    path = _save(
        tmp_path,
        [h.make_node("MatMul", ["a", "b"], ["y"])],
        [_vi("a", [2, 3, 4])], [_vi("y", [2, 3, 6])], [_init("b", [4, 6])],
    )
    assert onnx_profile(path)["macs"] == 2 * 3 * 6 * 4


def test_elementwise_and_free_ops(tmp_path):
    nodes = [
        h.make_node("MatMul", ["x", "w"], ["m"]),
        h.make_node("Relu", ["m"], ["r"]),
        h.make_node("Softmax", ["r"], ["s"]),
        h.make_node("Reshape", ["s", "shp"], ["y"]),
    ]
    shp = nh.from_array(np.array([5, 2], dtype=np.int64), "shp")
    path = _save(tmp_path, nodes, [_vi("x", [1, 10])], [_vi("y", [5, 2])], [_init("w", [10, 10]), shp])
    p = onnx_profile(path)
    assert p["macs"] == 100
    assert p["elementwise_ops"] == 10 + 5 * 10
    assert p["flops"] == 200 + 60
    assert p["notes"] == []


def test_batchnorm_and_pool(tmp_path):
    nodes = [
        h.make_node("BatchNormalization", ["x", "s", "b", "mu", "var"], ["bn"]),
        h.make_node("MaxPool", ["bn"], ["y"], kernel_shape=[2, 2], strides=[2, 2]),
    ]
    inits = [_init(n, [3]) for n in ("s", "b", "mu", "var")]
    path = _save(tmp_path, nodes, [_vi("x", [1, 3, 4, 4])], [_vi("y", [1, 3, 2, 2])], inits)
    p = onnx_profile(path)
    assert p["macs"] == 0
    assert p["elementwise_ops"] == 2 * 48 + 4 * 12


def test_dynamic_dims_assumed_one(tmp_path):
    path = _save(
        tmp_path,
        [h.make_node("Gemm", ["a", "b"], ["y"])],
        [_vi("a", ["batch", 5])], [_vi("y", ["batch", 7])], [_init("b", [5, 7])],
    )
    p = onnx_profile(path)
    assert p["macs"] == 35
    assert p["input_shapes"] == {"a": [1, 5]}
    assert any("batch" in n and "assumed 1" in n for n in p["notes"])


def test_lstm(tmp_path):
    seq, batch, i, hid = 6, 2, 3, 4
    path = _save(
        tmp_path,
        [h.make_node("LSTM", ["x", "w", "r"], ["y"], hidden_size=hid)],
        [_vi("x", [seq, batch, i])], [_vi("y", [seq, 1, batch, hid])],
        [_init("w", [1, 4 * hid, i]), _init("r", [1, 4 * hid, hid])],
    )
    assert onnx_profile(path)["macs"] == seq * batch * 4 * hid * (i + hid)


def test_gru_bidirectional(tmp_path):
    seq, batch, i, hid = 5, 1, 3, 4
    path = _save(
        tmp_path,
        [h.make_node("GRU", ["x", "w", "r"], ["y"], hidden_size=hid, direction="bidirectional")],
        [_vi("x", [seq, batch, i])], [_vi("y", [seq, 2, batch, hid])],
        [_init("w", [2, 3 * hid, i]), _init("r", [2, 3 * hid, hid])],
    )
    assert onnx_profile(path)["macs"] == 2 * seq * batch * 3 * hid * (i + hid)


def test_dft_counted(tmp_path):
    n = 16
    path = _save(
        tmp_path,
        [h.make_node("DFT", ["x"], ["y"], axis=1, onesided=0)],
        [_vi("x", [1, n, 1])], [_vi("y", [1, n, 2])],
    )
    p = onnx_profile(path)
    assert p["dft_flops"] == 5 * n * 4
    assert p["flops"] == p["dft_flops"]
    assert any("DFT" in x for x in p["notes"])


def test_unknown_op_noted(tmp_path):
    path = _save(
        tmp_path,
        [h.make_node("NonMaxSuppression", ["b", "s"], ["y"])],
        [_vi("b", [1, 4, 4]), _vi("s", [1, 1, 4])], [_vi("y", [1, 3], TP.INT64)],
    )
    p = onnx_profile(path)
    assert p["flops"] == 0
    assert any("NonMaxSuppression" in n for n in p["notes"])
