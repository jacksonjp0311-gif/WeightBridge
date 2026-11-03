import numpy as np

from experiments.published_reproduction.make_random_control import _bf16_bits, _bf16_values, _next_bf16


def test_bf16_roundtrip_and_minimum_nonzero_directional_control_edit():
    values = np.asarray([0x0000, 0x3F80, 0xBF80, 0x4120], dtype="<u2")
    assert np.array_equal(_bf16_bits(_bf16_values(values)), values)
    assert float(_bf16_values(np.asarray([_next_bf16(0x3F80, 1)], dtype="<u2"))[0]) > 1.0
    assert float(_bf16_values(np.asarray([_next_bf16(0xBF80, 1)], dtype="<u2"))[0]) > -1.0
    assert float(_bf16_values(np.asarray([_next_bf16(0, -1)], dtype="<u2"))[0]) < 0.0
