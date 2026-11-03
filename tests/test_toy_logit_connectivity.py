import numpy as np

from weightbridge.diagnostics.toy_mlp import ToyMLP


def test_logit_connectivity_finds_the_downstream_disconnected_unit():
    base = ToyMLP({
        "input_weight": [[1.0, 2.0], [3.0, 4.0]],
        "hidden_bias": [0.1, 0.2],
        "output_weight": [[0.0, -0.0], [0.0, 2.0]],
        "output_bias": [0.0, 0.0],
    })
    original = {name: tensor.copy() for name, tensor in base.tensors.items()}
    probe = base.logit_connectivity()
    assert probe["disconnected_hidden_units"] == [0]
    assert probe["single_edits_with_no_logit_path"] == {
        "input_weight": [[0, 0], [1, 0]],
        "hidden_bias": [[0]],
    }
    assert all(np.array_equal(tensor, original[name]) for name, tensor in base.tensors.items())
    samples = np.array([[0.1, -0.2], [0.5, 0.1]])
    for tensor, indices in probe["single_edits_with_no_logit_path"].items():
        for index in indices:
            changed = {name: values.copy() for name, values in original.items()}
            changed[tensor][tuple(index)] += 1
            np.testing.assert_array_equal(ToyMLP(changed).forward(samples), base.forward(samples))


def test_logit_connectivity_keeps_nonzero_outgoing_rows_even_when_tiny():
    base = ToyMLP({
        "input_weight": [[0.0, 0.0], [0.0, 0.0]],
        "hidden_bias": [0.0, 0.0],
        "output_weight": [[0.0, 1e-20], [-1e-20, 0.0]],
        "output_bias": [0.0, 0.0],
    })
    assert base.logit_connectivity()["disconnected_hidden_units"] == []
