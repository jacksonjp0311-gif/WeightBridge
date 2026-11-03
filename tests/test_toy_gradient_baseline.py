from experiments.original_weight_edits.run_toy_gradient_baseline import train
from weightbridge.diagnostics import ToyMLP, make_defect_benchmark


def test_conventional_sgd_updates_parameters_using_practice_only_labels():
    benchmark = make_defect_benchmark(seed=151, per_class=16)
    model = benchmark["model"]
    before = {name: value.copy() for name, value in model.tensors.items()}
    result = train(model, benchmark["practice"]["features"], benchmark["practice"]["labels"],
                   steps=100, learning_rate=0.05)
    assert result["optimizer"] == "full_batch_SGD"
    assert result["practice_loss_end"] < result["practice_loss_start"]
    assert any((model.tensors[name] != before[name]).any() for name in model.tensor_names)
