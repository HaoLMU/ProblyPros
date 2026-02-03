"""
Pytest test file for JAXDataGenerator

Test goals:
1. Verify the behavior of _count() after using jnp.bincount
2. Check that generate() produces a stable output structure with reasonable values
3. Ensure save() / load() correctly persist and restore results
4. Verify that save() is safe and has no side effects when results are empty

Design principles:
- Use a "perfect prediction" dummy model to avoid randomness
- Focus tests on behavior and contracts, not implementation details
"""

import json
import importlib
from pathlib import Path

import jax.numpy as jnp
import pytest


# Adjust this according to the actual module path of jax_generator
MODULE_PATH = "src.probly.data_generation.jax_generator"


@pytest.fixture(scope="module")
def jax_generator_module():
    """
    Dynamically import the module under test.
    """
    return importlib.import_module(MODULE_PATH)


@pytest.fixture
def perfect_dataset():
    """
    Construct a "perfectly predictable" binary classification dataset.

    Rule:
    - logits = [x0, -x0]
    - x0 >= 0 -> class 0
    - x0 <  0 -> class 1
    """
    x0 = jnp.array([1.0, 2.0, -1.0, -3.0, 0.5, -0.2])
    x = jnp.stack([x0, jnp.zeros_like(x0)], axis=1)
    y = jnp.where(x0 >= 0, 0, 1).astype(jnp.int32)
    return x, y


@pytest.fixture
def simple_model():
    """
    A minimal usable dummy model.

    Input:
        x: shape (N, 2)

    Output:
        logits: shape (N, 2)

    Used to test the JAXDataGenerator pipeline,
    not the correctness of the model itself.
    """
    def model(x: jnp.ndarray) -> jnp.ndarray:
        x0 = x[:, 0]
        return jnp.stack([x0, -x0], axis=1)

    return model


def test_count_bincount_behavior(jax_generator_module):
    """
    Test whether the bincount behavior of _count() matches expectations.

    Key assertions:
    1. The counts are correct
    2. Classes with zero count are not returned
    """
    JAXDataGenerator = jax_generator_module.JAXDataGenerator

    # dataset / model are irrelevant for this test
    gen = JAXDataGenerator(
        model=lambda x: x,
        dataset=(jnp.zeros((1, 1)), jnp.zeros((1,), dtype=jnp.int32)),
    )

    values = jnp.array([0, 0, 2, 5], dtype=jnp.int32)
    result = gen._count(values)

    # After bincount, only non-zero entries should be kept
    assert result == {0: 2, 2: 1, 5: 1}

    # Missing intermediate classes should not appear
    assert 1 not in result
    assert 3 not in result
    assert 4 not in result


def test_generate_structure_and_metrics(jax_generator_module, simple_model, perfect_dataset):
    """
    Test the overall output of generate().

    Coverage:
    - Stability of the returned structure
    - Correctness of accuracy
    - Consistency between predicted and ground_truth distributions
    - Reasonable confidence values
    """
    JAXDataGenerator = jax_generator_module.JAXDataGenerator

    gen = JAXDataGenerator(
        model=simple_model,
        dataset=perfect_dataset,
        batch_size=16,
        device="cpu",
    )

    results = gen.generate()

    # ---------- Structure validation ----------
    assert set(results.keys()) == {
        "info",
        "metrics",
        "class_distribution",
        "confidence",
    }

    assert results["info"]["framework"] == "jax"
    assert results["info"]["dataset_size"] == perfect_dataset[0].shape[0]
    assert results["info"]["batch_size"] == 16

    # ---------- Metrics validation ----------
    # With perfect predictions, accuracy must be 1
    assert results["metrics"]["accuracy"] == pytest.approx(1.0)

    # predicted and ground_truth distributions must match exactly
    assert (
        results["class_distribution"]["predicted"]
        == results["class_distribution"]["ground_truth"]
    )

    # ---------- Confidence sanity check ----------
    # After softmax, max probability should be > 0.5
    confidence = results["confidence"]
    assert 0.5 < confidence["mean"] <= 1.0
    assert confidence["std"] >= 0.0


def test_save_and_load_roundtrip(
    jax_generator_module,
    simple_model,
    perfect_dataset,
    tmp_path,
):
    """
    Test full round-trip consistency of save() / load().

    Workflow:
    1. generate()
    2. save() to JSON
    3. load() into a new instance
    4. Compare results for full equality
    """
    JAXDataGenerator = jax_generator_module.JAXDataGenerator

    gen = JAXDataGenerator(model=simple_model, dataset=perfect_dataset)
    results = gen.generate()

    out_file = tmp_path / "results.json"
    gen.save(str(out_file))

    assert out_file.exists()

    # File content should be valid JSON and match results exactly
    raw = json.loads(out_file.read_text(encoding="utf-8"))
    assert raw == results

    # After load(), the new instance should have identical state
    gen2 = JAXDataGenerator(model=simple_model, dataset=perfect_dataset)
    loaded = gen2.load(str(out_file))

    assert loaded == results
    assert gen2.results == results


def test_save_with_empty_results_does_nothing(
    jax_generator_module,
    simple_model,
    perfect_dataset,
    tmp_path,
):
    """
    Test: when results are empty, save() should not create a file.

    This is a safety test ensuring:
    - No empty JSON file is created
    - No exception is raised
    """
    JAXDataGenerator = jax_generator_module.JAXDataGenerator

    gen = JAXDataGenerator(model=simple_model, dataset=perfect_dataset)

    out_file = tmp_path / "should_not_exist.json"
    gen.save(str(out_file))

    assert not out_file.exists()
