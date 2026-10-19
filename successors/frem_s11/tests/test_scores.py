import numpy as np

from frem_s11.scores import Law, combine_laws, energy_score, moving_block_interval


def test_energy_score_matches_two_atom_reference_value() -> None:
    law = Law(np.asarray([[0.0], [2.0]]), np.asarray([0.5, 0.5]))
    assert energy_score(law, np.asarray([1.0])) == 0.5


def test_mixture_and_constant_block_interval_are_exact() -> None:
    left = Law(np.asarray([[0.0]]), np.asarray([1.0]))
    right = Law(np.asarray([[2.0]]), np.asarray([1.0]))
    mixture = combine_laws((left, right), np.asarray([0.25, 0.75]))
    np.testing.assert_array_equal(mixture.atoms, np.asarray([[0.0], [2.0]]))
    np.testing.assert_array_equal(mixture.probabilities, np.asarray([0.25, 0.75]))
    low, high = moving_block_interval(np.full(36, 0.2), draws=1000)
    np.testing.assert_allclose((low, high), (0.2, 0.2))
