import unittest

import numpy as np

from marginals import (
    add_dp_noise_count_queries,
    aggregate_count_queries,
    compute_1way_counts_shared,
    compute_2way_counts_shared,
    compute_shared_bin_edges,
    secure_sum_integer_vectors,
)


class ProtocolTests(unittest.TestCase):
    def test_modular_secure_sum_matches_plain_sum(self):
        vectors = [
            np.array([4, -2, 7], dtype=np.int64),
            np.array([3, 5, -1], dtype=np.int64),
            np.array([-6, 1, 2], dtype=np.int64),
        ]
        observed = secure_sum_integer_vectors(vectors, rng_seed=9)
        expected = np.sum(np.stack(vectors), axis=0)
        np.testing.assert_array_equal(observed, expected)

    def test_count_workload_uses_joint_sensitivity(self):
        rng = np.random.RandomState(4)
        parties = [{"X": rng.randn(20, 3)} for _ in range(2)]
        edges = compute_shared_bin_edges(parties, n_bins=4)
        one = compute_1way_counts_shared(parties[0]["X"], edges)
        two = compute_2way_counts_shared(parties[0]["X"], [(0, 1)], edges)
        _, _, _, sensitivity = add_dp_noise_count_queries(
            one, two, epsilon=1.0, delta=1e-5,
            rng=np.random.RandomState(5),
        )
        self.assertAlmostEqual(sensitivity, np.sqrt(2.0 * 4.0))

    def test_secure_aggregate_outputs_probability_tables(self):
        party_one = [[{"hist": np.array([3.0, 1.0]), "col": 0,
                       "edges": np.array([0.0, 1.0, 2.0])}],
                     [{"hist": np.array([2.0, 4.0]), "col": 0,
                       "edges": np.array([0.0, 1.0, 2.0])}]]
        aggregate, pairs = aggregate_count_queries(
            party_one, [[], []], secure=True,
        )
        self.assertEqual(pairs, [])
        np.testing.assert_allclose(aggregate[0]["hist"], [0.5, 0.5])

    def test_shared_edges_use_discrete_domains_for_categoricals(self):
        parties = [
            {"X": np.array([[0.0, -1.0], [2.0, 1.0]], dtype=np.float32)},
            {"X": np.array([[1.0, 0.0], [2.0, 2.0]], dtype=np.float32)},
        ]
        edges = compute_shared_bin_edges(
            parties, n_bins=20, categorical_indices=[0]
        )
        np.testing.assert_array_equal(
            edges[0], np.array([-0.5, 0.5, 1.5, 2.5])
        )
        self.assertEqual(len(edges[1]), 21)


if __name__ == "__main__":
    unittest.main()
