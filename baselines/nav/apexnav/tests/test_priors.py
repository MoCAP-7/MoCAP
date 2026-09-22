import tempfile
import unittest
from pathlib import Path

from baselines.nav.apexnav.priors import load_target_prior


class PriorTest(unittest.TestCase):
    def test_cached_prior_is_loaded(self):
        prior = load_target_prior(
            "baselines/nav/apexnav/target_priors.yaml", "chair"
        )
        self.assertEqual(prior.source, "cache")
        self.assertEqual(prior.room, "everywhere")
        self.assertEqual(prior.confidence_threshold, 0.30)

    def test_unknown_open_vocabulary_target_remains_supported(self):
        prior = load_target_prior(
            "baselines/nav/apexnav/target_priors.yaml", "blue recycling bin"
        )
        self.assertEqual(prior.source, "llm_disabled_default")
        self.assertEqual(prior.similar_labels, ())
        self.assertEqual(prior.confidence_threshold, 0.50)

    def test_more_than_four_similar_labels_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at most four"):
            load_target_prior(
                "baselines/nav/apexnav/target_priors.yaml",
                "object",
                similar_labels=["a", "b", "c", "d", "e"],
            )


if __name__ == "__main__":
    unittest.main()
