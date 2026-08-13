"""Regression tests for checkpoint-driven VETime decoder selection."""

import unittest

from src.utils.checkpoint_architecture import (
    checkpoint_model_config,
    checkpoint_state_dict,
    make_model_checkpoint,
)


class CheckpointArchitectureTests(unittest.TestCase):
    def test_new_checkpoint_uses_saved_query_decoder_setting(self):
        checkpoint = {
            "model_state_dict": {"query_decoder.task_token_rec": object()},
            "model_config": {"use_query_decoder": True},
        }

        self.assertTrue(checkpoint_model_config(checkpoint)["use_query_decoder"])

    def test_legacy_checkpoint_infers_query_decoder_from_state_dict(self):
        checkpoint = {"query_decoder.task_token_rec": object()}

        self.assertTrue(checkpoint_model_config(checkpoint)["use_query_decoder"])
        self.assertEqual(checkpoint_state_dict(checkpoint), checkpoint)

    def test_legacy_moe_checkpoint_defaults_to_moe(self):
        checkpoint = {"mm_w.Router.l1.weight": object()}

        self.assertFalse(checkpoint_model_config(checkpoint)["use_query_decoder"])

    def test_new_checkpoint_preserves_state_dict_and_architecture(self):
        state_dict = {"query_decoder.task_token_rec": object()}

        checkpoint = make_model_checkpoint(state_dict, use_query_decoder=True)

        self.assertEqual(checkpoint_state_dict(checkpoint), state_dict)
        self.assertTrue(checkpoint_model_config(checkpoint)["use_query_decoder"])


if __name__ == "__main__":
    unittest.main()
