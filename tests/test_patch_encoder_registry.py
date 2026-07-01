import sys
import unittest

sys.path.append('../')
from trident.patch_encoder_models import encoder_registry, RESIZE_SUPPORTED_PATCH_ENCODERS


class TestPatchEncoderRegistry(unittest.TestCase):
    def test_vit_encoder_registered(self):
        self.assertIn("vit", encoder_registry)
        self.assertIn("vit", RESIZE_SUPPORTED_PATCH_ENCODERS)


if __name__ == "__main__":
    unittest.main()
