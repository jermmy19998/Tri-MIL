import os
import sys
import tempfile
import unittest

from PIL import Image

sys.path.append('../')
from trident.wsi_objects.ImageWSI import ImageWSI


class TestImageWSI(unittest.TestCase):
    def test_default_mpp_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "sample.png")
            Image.new("RGB", (128, 64), color=(255, 255, 255)).save(img_path)

            slide = ImageWSI(img_path, lazy_init=False, default_mpp=0.5)

            self.assertEqual(slide.mpp, 0.5)
            self.assertEqual(slide.mag, 20)
            self.assertEqual(slide.get_dimensions(), (128, 64))


if __name__ == "__main__":
    unittest.main()
