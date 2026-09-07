"""ทดสอบส่วนสำคัญด้วยตัวอย่างที่รู้คำตอบแน่นอน."""
import unittest
import numpy as np
from segmentation import foreground_score, clean_mask, confusion_counts, metrics


class SegmentationTests(unittest.TestCase):
    def test_known_confusion(self):
        truth = np.array([0, 0, 1, 1], dtype=bool)
        prediction = np.array([0, 1, 0, 1], dtype=bool)
        counts = confusion_counts(truth, prediction)
        self.assertEqual(counts, dict(TN=1, FP=1, FN=1, TP=1))
        self.assertAlmostEqual(metrics(counts)["iou"], 1 / 3)

    def test_uniform_image_and_contrasting_object(self):
        image = np.zeros((21, 21, 3), dtype=np.uint8)
        self.assertTrue(np.all(foreground_score(image) == 0))
        image[8:13, 8:13] = 255
        score = foreground_score(image)
        self.assertGreater(score[10, 10], score[0, 0])
        self.assertTrue(np.all((score >= 0) & (score <= 1)))

    def test_morphology_removes_isolated_pixel(self):
        mask = np.zeros((21, 21), dtype=bool)
        mask[7:14, 7:14] = True
        mask[2, 2] = True
        cleaned = clean_mask(mask)
        self.assertFalse(cleaned[2, 2])
        self.assertTrue(cleaned[10, 10])


if __name__ == "__main__":
    unittest.main()
