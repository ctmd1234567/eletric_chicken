"""Check unsafe preview inputs are rejected before any model is loaded."""
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from sentinel.training.sam_mask_review import load_prompts


class PromptValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        Image.new('RGB', (40, 30)).save(self.root/'image.png')
        self.row = dict(id='smoke_1', image='image.png', box_xyxy=[1,2,30,25],
                        points=[[10,10]], point_labels=[1])

    def load(self, rows):
        p=self.root/'prompts.jsonl'
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        return load_prompts(p,12)

    def test_rejects_large_unreviewed_batch(self):
        with self.assertRaises(ValueError):
            self.load([dict(self.row,id=f'smoke_{i}') for i in range(13)])

    def test_rejects_colliding_or_traversing_output_names(self):
        for rows in ([self.row,self.row], [dict(self.row,id='../smoke')]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.load(rows)

    def test_rejects_outside_or_nonfinite_geometry(self):
        invalid=[dict(self.row,box_xyxy=[1,2,41,25]),
                 dict(self.row,points=[[float('nan'),10]]),
                 dict(self.row,points=[[40,10]]),
                 dict(self.row,point_labels=[])]
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.load([row])

    def test_accepts_relative_image_and_negative_points(self):
        row=dict(self.row,points=[[10,10],[20,20]],point_labels=[1,0])
        self.assertEqual(self.load([row]),[row])


if __name__=='__main__':
    unittest.main()
