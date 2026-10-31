"""Protect manual sampling, cloud relocation, and the selected training origin."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sentinel.training import cloud_bundle
from sentinel.training.urban_data import boxes_from_label
from scripts import train_expanded_smoke


class UrbanTrainingTests(unittest.TestCase):
    def test_labels_reject_missing_geometry_and_nonfinite_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            label=Path(directory)/'sample.txt'
            for content in ['0 .5 .5 .1', '0 nan .5 .1 .1', '2 .5 .5 .1 .1', '0 .01 .5 .4 .1']:
                label.write_text(content)
                with self.subTest(content=content), self.assertRaises(ValueError):
                    boxes_from_label(label)
            label.write_text('0 .5 .5 .2 .2\n1 .25 .25 .1 .1\n')
            self.assertEqual(len(boxes_from_label(label)),2)

    def test_cloud_relocation_preserves_training_repeats(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            image=root/'data/images/manual.jpg';image.parent.mkdir(parents=True);image.write_bytes(b'placeholder')
            (root/'cloud_splits.json').write_text(json.dumps(dict(train=['data/images/manual.jpg']*3,val=[],test=[])))
            (root/'cloud_bundle.json').write_text(json.dumps(dict(data_dir='data')))
            with patch.object(cloud_bundle,'ROOT',root):
                cloud_bundle.prepare()
            self.assertEqual((root/'data/merged/train_balanced.txt').read_text().splitlines(),[image.resolve().as_posix()]*3)

    def test_cloud_relocation_rejects_output_outside_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'cloud_splits.json').write_text('{}')
            (root/'cloud_bundle.json').write_text(json.dumps(dict(data_dir='../escape')))
            with patch.object(cloud_bundle,'ROOT',root), self.assertRaises(ValueError):
                cloud_bundle.prepare()

    def test_default_training_origin_is_ludan_and_dry_run_never_trains(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            merged=root/'datasets/ebike_focused_20261031_v2/merged';merged.mkdir(parents=True)
            for name in ('data_balanced.yaml','train_balanced.txt','val.txt','test.txt'):
                (merged/name).write_text('example\n')
            with patch.object(train_expanded_smoke,'ROOT',root), \
                 patch('sentinel.training.focused_data.verify_runtime_pairs',return_value=1), \
                 patch('sentinel.models.weights.verify',return_value=True) as verify, \
                 patch('sentinel.training.dfire_small_smoke.main') as train:
                self.assertEqual(train_expanded_smoke.main(['--dry-run']),0)
            train.assert_not_called()
            self.assertEqual(verify.call_args.args[1]['sha256'],
                '22fd0c9f6d2498e3ea3d5c3f456f1f13393d66e068a4d1d6c698b82777f2f810')


if __name__=='__main__':
    unittest.main()
