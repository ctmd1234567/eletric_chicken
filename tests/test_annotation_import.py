"""Import-only exports must preserve reviewed negatives and clip, not drop, edge boxes."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from PIL import Image

from sentinel.training.annotation_import import clip_labels, import_export
from sentinel.training.focused_data import sampling_repeats, verify_runtime_pairs


class AnnotationImportTests(unittest.TestCase):
    def test_manifest_only_label_paths_are_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'merged').mkdir();(root/'images').mkdir();(root/'annotations').mkdir()
            image=root/'images/one.png';Image.new('RGB',(32,32)).save(image)
            label=root/'annotations/one.txt';label.write_text('0 .5 .5 .2 .2\n')
            row=dict(image='images/one.png',label='annotations/one.txt',label_sha256=hashlib.sha256(label.read_bytes()).hexdigest())
            (root/'merged/manifest.jsonl').write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError,'Actual YOLO label path'):
                verify_runtime_pairs(root)
            (root/'labels').mkdir();(root/'labels/one.txt').write_bytes(label.read_bytes())
            row['label']='labels/one.txt';(root/'merged/manifest.jsonl').write_text(json.dumps(row)+'\n')
            self.assertEqual(verify_runtime_pairs(root),1)
    def test_clip_retains_visible_intersection_and_other_boxes(self):
        text,changes=clip_labels('0 0.95 0.9 0.2 0.4\n1 .5 .5 .1 .1\n',{0:0,1:1})
        first=list(map(float,text.splitlines()[0].split()))
        self.assertEqual(len(text.splitlines()),2)
        self.assertAlmostEqual(first[1],.925)
        self.assertAlmostEqual(first[3],.15)
        self.assertAlmostEqual(first[4],.3)
        self.assertEqual(text.splitlines()[1],'1 .5 .5 .1 .1')
        self.assertEqual(len(changes),1)

    def test_invalid_or_non_intersecting_boxes_are_rejected(self):
        for text in ['0 nan .5 .2 .2','0 2 2 .1 .1','2 .5 .5 .1 .1']:
            with self.subTest(text=text),self.assertRaises(ValueError):clip_labels(text,{0:0,1:1})

    def test_missing_files_become_background_only_with_explicit_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'images').mkdir();rows=[]
            for name in ('one','two'):
                image=root/'images'/(name+'.png');Image.new('RGB',(32,32)).save(image)
                rows.append(dict(image='images/'+image.name,image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),source_id=name))
            (root/'manifest.json').write_text(json.dumps(rows))
            archive=root/'export.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('data.yaml','names: [smoke, fire]\n')
                z.writestr('labels/train/one.txt','0 .5 .5 .2 .2\n')
            first=import_export(archive,root,root/'first')
            self.assertEqual(first['labelled_images'],1)
            self.assertFalse(first['missing_labels_used_as_background'])
            second=import_export(archive,root,root/'second',review_missing_as_background=True,training_authorized=True)
            self.assertEqual(second['labelled_images'],2)
            self.assertEqual(second['explicit_empty_labels'],1)
            self.assertEqual((root/'second/labels/two.txt').read_text(),'')
            self.assertEqual(second['rights_status'],'user_confirmed_training_authorization')

    def test_manual_and_interference_sampling_shares_increase(self):
        counts=dict(manual=140,jjs=32,regular=590)
        repeats=sampling_repeats(counts)
        self.assertEqual(repeats,dict(manual=4,jjs=7,regular=1))
        total=sum(counts[k]*repeats[k] for k in counts)
        self.assertGreater(counts['manual']*repeats['manual']/total,.4)
        self.assertGreater(counts['jjs']*repeats['jjs']/total,.15)


if __name__=='__main__':unittest.main()
