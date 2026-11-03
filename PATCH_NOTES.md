# Det-MSCEA B3 implementation branch

Branch: \`feature/det-mscea-b3-mls\`

This branch adds an isolated B3 local-MLS augmentation baseline. It does not alter
the default \`main\` branch, existing model weights, source datasets, or training
entry points.

Added files:
- \`sentinel/augmentation/__init__.py\`
- \`sentinel/augmentation/mls.py\`
- \`scripts/import_yolo_segmentation_masks.py\`
- \`scripts/generate_mls_dataset.py\`
- \`scripts/prepare_mls_b3.py\`
- \`tests/test_mls_augmentation.py\`
- \`docs/det_mscea_b3.md\`
- \`docs/mscea_b3_manifest.example.jsonl\`

Validation performed in an isolated Python 3.13 environment:
\`python -m unittest tests.test_mls_augmentation -v\` -> 6 tests passed.

Not yet performed:
- Full project test suite.
- Running the scripts against the user's actual CVAT export.
- Real-data augmentation visual review.
- YOLO B2/B3 training and evaluation.

Scientific scope:
- This is random local affine-MLS augmentation for the B3 ablation.
- It is not a reproduction of the complete MSCEA method.
- DE, DASA-DE constraint handling, feasible-neighborhood expansion, and
  feature-diversity optimization remain future work.
- Do not claim any detection improvement before real-data experiments.
