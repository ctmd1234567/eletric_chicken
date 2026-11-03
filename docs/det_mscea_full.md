# Det-MSCEA: constrained evolutionary MLS augmentation for smoke detection

Status: implementation branch **feature/det-mscea-full**. This is a detection-task adaptation of Xu et al., DOI 10.1109/TEVC.2026.3706679; it is not a reproduction of the paper's image-classification benchmark tables.

## What is implemented

- Deterministic local MLS warping driven by an explicit contour-displacement vector.
- A microscale search region represented as a union of alpha-neighborhood boxes. The initial box is centered on zero displacement; each distinct feasible candidate adds its own neighborhood.
- Differential evolution with objective/constraint comparison.
- Correlation-guided dynamic allocation (DAS-DE): two comparison populations, correlation measurement, kappa-based objective/constraint allocation, epsilon-OP, feasibility-rule CP-C, and the FROFI constraint-priority branch.
- Detection adaptation of the paper's objective: sum of per-channel variances after appending the candidate instance feature to a same-class, same-size-bucket feature bank.
- Hard constraints:
  - C1: matched smoke confidence after warping is not lower than the source-instance confidence.
  - C2: no newly introduced smoke detection outside the local ROI may exceed the configured confidence threshold.
  - C3: warped/original tight-bbox area ratio stays in [0.70, 1.40].
  - C4: the smoke mask bottom edge moves by no more than 5% of the local ROI height.
- A frozen Ultralytics YOLO evaluator that extracts the Detect-head input feature maps and reuses the same model for source confidence, candidate confidence, background false positives, and feature vectors.
- The generator saves only feasible variants, updates the target YOLO box from the warped mask, preserves all other labels, and records optimizer diagnostics and hashes.

## Difference from the classification paper

The paper optimizes whole-image classification examples using Canny edge points and a classification feature/probability evaluator. Det-MSCEA deliberately changes the problem for small-smoke detection:

- The decision controls are sampled from a reviewed smoke-instance mask, not Canny edges from the full image.
- MLS is restricted to a local ROI with fixed boundary anchors; the rest of the frame remains unchanged.
- Features are pooled from the YOLO neck level selected for the instance scale, and the reference bank is grouped by that scale.
- C1 uses the frozen detector's matched instance confidence. C2 and the geometry constraints are detection-specific extensions.
- Hard examples are selected by source confidence (default [0.05, 0.50]); C1 itself remains relative to the source confidence, as in the paper. It is not inverted into a “make the sample harder” objective.

Consequently, results from this code must be described as a Det-MSCEA adaptation, not as exact reproduction of the original classification method.

## Files

- sentinel/augmentation/mls.py: deterministic contour controls and MLS warping.
- sentinel/augmentation/das_de.py: correlation-guided DAS-DE and FROFI constraint-priority search.
- sentinel/augmentation/mscea.py: microscale feasible-region expansion, diversity objective, detection constraints, and per-instance optimization.
- scripts/generate_det_mscea.py: frozen-YOLO adapter and dataset generator.
- tests/test_mscea_optimizer.py: synthetic optimizer, region, feature-objective, and MLS tests.

## Prepare masks

The input manifest is the output of scripts/import_yolo_segmentation_masks.py. Each row must identify one image, one smoke-instance mask, one detection-label file, and the target label index. Masks must be manually reviewed; do not synthesize a smoke mask from a detection rectangle.

Example:

~~~powershell
.\.venv\Scripts\python.exe scripts\import_yolo_segmentation_masks.py --images datasets/cvat_seg/images --seg-labels datasets/cvat_seg/labels --det-labels datasets/ebike_focused_20261031_v2/labels/train --out datasets/mscea_b3/mask_import --smoke-class-id 0 --fire-class-id 1 --min-iou 0.5
~~~

## Run a pilot

Start with ten randomly selected manifest instances. The default confidence interval selects hard but detected smoke examples; rows without a baseline matched detection are skipped because C1 cannot be evaluated for them.

~~~powershell
.\.venv\Scripts\python.exe scripts\generate_det_mscea.py --manifest datasets/mscea_b3/mask_import/manifest.jsonl --weights runs/dfire_ludan/experiment_002/train/weights/best.pt --out outputs/det_mscea/pilot_001 --device 0 --imgsz 960 --max-instances 10 --max-reference-instances 256 --min-source-confidence 0.05 --max-source-confidence 0.50 --population-size 30 --max-evals 300 --alpha 0.07 --roi-scale 2.0 --top-k 3
~~~

The output has:

- images/train/: generated training images.
- labels/train/: updated YOLO labels.
- masks/train/: transformed instance masks.
- manifest.jsonl: one record per feasible candidate.
- summary.json: selected source instances, skipped reasons, settings, and counts.

Every candidate requires a detector forward pass. Do not run the full dataset until the pilot images, masks, boxes, constraint values, and false-positive behavior have been reviewed.

## Build a training config and train

The existing isolated dataset-config builder can consume this manifest because the output layout is the same as the B3 generator:

~~~powershell
.\.venv\Scripts\python.exe scripts\prepare_mls_b3.py --base-data datasets/ebike_focused_20261031_v2/merged/data_balanced.yaml --generated-root outputs/det_mscea/pilot_001 --out outputs/det_mscea/prepared_001 --repeat 1
~~~

After manually auditing generated samples, train into a fresh run directory:

~~~powershell
.\.venv\Scripts\python.exe scripts\finetune_dfire_small_smoke.py --data outputs/det_mscea/prepared_001/data_b3.yaml --out runs/dfire_det_mscea/pilot_001 --base-model dfire_ludan --epochs 30 --imgsz 960 --batch 8 --lr0 0.0001 --freeze 10
~~~

Do not compare this pilot with B2/B3 unless training split, base checkpoint, training schedule, seed, and total sample count are aligned.

## Algorithm and parameter notes

- Search population: 30; maximum candidate evaluations: 300; return up to three best feasible variants per source instance.
- The DAS correlation stage must complete at least one paired measurement generation. With a population of 30 this requires at least 45 evaluations (15 initial plus 30 offspring). This minimum is enforced even if 5% of the total budget would be smaller. Record this setting when comparing against the paper.
- The standard DE branch currently uses F=0.5 and CR=0.9. The FROFI branch uses its published parameter pools F in {0.6, 0.8, 1.0} and CR in {0.1, 0.2, 1.0}. Confirm the main MSCEA paper's exact DE settings against the author's full manuscript before claiming parameter-exact reproduction.
- The objective is minimized internally as negative feature diversity; the reported diversity is positive.
- Constraint violations are normalized and summed. Feasibility requires all four components to be zero.

## Current verification boundary

The optimizer and geometry code have synthetic unit tests, but this branch has not yet been run against the user's local weights, real smoke masks, or a GPU. A passing unit test does not establish useful augmentation or detection improvement. The first real run must verify that the Ultralytics Detect-head hook captures the expected three feature levels for the selected checkpoint and that the letterbox-to-feature-map ROI mapping is correct. The full experiment requires B0-B8 controls, three seeds, and an untouched real-image test set as described in the project research plan.
