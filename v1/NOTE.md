# v1 (October 2025): kept as history

These are my original ML mini-project notebooks from October 2025, written before I used AI tools.
They replicate the Stanford CS229 report *Amazon Inventory Reconciliation Using AI*
(Rodriguez Bertorello, Sripada, Dendumrongsup, 2018) on a very small subset of the Amazon Bin Image Dataset,
loaded from my own GCS bucket in Colab. The notebooks and the original README (`README_v1.md`) are unchanged.

## What the saved notebook outputs actually show

The old README said "CNN 35% vs SVM 20%, 333 images". Re-reading the outputs in Nov 2026, that summary is not right:

| v1 model | Result in the saved output |
|---|---|
| Always predict "3 items" (majority class) | 30.6% val (22/72) |
| SVM, blob features | 29.2% val |
| SVM, HOG on Laplacian | 17.3% test (n=75, a different random 80/20 split) |
| SVM, raw pixels + PCA (SVC C=1e2, gamma=1e-7) | 34.7% val, 25.0% test (n=36) |
| ResNet-18, SGD | 33.3% val (best epoch) |
| ResNet-34, SGD | 31.9% val (best epoch) |
| ResNet-34, Adam | 34.7% val (best epoch) |

- The filtered set was 371 images with 0-5 items (263 train / 72 val / 36 test), not 333. The val set had no empty bins.
- With 72 val images the 95% interval is about +/-11 points, so every model is within noise of the majority-class guess.
  The CNNs did not beat the SVM.
- The CNN numbers are the best epoch picked on val, and the CNNs were never run on test.
- The split was shuffled without a seed, and the pixel SVM loaded a stale batch file
  (`counting_train.X1.npy`, 61 rows) left over from an earlier split, so train/val may overlap.

v2 (the rest of this repo) redoes the experiment at a sensible scale with fixed splits, then adds real object detection.
