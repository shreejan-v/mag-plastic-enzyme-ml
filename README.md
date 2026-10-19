# Sequence-based classification of plastic-degrading enzymes in marine MAGs

Machine-learning classifiers (logistic regression, XGBoost, 1D-CNN) that flag plastic-degrading enzyme sequences from protein sequence alone, benchmarked against DIAMOND homology search. Built on the PlasticDB reference set and the two metagenome-assembled genomes (bin.3, *Desulfobacterota*; maxbin.002, *Chloroflexota*) from a mangrove-fringed coastal sediment metagenome (see [Genome-Resolved-Metagenomics](https://github.com/shreejan-v/Genome-Resolved-Metagenomics)).

## Data
- **Positives (328):** PlasticDB curated plastic-degrading enzymes.
- **Negatives (8,146):** Prokka-predicted MAG proteins with no PlasticDB hit (DIAMOND, e <= 1e-5); 214 are "hard negatives" annotated as hydrolases, esterases, lipases, proteases, etc.
- **External set (79):** MAG proteins that do hit PlasticDB. Never used for training.

## Method
1. Proteins clustered at 30% identity / 50% coverage; cross-validation splits **whole clusters**, so near-duplicates never straddle train and test.
2. Features: amino-acid, dipeptide and reduced-alphabet tripeptide composition plus length (764 features); the CNN uses the raw sequence (first 1,000 residues).
3. Models: DIAMOND homology baseline (best bitscore to training-fold positives only), logistic regression, XGBoost, 1D-CNN. 5-fold StratifiedGroupKFold, seed 42.

## Results (out-of-fold, homology-grouped)

| Model | ROC-AUC | PR-AUC | Recall @ 1% FPR | ROC-AUC vs hard negatives |
|---|---|---|---|---|
| DIAMOND homology baseline | 0.875 | 0.760 | 0.750 | 0.875 |
| Logistic regression | 0.943 | 0.524 | 0.412 | 0.943 |
| XGBoost | 0.961 | 0.734 | 0.655 | 0.955 |
| **1D-CNN** | **0.965** | **0.858** | **0.805** | **0.968** |

Positive rate is 3.9%, so PR-AUC is the most informative column. Figures: `outputs/pr_curves.png`, `outputs/shap_summary.png`.

## Limitations
- All negatives come from two MAGs while positives come from many organisms, so models may partly learn genome/taxonomic composition rather than function.
- External check on the 79 MAG proteins with PlasticDB hits was mixed: scores did not rank hit tiers cleanly (the CNN scored remote Tier 3 hits above Tier 2). The models do not reliably recover homology-detected candidates.
- `outputs/candidates_no_homology.csv` (top-scoring no-hit proteins, mostly hypothetical) is exploratory only. Any candidate is putative and annotation/sequence-based, not experimentally validated.
- Small positive set (64 homology clusters); PETases dominate.

## Reproduce
```bash
pip install -r requirements.txt
python src/01_prepare.py     # needs the PlasticDB FASTA, MAG proteomes and DIAMOND
python src/02_train.py       # ~40 min on 8 CPU cores
python src/03_explain_apply.py
```
Paths in `01_prepare.py` point to the original project layout; edit `ROOT` to reuse.
