# Research status

October 2026.

## In the repository

| Piece | State |
| --- | --- |
| EvoMoE module | Specified in `model_evonex.py`. 1,485,605 parameters. Median CPU latency 7.64 ms |
| Reduction and BLS fold | Implemented. Period grid 0.5–20 days |
| AstroLens and the FastAPI service | Implemented. The forward pass is `POST /predict` |
| Reference library | 18 series in `demo/`, with measured depths in the SCIE manuscript |
| Processed SPOC cache | Six TESS light curves |
| Manuscript | `paper/scie/AstroVerse_SCIE_manuscript.pdf` |

## Next measurement

Classification scores on a held-out TESS Objects of Interest sample, with average precision and a reliability diagram, using the catalog schedule in `research/evonex/configs/paper.yaml`. Until that run is reported, this repository does not publish an accuracy, an F1, or an AUPRC for EvoMoE.

Literature operating points for AstroNet, ExoMiner, and Astronet-Triage-v2 are cited in the manuscript. They are those papers' numbers.
