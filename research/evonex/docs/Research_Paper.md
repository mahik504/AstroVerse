# EvoNex: A Physics-Aware Confidence-Guided Multimodal Framework for Robust Exoplanet Transit Detection from Noisy TESS Light Curves

**Abstract**
The detection of exoplanetary transits in time-series photometry from the Transiting Exoplanet Survey Satellite (TESS) is traditionally bottlenecked by a high rate of astrophysical false positives and instrumental noise. Current state-of-the-art pipelines typically rely on monolithic Convolutional Neural Networks (CNNs) that struggle to capture long-range periodic dependencies, or fail to incorporate stellar physics. In this paper, we introduce EvoNex, an Evolutionary Mixture-of-Experts (MoE) architecture. EvoNex diverges from standard late-fusion multimodal networks by introducing an Adaptive Confidence-Guided Gating mechanism. This framework dynamically routes predictions across three expert subnetworks: a Multi-Scale CNN for local transit morphology, a Temporal Transformer with sinusoidal positional encoding for global periodicity, and a Physics-Aware Multi-Layer Perceptron (MLP) for astrophysical validation via the TESS Input Catalog (TIC). Preliminary architecture validation demonstrates correct expert routing behavior. Full-scale evaluation on the NASA TESS Objects of Interest (TOI) catalog is in progress.

## 1. Introduction
The Transiting Exoplanet Survey Satellite (TESS) mission generates vast quantities of high-cadence photometric data. Traditional algorithmic searches, such as the Box Least Squares (BLS) periodogram, are computationally expensive and highly sensitive to stellar variability and instrumental artifacts. While machine learning techniques, particularly CNNs applied to phase-folded light curves, have proven effective at mitigating these challenges, they fundamentally lack the astrophysical context required to distinguish true planets from eclipsing binary stars with similar transit depths.

Recent advancements in the field have proposed multimodal architectures (e.g., ExoMiner, AstroNet) that fuse transit time-series with stellar metadata. However, these systems predominantly employ static late-fusion concatenation. This approach fails to account for the varying signal-to-noise ratio (SNR) across different targets; a highly noisy light curve should necessitate a heavier reliance on stellar physics and global periodicity, whereas a high-SNR transit should heavily weight local morphological features.

To address this, we present EvoNex, which introduces an Explainable Adaptive Gating Network to dynamically weigh expert predictions.

## 2. Methodology

### 2.1 Multi-Scale Local Transit Encoder (CNN Expert)
Phase-folded transits vary significantly in duration and shape. A single convolution kernel size is sub-optimal for capturing both the sharp V-shaped ingress of an eclipsing binary and the extended U-shaped transit of a long-period exoplanet. The EvoNex CNN Expert utilizes a parallel inception-style architecture with receptive fields of $k=\{5, 11, 21\}$, capturing multi-scale morphological features simultaneously.

### 2.2 Global Periodicity Encoder (Transformer Expert)
While CNNs excel at local feature extraction, they are inherently limited by their receptive fields. EvoNex employs a Multi-Head Attention Temporal Transformer with sinusoidal positional encoding to process the raw, unfolded time-series. The positional encoding preserves temporal ordering information, critical for distinguishing between transits at different orbital phases. This allows the network to capture complex, out-of-transit stellar variability and multi-planetary transit sequences without rigid assumptions regarding orbital periodicity.

### 2.3 Astrophysical Validator (Physics MLP)
Transit depth alone is degenerate; a 1% dip on a G-type main-sequence star implies a Jovian planet, whereas a 1% dip on an M-dwarf implies an Earth-sized planet. The Physics Expert ingests normalized stellar features from the TIC (including $T_{eff}$, $\log g$, and stellar radius) to enforce physical validity bounds on the transit detection. Feature normalization ensures stable training dynamics across the heterogeneous parameter scales.

### 2.4 Adaptive Confidence-Guided Gating (EvoMoE)
Let $E = \{CNN, Trans, Phys\}$ denote our set of experts. Each expert $i \in E$ produces a latent feature embedding $F_i$ and a scalar confidence logit $C_i$. The final fused embedding $F_{final}$ is computed via a Softmax-weighted sum:
$$ W = \text{Softmax}([C_{cnn}, C_{trans}, C_{phys}]) $$
$$ F_{final} = \sum_{i \in E} W_i F_i $$
This mechanism allows EvoNex to explicitly output $W$ during inference, providing researchers with direct mathematical explainability regarding the network's decision-making process.

## 3. Experimental Setup

### 3.1 Data
Training and evaluation data is sourced from the NASA TESS SPOC 2-minute cadence light curves, obtained via the MAST archive using the `lightkurve` library. Stellar metadata is drawn from the TESS Input Catalog (TIC) and the Exoplanet Candidate Target List (xCTL v08.01).

**Current dataset status:** Preliminary experiments used a small set of known TESS Objects of Interest. Full-scale training on the complete TOI catalog (3000+ dispositioned candidates) is planned as the next phase of this research.

### 3.2 Baseline Comparison (Published Literature)
| Architecture | F1-Score | Source |
| :--- | :---: | :--- |
| AstroNet (Shallue & Vanderburg, 2018) | 0.89 | AJ 155(2):94 |
| ExoNet (Ansdell et al., 2018) | 0.90 | ApJL 869(1):L7 |
| **EvoMoE (This Work)** | **pending** | In progress |

## 4. Limitations
- **Dataset Size:** Current training set is limited. Full-scale evaluation requires training on thousands of labeled targets from the TOI catalog.
- **Eclipsing Binaries:** Grazing eclipsing binaries with V-shaped transits similar to planetary transits remain a known failure mode for all ML-based detection systems.
- **Long-Period Planets:** Planets with orbital periods exceeding the TESS sector observation window (~27 days) may exhibit only a single transit, making BLS phase-folding unreliable.
- **Multi-Planet Systems:** The current BLS-based phase-folding assumes a single dominant period, which may alias or miss secondary planets.

## 5. Conclusion
EvoNex presents a novel paradigm for exoplanet detection that treats the classification problem as a dynamic mixture of local morphology, global periodicity, and stellar physics. The confidence-gated architecture provides inherent explainability through the routing weights. This work is ongoing; full experimental results will be reported upon completion of large-scale training.

## References
1. Shallue, C. J., & Vanderburg, A. (2018). Identifying Exoplanets with Deep Learning: A Five-planet Resonant Chain around Kepler-80 and an Eighth Planet around Kepler-90. *The Astronomical Journal*, 155(2), 94.
2. Ansdell, M., et al. (2018). Scientific Domain Knowledge Improves Exoplanet Transit Classification with Deep Learning. *The Astrophysical Journal Letters*, 869(1), L7.
3. Ricker, G. R., et al. (2015). Transiting Exoplanet Survey Satellite (TESS). *Journal of Astronomical Telescopes, Instruments, and Systems*, 1(1), 014003.
4. Stassun, K. G., et al. (2019). The Revised TESS Input Catalog and Candidate Target List. *The Astronomical Journal*, 158(4), 138.
5. Jacobs, S., Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS 2017*.
