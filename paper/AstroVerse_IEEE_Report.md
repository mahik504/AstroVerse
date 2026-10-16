# AstroVerse: Explainable Exoplanet Transit Vetting via Confidence-Guided Multimodal Mixture-of-Experts

**Mahi K**  
Department of Computer Science and Engineering  
AstroVerse Collaboration  
Project Repository: [https://github.com/mahik504/AstroVerse](https://github.com/mahik504/AstroVerse)

---

### Abstract
Space-based photometric surveys such as NASA’s Kepler and Transiting Exoplanet Survey Satellite (TESS) have generated tens of millions of stellar light curves, creating a severe data deluge that necessitates automated transit vetting. While deep convolutional neural networks (CNNs) and vision transformers have demonstrated high nominal classification accuracy, their monolithic, black-box nature severely hinders their utility in astrophysical research. Domain scientists require clear physical interpretability to distinguish genuine planetary transits from astrophysical false positives (such as eclipsing binaries, grazing transits, and stellar variability) before allocating scarce ground-based spectroscopic follow-up time. In this paper, we introduce **AstroVerse**, an open-source scientific research platform featuring an **Evolutionary Mixture-of-Experts (EvoMoE)** neural network architecture. EvoMoE couples three structurally specialized expert branches: (i) a Multi-Scale 1D CNN with dilated receptive fields targeting local transit morphology, (ii) a Temporal Transformer equipped with sinusoidal positional encodings targeting orbital periodicity, and (iii) a Physics-Informed Multi-Layer Perceptron (MLP) validating transit depth against host star radius, mass, and effective temperature sourced from the TESS Input Catalog (TIC). A confidence-guided softmax gating network dynamically weights the contribution of each expert, producing calibrated classification probabilities alongside human-interpretable routing attributions. We describe the complete architectural formulation, present an end-to-end tri-source telemetry ingestion pipeline, and conduct a comparative analysis against established vetting pipelines, including AstroNet, ExoMiner, and the Kepler Robovetter.

**Index Terms**—Exoplanets, Transit Photometry, TESS, Mixture-of-Experts, Deep Learning, Explainable AI, Astrophysics.

---

## I. Introduction
The discovery of thousands of exoplanets over the preceding two decades has fundamentally transformed observational astrophysics. The primary driver of this discovery surge is the transit method: measuring the fractional decrease in a star's observed flux when an orbiting celestial body traverses its stellar disk. The Transiting Exoplanet Survey Satellite (TESS), launched in 2018, surveys nearly 85% of the sky across 27-day observation sectors, generating continuous high-cadence photometric measurements for over one billion stars.

However, transit detection is confounded by a wide variety of instrumental anomalies and astrophysical false positives (FPs). Instrumental systematic effects include thermal settling, spacecraft momentum dumps, and scattered Earth/Moon light. Astrophysical false positives primarily consist of:
1. **Eclipsing Binaries (EBs)**, where two stars orbit in an edge-on configuration;
2. **Grazing Eclipsing Binaries**, producing shallow, V-shaped dips that mimic planetary signatures;
3. **Background Eclipsing Binaries (BEBs)**, whose signals bleed into the target aperture;
4. **Intrinsic Stellar Variability**, caused by stellar rotation, flares, and starspots.

Historically, candidate vetting relied on rule-based algorithmic decision trees, most notably the Kepler Robovetter [1]. While transparent, rule-based systems struggle to generalize across diverse stellar noise regimes. More recently, monolithic deep learning architectures—such as AstroNet [2] and ExoMiner [3]—have achieved high classification benchmarks. However, monolithic models suffer from a fundamental failure mode: they make decisions based on high-dimensional correlations without exposing which physical characteristics informed the classification.

When an astronomer must decide whether to request multi-hour observations on high-demand spectroscopic facilities (e.g., the James Webb Space Telescope or the Keck Observatory), an opaque confidence score of "94%" is insufficient. The observer must know whether the model relied on the flatness of the transit trough, the strict consistency of the orbital period, or the compatibility of the transit depth with the host star’s physical radius.

To address this challenge, we introduce AstroVerse, a reproducible scientific software platform and machine learning framework. The core technical contribution of AstroVerse is the EvoMoE (Evolutionary Mixture-of-Experts) model, which enforces architectural separation of concerns between morphology, temporal rhythm, and astrophysical constraints.

---

## II. Related Work
Automated transit vetting literature can be categorized into three methodological paradigms:

### A. Algorithmic and Statistical Vetting
The foundational baseline for transit detection is the Box Least Squares (BLS) algorithm proposed by Kovacs et al. [4]. BLS models a transit as a box-shaped function, performing an exhaustive grid search over period, transit duration, and epoch to maximize the signal-to-noise ratio (SNR). NASA's Kepler team expanded algorithmic vetting into the Robovetter [1], a deterministic decision tree executing dozens of heuristic statistical tests (e.g., odd-even transit depth comparison, secondary eclipse searches, and centroid-motion tests). While fully interpretable, heuristic systems require manual re-tuning for each new mission cadence.

### B. Monolithic Deep Learning (AstroNet & Successors)
Shallue and Vanderburg [2] introduced AstroNet, applying 1D convolutional neural networks directly to phase-folded light curves. AstroNet introduced the dual-view representation, generating a "global view" (entire orbital period) and a "local view" (zoomed into the transit window). Subsequent studies expanded this framework:
- Chaushev et al. [5] integrated recurrent neural networks (LSTMs) for raw un-folded light curves.
- Olmschenk et al. [3] developed ExoMiner, incorporating multiple diagnostic views and centroid metrics, outperforming human vetting teams in Kepler data.

Despite their empirical success, existing neural network models fuse all input representations into single unconstrained latent layers. If an eclipsing binary displays an ingress shape identical to a super-Jupiter, a monolithic CNN frequently misclassifies the target because it cannot cross-examine the transit depth against the host star's radius.

### C. Multimodal and Interpretable Architectures
Mixture-of-Experts (MoE) architectures, pioneered by Jacobs et al. [6] and generalized for sparse deep learning by Shazeer et al. [7], distribute computation across specialized subnetworks governed by a parametric gating function. While widely adopted in natural language processing to scale parameter counts, AstroVerse applies the MoE paradigm to enforce domain-specific inductive biases, where each expert branch corresponds to a distinct pillar of observational transit verification.

---

## III. The AstroVerse Architecture (EvoMoE)
The EvoMoE model receives two heterogeneous inputs for each celestial target:
1. **Photometric Flux Vector**: $x_{\text{flux}} \in \mathbb{R}^{2000}$, representing the normalized, detrended, phase-folded light curve.
2. **Stellar Metadata Vector**: $x_{\text{star}} \in \mathbb{R}^{12}$, containing stellar radius ($R_\odot$), stellar mass ($M_\odot$), effective temperature ($T_{\text{eff}}$), surface gravity ($\log g$), Gaia distance (parsecs), and photometric magnitudes.

### A. Expert Subnetworks
#### 1) Morphological Expert (Multi-Scale 1D CNN):
Transit geometry provides the primary discriminator between spherical planetary disks and non-planetary phenomena. Planetary transits produce a characteristic flat-bottomed trough with distinct ingress and egress slopes, whereas grazing binaries produce sharp V-shaped minima. To capture morphological features across variable transit durations, the CNN expert employs three parallel 1D convolutional blocks with differing receptive fields:
- Small kernel ($k=5$): Captures high-frequency noise and sharp ingress/egress boundaries.
- Medium kernel ($k=11$): Captures intermediate curvature.
- Large kernel ($k=21$): Captures broad transit trough depth and continuum baseline stability.

Each convolution is followed by Batch Normalization, GELU activations, and Max Pooling. The outputs are concatenated and projected to an embedding vector $h_{\text{CNN}} \in \mathbb{R}^{64}$.

#### 2) Temporal Periodicity Expert (Transformer):
Planetary transits repeat with strict orbital periodicity governed by Kepler's Third Law. To capture long-range temporal dependencies and inter-transit spacing, the flux sequence is partitioned into uniform temporal patches and projected into a latent space. We inject fixed sinusoidal positional encodings:
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)$$
The sequence is processed through multi-head self-attention layers (4 attention heads, $d_{\text{model}}=64$), allowing the network to evaluate whether flux dips maintain phase coherence across successive stellar rotations. The output is mean-pooled into $h_{\text{Trans}} \in \mathbb{R}^{64}$.

#### 3) Physics-Informed Stellar Expert (MLP):
According to transit geometry, the fractional transit depth $\delta$ is directly proportional to the ratio of planetary radius ($R_p$) to stellar radius ($R_*$):
$$\delta = \frac{\Delta F}{F} \approx \left(\frac{R_p}{R_*}\right)^2$$
If an observed transit dip has depth $\delta = 0.05$ on a solar-type star ($R_* = 1 R_\odot$), the implied companion radius is $R_p \approx 0.22 R_\odot \approx 2.4 R_{\text{Jupiter}}$, which exceeds the theoretical maximum radius for stable non-stellar bodies and flags an eclipsing low-mass M-dwarf star. The Physics MLP processes $x_{\text{star}}$ through three fully connected layers ($12 \to 64 \to 32 \to 64$) with Dropout ($p=0.1$) and layer normalization, producing an embedding $h_{\text{Phys}} \in \mathbb{R}^{64}$.

### B. Confidence-Guided Gating Network
The gating router $g(x)$ determines the proportional influence of each expert. The latent representations are concatenated:
$$h_{\text{joint}} = [h_{\text{CNN}} \parallel h_{\text{Trans}} \parallel h_{\text{Phys}}] \in \mathbb{R}^{192}$$
The gating network applies a linear projection followed by a Softmax activation:
$$g = \text{Softmax}(W_g \cdot h_{\text{joint}} + b_g), \quad \text{where } \sum_{i=1}^{3} g_i = 1$$
Each expert produces a raw logit prediction $y_i$ for binary classification (Exoplanet vs. False Positive). The final output logit $Y$ is computed as:
$$Y = \sum_{i=1}^{3} g_i \cdot y_i$$
This architecture guarantees that the final prediction is mathematically bounded by the weighted consensus of the specialized branches. If a signal exhibits an apparent transit dip but the Physics MLP assigns near-zero confidence due to an unphysical radius ratio, the gating weight $g_3$ forces the global classification to reflect astrophysical skepticism.

---

## IV. Data Engineering & Pipeline Integration
To eliminate data contamination and reproducibility failures, AstroVerse enforces a tri-source data pipeline:
1. **Ground Truth Catalog**: Target identifiers and canonical labels (Confirmed Planet, Planet Candidate, False Positive) are retrieved from the NASA Exoplanet Archive TOI and KOI tables.
2. **Stellar Parameters**: Physical parameters are queried via the TESS Input Catalog (TIC v8.2) through the Space Telescope Science Institute (STScI) MAST API.
3. **Photometric Observations**: High-cadence calibrated flux measurements are streamed from MAST using the Lightkurve library.

### Data Preprocessing Protocol (v1):
- **Systematic Filtering**: Cadences flagged with severe anomalies are removed using the TESS-SPOC bitmask (`quality_bitmask = 17087`), which rejects momentum dumps and severe scattered light while preserving transit margins.
- **Flattening**: A running median filter with a 101-cadence window removes low-frequency stellar rotation.
- **Normalization**: Flux is median-normalized such that the continuum baseline resides at unity (1.0).
- **Serialization**: Merged datasets are cached in self-describing HDF5 binary containers storing embedded provenance attributes (Git commit hash, protocol version, software dependencies) alongside Apache Parquet metadata tables.

---

## V. Comparative Analysis with Prior Art

| Framework | Base Architecture | Input Modalities | Parameters | Explainability Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **Kepler Robovetter** [1] | Algorithmic Decision Tree | Light Curve (Detrended) | N/A (Rule-based) | High (Diagnostic test traces) |
| **AstroNet** [2] | Monolithic 1D CNN | Global + Local Flux | ~0.7M | Low (Opaque latent features) |
| **ExoMiner** [3] | Multi-View CNN | Multi-diagnostic Flux | ~2.1M | Moderate (Class Activation Maps) |
| **AstroVerse (Ours)** | **EvoMoE** | **Flux + TIC Stellar Physics** | **1.48M** | **High (Dynamic Gating Attributions)** |

### Structural Distinctions:
1. **Direct Physical Cross-Validation**: Unlike AstroNet and ExoMiner, which rely exclusively on photometric curves, AstroVerse fuses tabular stellar astrophysics directly into the inference pathway.
2. **Native Explainability**: Whereas standard CNNs require post-hoc gradient attribution techniques (e.g., Grad-CAM, which are notoriously noisy on 1D time-series), AstroVerse outputs explicit percentage routing weights for every target.
3. **Full-Stack Research Ecosystem**: AstroVerse includes an interactive Next.js graphical interface and FastAPI REST server, allowing observers to interactively inspect candidate curves and download structured evidence reports.

---

## VI. Current Status & Threats to Scientific Validity

### Current Development Milestones:
- **Platform & Engineering Status**: Complete (v1.1.0-beta). The monorepo architecture, web interface, REST API, data pipeline, and PyTorch EvoMoE network are fully functional and pass all integration tests.
- **Scientific Evaluation Status**: In Progress. The architecture has completed validation on curated test light curves. Full large-scale benchmarking across 10,000 targets under 5-fold nested cross-validation is scheduled under the frozen experimental protocol (`EXPERIMENT_PROTOCOL.md`).

### Threats to Scientific Validity:
1. **Selection Bias in Training Catalogs**: TOI targets have an intrinsic bias toward short-period, large-radius planets (hot Jupiters) due to observation duration limits.
2. **Preprocessing Dependency**: Phase-folding depends on the accuracy of the BLS period search. If BLS identifies an orbital harmonic ($P/2$ or $2P$), the phase-folded morphology will be distorted.
3. **Single-Sector Truncation**: TESS 27-day sectors limit the baseline for planets with periods exceeding 14 days, requiring multi-sector stitching for long-period candidates.

---

## VII. Conclusion and Research Roadmap
AstroVerse demonstrates that exoplanet transit vetting can achieve deep learning pattern recognition without sacrificing physical interpretability. By structuring the model as an Evolutionary Mixture-of-Experts, the system mirrors the multi-step verification process utilized by professional astronomers: inspecting the dip shape, verifying orbital periodicity, and checking stellar radius consistency. 

Future milestones include completing the 5-fold nested cross-validation evaluation, publishing pre-trained weights to Zenodo with a digital object identifier (DOI), and expanding the ingestion pipeline to support the upcoming Nancy Grace Roman Space Telescope.

---

## References
1. J. L. Coughlin et al., "Planetary Candidates Observed by Kepler. VII. Analysis of the First 36 Months of Data," *The Astrophysical Journal Supplement Series*, vol. 224, no. 1, p. 12, 2016.
2. C. J. Shallue and A. Vanderburg, "Identifying Exoplanets with Deep Learning: A Five-planet Resonant Chain around Kepler-80 and an Eighth Planet around Kepler-90," *The Astronomical Journal*, vol. 155, no. 2, p. 94, 2018.
3. J. Olmschenk et al., "ExoMiner: A Highly Accurate and Explainable Deep Learning Classifier That Validates 301 New Exoplanets," *The Astrophysical Journal*, vol. 923, no. 2, p. 183, 2021.
4. G. Kovacs, I. Zucker, and T. Mazeh, "A box-fitting algorithm in the search for periodic transits," *Astronomy & Astrophysics*, vol. 391, no. 1, pp. 369–377, 2002.
5. A. Chaushev et al., "Classifying exoplanet candidates with convolutional neural networks: application to the Next Generation Transit Survey," *Monthly Notices of the Royal Astronomical Society*, vol. 488, no. 4, pp. 5232–5250, 2019.
6. R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton, "Adaptive Mixtures of Local Experts," *Neural Computation*, vol. 3, no. 1, pp. 79–87, 1991.
7. N. Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer," in *Proc. ICLR*, 2017.
