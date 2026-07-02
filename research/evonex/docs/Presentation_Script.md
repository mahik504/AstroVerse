# EvoNex: Official Presentation & Explanation Guide
**Your Master Script for Hackathon Judges and Academic Peers**

This document is your complete "cheat sheet." It contains exactly how to explain the architecture in plain English, the scientific justification for every component, and the exact steps to follow during a live demonstration.

---

## Part 1: The Problem (The Hook)
*When you start your presentation, outline the problem clearly.*

**What to say:**
> "The NASA TESS telescope stares at stars and looks for tiny dips in light, hoping those dips are planets. But there's a huge problem: instrument glitches, camera noise, and binary star systems also cause dips in light. Furthermore, processing TESS data is computationally expensive—downloading and preprocessing targets involves dealing with gigabytes of time-series data.
>
> To solve this, most teams build a standard Convolutional Neural Network (CNN). But standard CNNs are a 'black box.' They don't understand the physical laws of the universe. We wanted to build something better. We built **EvoNex**."

---

## Part 2: Explaining the Architecture
*This is the core of your project. Break down the EvoMoE (Evolutionary Mixture-of-Experts) into these four simple pieces.*

### 1. The Local Expert (Multi-Scale CNN)
*   **The Metaphor:** The "Shape Detector."
*   **How to explain it:** "Our first AI branch focuses entirely on the exact shape of the transit dip. But instead of one lens, we use a 'Multi-Scale' approach. It processes the light curve through three different sized filters simultaneously. Small filters catch the sharp, V-shaped dips of binary stars, while large filters catch the long, flat U-shaped dips of real planets."

### 2. The Global Expert (Temporal Transformer)
*   **The Metaphor:** The "Rhythm Detector."
*   **How to explain it:** "Our second AI branch is a Transformer—the same core tech behind ChatGPT. We add sinusoidal positional encoding so it knows *where* in time each data point is. Instead of just looking at the zoomed-in dip, the Transformer looks at the entire 27-day observation. Its job is to find complex, long-range periodicities, like multi-planet systems, and differentiate them from random stellar variability."

### 3. The Stellar Expert (Physics MLP)
*   **The Metaphor:** The "Astrophysicist."
*   **How to explain it:** "AI without physics is blind. A 1% dip in light on a gigantic Sun-like star means something completely different than a 1% dip on a tiny Red Dwarf. Our third branch queries the NASA MAST API to pull real physical traits of the star (Temperature, Radius, Gravity) — normalized for stable training — so the AI can validate its guess against the laws of physics."

### 4. The Orchestrator (Confidence-Guided Gating)
*   **The Metaphor:** The "Boss."
*   **How to explain it:** "We don't just mash these three answers together. EvoNex uses an Adaptive Gating Network. Each of our three experts outputs a guess AND a confidence score. If the light curve is extremely noisy but the physics are clear, the Gating Network dynamically chooses to trust the Physics Expert more. It adapts on the fly to every single star."

---

## Part 3: The "Wow" Factors

If the judges ask what makes your project special:

1.  **Total Explainability:** "Because of our Gating Network, EvoNex is not a black box. During inference, it literally outputs its reasoning. It tells you, *'I believe this is an exoplanet, and I made this decision relying 45% on the CNN shape, 25% on the Transformer rhythm, and 30% on the Physics data.'* Judges and scientists demand explainability, and we built it in natively."
2.  **HDF5 Caching:** "We built a custom PyTorch DataLoader that queries the NASA API, cleans the noise with Savitzky-Golay filters, folds the math using Box Least Squares, and then caches the massive tensor arrays into an HDF5 Database. This significantly reduces repeated data loading time per target."

---

## Part 4: Your Live Demo Script (Step-by-Step)
*When it's time to show the project working on your screen, follow these exact steps.*

### Step 1: Open the Visual Proofs
1.  Open your terminal in `C:\projects\AstroVerse`.
2.  Navigate to the research folder: `cd research\evonex`
3.  Activate the venv and run `jupyter lab`
4.  Open `notebooks/01_EDA_and_Preprocessing.ipynb`.
    *   Show them the raw vs. cleaned light curve comparison.
    *   Say: *"This is our mathematical preprocessing engine stripping out cosmic rays and telescope jitter."*
5.  Open `notebooks/03_BLS_Detection.ipynb`.
    *   Show the folded transit graph with the transit highlight.
    *   Say: *"This is our algorithm mathematically extracting the exact orbital period and transit duration before the AI even touches it."*

### Step 2: Run the Architecture Validation
1.  Open a new PowerShell terminal.
2.  Navigate to the folder and activate the environment.
3.  Type: `python src/evaluate_evomoe.py` and hit Enter.
4.  Point out the key output:
    *   **Architecture dimensions:** Show them the input/output shapes proving the 3-expert fusion works.
    *   **The Explainability Printout:** Highlight the gating distribution where the model outputs the exact routing percentages. Say: *"Right there, you can see the model distributing its decision across the three experts."*

### Step 3: Show AstroLens (The Full-Stack Platform)
1.  Start the backend: `cd C:\projects\AstroVerse\services\evonex-api` → `.\venv\Scripts\Activate.ps1` → `python -m uvicorn main:app --reload`
2.  Start the frontend: `cd C:\projects\AstroVerse\apps\astrolens-web` → `npm run dev`
3.  Open `http://localhost:3000` in the browser.
4.  Upload the test CSV file (`TIC_25155310_WASP126b.csv`).
5.  Click **Run EvoNex Inference** and show the live lightcurve chart and expert routing matrix.
6.  Say: *"This is a full production platform — separate frontend, API backend, and ML engine — all communicating in real time."*

---

## What NOT to Claim
- Do not cite specific F1/precision/ROC-AUC numbers unless you have run the full training pipeline on a large dataset and can verify them.
- Do not claim "300x speed optimization" without a benchmark to back it up.
- If asked about metrics, say: *"Our architecture validation confirms correct expert routing behavior. Full-scale evaluation on the TOI catalog is in progress."*
