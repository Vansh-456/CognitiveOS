# Cognitive State Compilation (CSC) — Presentation Suite

This folder contains the complete, publication-grade presentation materials for the research paper:  
**"Cognitive State Compilation: Resource-Bounded Construction of Reasoning States for Long-Horizon LLM Agents"** (NeurIPS 2026 Submission).

---

## 🖥️ 1. Interactive Web Presentation Deck (`index.html`)

A modern, responsive 16:9 presentation web application designed for academic conferences, lab meetings, or executive reviews.

### How to Open:
- Double-click [`index.html`](file:///c:/Projects/cognitive_os_1/presentation/index.html) or open it in any modern browser (Chrome, Edge, Firefox, Safari).
- Or run a quick local server:
  ```powershell
  python -m http.server 8000 --directory c:\Projects\cognitive_os_1\presentation
  ```
  and navigate to `http://localhost:8000`.

### Presentation Controls & Shortcuts:
- **`→` / `Space` / `PageDown`**: Advance to next slide
- **`←` / `Backspace` / `PageUp`**: Return to previous slide
- **`Home` / `End`**: Jump to first / last slide
- **`P`**: Toggle **Speaker Talk Track / Presenter Notes** drawer (includes elapsed speech timer)
- **`O`**: Toggle **Slide Overview Grid** (instant thumbnail navigation to any slide)
- **`F`**: Toggle **Fullscreen** presentation mode
- **`Theme`**: Switch between dark glassmorphism and crisp academic light mode
- **Click on any figure**: Opens a high-resolution zoom lightbox
- **`Ctrl + P` (or "Export PDF" button)**: Clean print stylesheet formatted for 16:9 PDF export with page breaks per slide

---

## 📊 2. PowerPoint Presentation (`CSC_Paper_Presentation.pptx`)

A standalone 16:9 widescreen PowerPoint deck formatted with theme colors (Slate `#0f172a`, Indigo `#6366f1`, Cyan `#38bdf8`, Emerald `#34d399`), structured comparison tables, embedded figures, and speaker notes on every slide.

- **File Path**: [`CSC_Paper_Presentation.pptx`](file:///c:/Projects/cognitive_os_1/presentation/CSC_Paper_Presentation.pptx)
- **Regenerate Script**: [`export_pptx.py`](file:///c:/Projects/cognitive_os_1/presentation/export_pptx.py)
  ```powershell
  python export_pptx.py
  ```

---

## 📑 3. Slide Outline & Flow (16 Slides)

| # | Slide Title | Core Focus & Artifacts |
|---|-------------|------------------------|
| **01** | **Title Slide** | Paper metadata, NeurIPS 2026 track, core thesis |
| **02** | **Executive Summary & Highlights** | 100% coverage @ B=500, 88.9% live F1, 100% retention @ N=16 |
| **03** | **The Token Budget Crisis** | Unbounded history growth vs fixed context; the heterogeneous triad |
| **04** | **Landscape & Taxonomy** | Comparison matrix vs MemGPT, A-MEM, Memory-R1, AgeMem, MemOS, MEM1 |
| **05** | **0/1 Knapsack Mathematical Formulation** | Formal objective $\max \sum u$, Dantzig density principle $\rho = u / \text{cost}$ |
| **06** | **System Architecture Pipeline** | 5-stage pipeline diagram: Pools $\to$ 9-D Features $\to$ GBT $\to$ Selector $\to$ LLM |
| **07** | **9-D Features & Production Scorer** | Feature breakdown table, 5-fold CV (98.6% Acc, 94.2% F1), fallback scorer |
| **08** | **Greedy vs Exact DP Selector Audit** | Table 9: 0.0% optimality gap across all domains; &lt; 0.15ms runtime |
| **09** | **Experimental Setup & Baselines** | 4 benchmark domains (HotpotQA, NQ, Chained Q4, WebShop) + 6 baselines |
| **10** | **HotpotQA Coverage @ B=500 Tokens** | Table 2 & Figure 1(a): CSC hits 100% coverage under 460 tokens |
| **11** | **Token Efficiency Frontier** | Figure 1(b): Leftmost Pareto frontier dominance under actual tokens |
| **12** | **Downstream Live LLM Evaluation** | Table 5: Groq Qwen3.6-27B (+22.2 F1 over RAG) & budget saturation |
| **13** | **Long-Horizon Retention vs MEM1** | Table 6 & Figure 3: Horizons N=2..16; 100% retention with zero RL overhead |
| **14** | **Component Ablation Study** | Table 7 & Figure 2(b): Resource-type awareness accounts for -24.4% drop |
| **15** | **Resource Pool Allocation Dynamics** | Table 8 & Figure 2(c): Monotonic shift from Experience/Compute to Knowledge |
| **16** | **Disclosed Limitations & Roadmap** | Scientific integrity disclosures, disjoint test splits, future directions |
