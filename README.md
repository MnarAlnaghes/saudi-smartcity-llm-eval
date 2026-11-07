# Sovereign LLMs for Saudi Smart Cities: A Comparative Evaluation

Code, benchmark, and results for the paper *"Free and Sovereign LLMs for Saudi Smart Cities:
A Comparative Evaluation"* (submitted to ICAISC 2027).

We compare five open-weight LLMs that a city can run on its own infrastructure (one GPU, no
per-token cost): **ALLaM-7B** (SDAIA), **GPT-OSS-20B** (OpenAI), **Gemma 4 12B** (Google),
**Qwen3-8B** (Alibaba), and **Llama 3.1 8B** (Meta), on:

- **STBench+** — 18 international urban spatio-temporal tasks (POIs, trajectories, regions,
  navigation, traffic semantics, anomaly detection, flow and trajectory forecasting).
- **KSA-UrbanBench** (new) — 300 bilingual (Arabic/English) questions on Riyadh, Jeddah, Makkah,
  Madinah, and Dammam, with ground truth computed from Saudi National Address district boundaries.
- **Service readiness** — ArabicMMLU, SimpleQA, IFEval, HumanEval+, XSTest (safe prompts).

## Repository layout

| Folder | Contents |
|---|---|
| `harness/` | Colab notebook that runs the whole study (`saudi_smartcity_llm_eval.ipynb`), the same code as a Python file, and `make_figs.py` |
| `benchmark/` | `ksa_urbanbench.py` (generator) and `ksa_urbanbench_items.jsonl` (the 600 prompts used) |
| `results/` | Per-item scores, summary statistics, pairwise tests, and the LaTeX tables used in the paper |
| `figures/` | Figures used in the paper and supplement |
| `raw_outputs/` | Every model response (gzipped JSONL), judge outputs, model load status, and GPU timing |
| `supplementary/` | Supplementary material (per-task, per-city, accuracy vs. size, all p-values) |

## Reproducing

1. Open `harness/saudi_smartcity_llm_eval.ipynb` in Google Colab with an **A100 GPU**.
2. Add a Colab secret `HF_TOKEN` (Hugging Face read token) and accept the Llama 3.1 license on Hugging Face.
3. Mount Google Drive, set `MODE = "full"` in Cell 2, and run all. Runs resume after disconnects.

Evaluation run: November 6–7, 2026, one NVIDIA A100-SXM4-40GB, vLLM, 16-bit weights.

## Regenerating KSA-UrbanBench

```bash
git clone https://github.com/zalabbad/Saudi-Arabia-Regions-Cities-and-Districts.git ksa_districts
git -C ksa_districts checkout f9813f37ec006f52e99b9082a81bb7fa66f88dec
pip install shapely
python benchmark/ksa_urbanbench.py ksa_districts ksa_urbanbench_items.jsonl 10
```

## Data sources and licenses

- District boundaries: [Saudi-Arabia-Regions-Cities-and-Districts](https://github.com/zalabbad/Saudi-Arabia-Regions-Cities-and-Districts)
  (data collected from the Saudi National Address maps), GPL-2.0.
- STBench+: [LwbXc/STBench](https://github.com/LwbXc/STBench), pinned commit `2022eaa`.
- Other benchmarks are downloaded from their original sources and keep their own licenses.

## Citation

```bibtex
@inproceedings{TODO_saudi_sovereign_llm_2027,
  title     = {Free and Sovereign {LLMs} for {Saudi} Smart Cities: A Comparative Evaluation},
  author    = {TODO},
  booktitle = {Proc. 3rd Int. Conf. Advanced Innovations in Smart Cities (ICAISC)},
  year      = {2027}
}
```
