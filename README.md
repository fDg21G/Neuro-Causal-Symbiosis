# Neuro-Causal-Symbiosis
# 🧠 Neuro-Causal Symbiosis (NCS)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Preprint](https://img.shields.io/badge/Status-Preprint-blue.svg)]()

> **Escaping the Embedding Trap:** A hybrid AI architecture that strictly separates linguistic processing from causal inference, delegating causal truth to a deterministic empirical engine.

## 📄 Research Paper

**[Read the official preprint: Escaping the Embedding Trap](papers/NCS_Research_Paper.pdf)**

Large Language Models (LLMs) exhibit a systematic failure termed the **"Embedding Trap"**—the conflation of distributional co-occurrence statistics with physical causal structure. This repository contains the official implementation of **Neuro-Causal Symbiosis (NCS)**, a novel architecture that resolves this limitation through strict functional decomposition.

## 🏗️ Architecture Overview

NCS designates the LLM as a pure semantic router and entity extractor, delegating actual causal inference to a deterministic engine grounded in real-world time-series data from the Federal Reserve Economic Data (FRED).

* **System 1 (Empirical Reflex Core):** A multi-evidence ensemble (Granger Causality, Transfer Entropy, Phase-Space Lag-Correlation) operating on real macroeconomic time-series.
* **System 2 (Semantic Cloud):** An LLM fallback reserved *exclusively* for abstract/semantic domains where no measurable physical time-series exists (outputting a Semantic Consensus Index).
* **The Router:** A deterministic confidence-threshold mechanism that arbitrates between empirical truth and semantic consensus.

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy scipy pandas statsmodels spacy
python -m spacy download en_core_web_sm
Running the NCS Router
Python
from ncs_core import NCSRouter

# Initialize the router
router = NCSRouter()

# Empirical Query (Routes to System 1 -> FRED Data)
router.evaluate("Does inflation cause interest rates to rise?")

# Abstract Query (Routes to System 2 -> Semantic Cloud)
router.evaluate("Does freedom cause happiness?")
👨‍💻 Author
Houssam Rharbi Independent Researcher Casablanca, Morocco
