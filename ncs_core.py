"""
Neuro-Causal Symbiosis (NCS) Engine  — v2.0 PRODUCTION
System 1 (Empirical) + System 2 (LLM Semantic) Hybrid
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ==============================================================================
# 🚀 IMPORT THE PURE MATH ENGINE (LTI)
# ==============================================================================
from lti_engine_v5_final import robust_causal_direction

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("NCS_Engine")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 ─ DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CausalResult:
    """Fully structured causal analysis result."""
    query:        str
    entity_a:     str
    entity_b:     str
    direction:    str          # "A → B" | "B → A" | "BIDIRECTIONAL" | "UNDETERMINED"
    confidence:   float        # [0.0 – 1.0]
    method:       str
    system_used:  int          # 1 = Empirical Reflex,  2 = Semantic LLM
    data_source:  str = "Unknown"
    time_range:   Optional[str] = None
    explanation:  Optional[str] = None
    warnings:     List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "entity_a": self.entity_a,
            "entity_b": self.entity_b,
            "direction": self.direction,
            "confidence": float(self.confidence),
            "method": self.method,
            "system_used": self.system_used,
            "data_source": self.data_source,
            "time_range": self.time_range,
            "explanation": self.explanation,
            "warnings": self.warnings,
        }

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 ─ FRED REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

FRED_REGISTRY: Dict[str, str] = {
    "inflation": "CPIAUCSL",
    "cpi": "CPIAUCSL",
    "consumer price index": "CPIAUCSL",
    "core inflation": "CPILFESL",
    "pce": "PCE",
    "ppi": "PPIACO",
    "producer price index": "PPIACO",
    "producer prices": "PPIACO",
    "interest rates": "FEDFUNDS",
    "interest rate": "FEDFUNDS",
    "federal funds rate": "FEDFUNDS",
    "fed funds rate": "FEDFUNDS",
    "treasury yield": "GS10",
    "10 year yield": "GS10",
    "mortgage rate": "MORTGAGE30US",
    "gdp": "GDP",
    "real gdp": "GDPC1",
    "industrial production": "INDPRO",
    "manufacturing": "IPMAN",
    "unemployment": "UNRATE",
    "unemployment rate": "UNRATE",
    "nonfarm payrolls": "PAYEMS",
    "employment": "PAYEMS",
    "wages": "AHETPI",
    "money supply": "M2SL",
    "m2": "M2SL",
    "credit": "TOTALSL",
    "oil": "DCOILWTICO",
    "oil prices": "DCOILWTICO",
    "crude oil": "DCOILWTICO",
    "housing prices": "CSUSHPISA",
    "home prices": "CSUSHPISA",
    "sp500": "SP500",
    "stock market": "SP500",
    "exports": "EXPGSC1",
    "imports": "IMPGSC1",
    "dollar": "DTWEXBGS",
    "consumer confidence": "UMCSENT",
    "retail sales": "RSAFS",
}

ABSTRACT_CONCEPTS = frozenset({
    "love", "happiness", "sadness", "anger", "fear", "justice",
    "freedom", "democracy", "beauty", "morality", "ethics",
    "consciousness", "meaning", "purpose", "truth", "trust",
    "hope", "grief", "pain", "suffering", "well-being", "fairness",
})


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 ─ SYNTHETIC DATA (Sandbox mode)
# ══════════════════════════════════════════════════════════════════════════════

_SERIES_META: Dict[str, dict] = {
    "CPIAUCSL":     dict(level=200.0,  std=0.30,  trend=0.18,  unit="Index"),
    "FEDFUNDS":     dict(level=2.50,   std=0.15,  trend=0.0,   unit="%"),
    "UNRATE":       dict(level=5.50,   std=0.10,  trend=-0.005,unit="%"),
    "GDP":          dict(level=14000,  std=40.0,  trend=42.0,  unit="Bil $"),
    "GDPC1":        dict(level=13000,  std=35.0,  trend=28.0,  unit="Bil $"),
    "M2SL":         dict(level=7000,   std=20.0,  trend=55.0,  unit="Bil $"),
    "PAYEMS":       dict(level=130000, std=50.0,  trend=7.0,   unit="K"),
    "INDPRO":       dict(level=100.0,  std=0.50,  trend=0.04,  unit="Index"),
    "GS10":         dict(level=3.00,   std=0.10,  trend=0.0,   unit="%"),
    "SP500":        dict(level=1500,   std=20.0,  trend=16.0,  unit="Index"),
    "DCOILWTICO":   dict(level=60.0,   std=2.0,   trend=0.0,   unit="$/bbl"),
    "CSUSHPISA":    dict(level=150.0,  std=0.80,  trend=0.8,   unit="Index"),
    "AHETPI":       dict(level=20.0,   std=0.08,  trend=0.04,  unit="$/hr"),
    "UMCSENT":      dict(level=80.0,   std=2.0,   trend=0.0,   unit="Index"),
    "PPIACO":       dict(level=160.0,  std=0.80,  trend=0.12,  unit="Index"),
    "PCE":          dict(level=10000,  std=30.0,  trend=35.0,  unit="Bil $"),
    "MORTGAGE30US": dict(level=5.0,    std=0.08,  trend=0.0,   unit="%"),
    "RSAFS":        dict(level=380000, std=1500,  trend=250.0, unit="Mil $"),
    "TOTALSL":      dict(level=2500,   std=8.0,   trend=10.0,  unit="Bil $"),
    "CPILFESL":     dict(level=185.0,  std=0.20,  trend=0.14,  unit="Index"),
    "DTWEXBGS":     dict(level=100.0,  std=0.50,  trend=0.0,   unit="Index"),
    "EXPGSC1":      dict(level=2000,   std=10.0,  trend=8.0,   unit="Bil $"),
    "IMPGSC1":      dict(level=2500,   std=12.0,  trend=9.0,   unit="Bil $"),
    "TOTLL":        dict(level=9000,   std=25.0,  trend=20.0,  unit="Bil $"),
    "IPMAN":        dict(level=95.0,   std=0.40,  trend=0.02,  unit="Index"),
    "BOPGSTB":      dict(level=-500,   std=8.0,   trend=-0.5,  unit="Bil $"),
}

_CAUSAL_GRAPH = [
    ("CPIAUCSL",   "FEDFUNDS",     6,   0.75),
    ("FEDFUNDS",   "MORTGAGE30US", 2,   0.85),
    ("FEDFUNDS",   "UNRATE",       9,   0.40),
    ("M2SL",       "CPIAUCSL",    12,   0.55),
    ("DCOILWTICO", "PPIACO",       2,   0.80),
    ("PPIACO",     "CPIAUCSL",     3,   0.60),
    ("PAYEMS",     "CPIAUCSL",     6,   0.30),
    ("CPIAUCSL",   "GS10",         3,   0.50),
    ("GS10",       "CSUSHPISA",    6,   0.45),
]

def _build_fred_dataset(start: str = "2000-01-01", end: str = "2024-12-01") -> Dict[str, pd.Series]:
    rng   = np.random.default_rng(42)
    dates = pd.date_range(start, end, freq="MS")
    n     = len(dates)
    innovations: Dict[str, np.ndarray] = {}
    
    for sid, meta in _SERIES_META.items():
        innovations[sid] = rng.normal(0, meta["std"], n)
        
    for cause_id, effect_id, lag, strength in _CAUSAL_GRAPH:
        if cause_id not in innovations or effect_id not in innovations:
            continue
        cause_inn  = innovations[cause_id]
        effect_inn = innovations[effect_id].copy()
        scale = np.std(cause_inn) / (np.std(effect_inn) + 1e-9)
        for t in range(lag, n):
            effect_inn[t] += strength * cause_inn[t - lag] * scale
        innovations[effect_id] = effect_inn
        
    dataset: Dict[str, pd.Series] = {}
    for sid, meta in _SERIES_META.items():
        trend  = np.arange(n) * meta["trend"]
        levels = meta["level"] + trend + np.cumsum(innovations[sid])
        dataset[sid] = pd.Series(levels, index=dates, name=sid)
    return dataset

_FRED_DATASET: Optional[Dict[str, pd.Series]] = None

def _get_cached_dataset() -> Dict[str, pd.Series]:
    global _FRED_DATASET
    if _FRED_DATASET is None:
        _FRED_DATASET = _build_fred_dataset()
    return _FRED_DATASET

def fetch_fred_series(series_id: str, name: str, start: str = "2000-01-01", end: str = "2024-12-01") -> Optional[pd.Series]:
    ds = _get_cached_dataset()
    if series_id not in ds:
        return None
    series = ds[series_id].loc[start:end].dropna()
    logger.info("  Fetched %s (%s): %d obs", series_id, name, len(series))
    return series

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 ─ SYSTEM 2: SEMANTIC CLOUD (LLM Integration)
# ══════════════════════════════════════════════════════════════════════════════

class SemanticCloud:
    def __init__(self, provider: str = "anthropic", api_key: Optional[str] = None):
        self.provider = provider.lower()
        if self.provider == "anthropic":
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not self.api_key:
                logger.warning("ANTHROPIC_API_KEY not set. Using local fallback.")
                self.client = None
            else:
                try:
                    import anthropic
                    self.client = anthropic.Anthropic(api_key=self.api_key)
                except ImportError:
                    logger.warning("anthropic library not installed. Install: pip install anthropic")
                    self.client = None
        elif self.provider == "openai":
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                logger.warning("OPENAI_API_KEY not set. Using local fallback.")
                self.client = None
            else:
                try:
                    import openai
                    self.client = openai.Client(api_key=self.api_key)
                except ImportError:
                    logger.warning("openai library not installed. Install: pip install openai")
                    self.client = None
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def reason(self, query: str, entity_a: str, entity_b: str, s1_hint: Optional[Tuple[str, float]] = None) -> Tuple[str, float, str]:
        if not self.client:
            return self._fallback_reasoning(query, entity_a, entity_b)

        s1_context = ""
        if s1_hint:
            direction, confidence = s1_hint
            s1_context = f"\n\nNote: An empirical analysis was attempted with confidence {confidence:.0%}, suggesting direction: {direction}. Use this as a cross-check, but rely on semantic domain knowledge."

        prompt = self._build_prompt(query, entity_a, entity_b, s1_context)

        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = response.content[0].text
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = response.choices[0].message.content
            return self._parse_response(response_text, entity_a, entity_b)
        except Exception as e:
            logger.error("LLM API call failed: %s. Using fallback.", e)
            return self._fallback_reasoning(query, entity_a, entity_b)

   def _build_prompt(self, query: str, entity_a: str, entity_b: str, s1_context: str) -> str:
        return f"""You are a causal inference expert reasoning about abstract or semantic concepts where empirical time-series data is unavailable.

User Query: "{query}"
Concepts: "{entity_a}" ↔ "{entity_b}"

TASK:
1. Analyze the causal relationship between these two concepts using domain knowledge, philosophy, psychology, and social understanding.
2. Determine the most likely causal direction:
   - "{entity_a} → {entity_b}" (A causes B)
   - "{entity_b} → {entity_a}" (B causes A)
   - BIDIRECTIONAL (mutual causation)
   - UNDETERMINED (insufficient evidence)
3. Provide a Semantic Consensus Index (SCI) from 0.40 (very low) to 0.95 (very high) reflecting the degree of sociological and linguistic consensus on this relationship.

IMPORTANT CONSTRAINTS:
- This is SEMANTIC reasoning, not empirical data analysis
- Be skeptical of bidirectional claims unless truly justified
- The SCI reflects human consensus, not a physical probability
- Consider alternative interpretations and edge cases{s1_context}

RESPONSE FORMAT (JSON):
{{
    "direction": "A → B",
    "sci_score": 0.65,
    "reasoning": "Brief explanation of the causal mechanism",
    "caveats": "Important limitations or alternative views"
}}
Respond ONLY with valid JSON, no preamble."""

    def _parse_response(self, response: str, entity_a: str, entity_b: str) -> Tuple[str, float, str]:
        try:
            match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
            if not match:
                return self._fallback_reasoning("", entity_a, entity_b)
            data = json.loads(match.group())
            direction = data.get("direction", f"{entity_a} → {entity_b}").strip()
            confidence = float(data.get("sci_score", data.get("confidence", 0.55)))
            reasoning = data.get("reasoning", "")
            caveats = data.get("caveats", "")
            explanation = f"{reasoning} [Caveats: {caveats}]" if caveats else reasoning
            if "→" not in direction:
                if "B → A" in direction or entity_b in direction.split("→")[0]:
                    direction = f"{entity_b} → {entity_a}"
                else:
                    direction = f"{entity_a} → {entity_b}"
            confidence = max(0.4, min(0.95, confidence))
            return direction, confidence, explanation
        except (json.JSONDecodeError, ValueError, AttributeError):
            return self._fallback_reasoning("", entity_a, entity_b)

    def _fallback_reasoning(self, query: str, entity_a: str, entity_b: str) -> Tuple[str, float, str]:
        logger.info("Using fallback semantic reasoning (no LLM available).")
        direction = f"{entity_a} → {entity_b}"
        confidence = 0.55
        explanation = f"Semantic inference: '{entity_a}' likely influences '{entity_b}' through causal mechanisms not captured by time-series data. This is a domain knowledge estimate without empirical validation."
        return direction, confidence, explanation

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 ─ NLP EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

class NLPExtractor:
    def __init__(self):
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            self.has_model = True
        except (OSError, ImportError):
            logger.warning("spaCy model not available. Using regex fallback.")
            self.has_model = False
            self.nlp = None

    def extract_entities(self, query: str) -> List[str]:
        if self.has_model and self.nlp:
            return self._extract_spacy(query)
        return self._extract_regex(query)

    def _extract_spacy(self, query: str) -> List[str]:
        doc = self.nlp(query.lower())
        entities = []
        for chunk in doc.noun_chunks:
            clean = re.sub(r'[^a-z0-9 ]', '', chunk.text.strip())
            if clean:
                entities.append(clean)
        return entities

    def _extract_regex(self, query: str) -> List[str]:
        words = re.findall(r'\b([a-z]+(?:\s+[a-z]+)*)\b', query.lower())
        stop = {'does', 'do', 'is', 'the', 'a', 'an', 'and', 'or', 'in', 'to', 'of'}
        return [w for w in words if w not in stop and len(w) > 2]

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 ─ THE NCS ENGINE (Main Orchestrator)
# ══════════════════════════════════════════════════════════════════════════════

class NCS_Engine:
    def __init__(self, llm_provider: str = "anthropic", llm_api_key: Optional[str] = None, confidence_threshold: float = 0.80):
        self.extractor = NLPExtractor()
        self.semantic_cloud = SemanticCloud(provider=llm_provider, api_key=llm_api_key)
        self.threshold = confidence_threshold
        logger.info("NCS Engine initialised (threshold=%.0f%%)", self.threshold * 100)

    def _map_to_registry(self, entities: List[str]) -> List[Tuple[str, str]]:
        mapped = []
        for entity in entities:
            entity_lower = entity.lower()
            if entity_lower in FRED_REGISTRY:
                mapped.append((entity_lower, FRED_REGISTRY[entity_lower]))
                continue
            for reg_key, series_id in FRED_REGISTRY.items():
                if entity_lower in reg_key or reg_key in entity_lower:
                    if (entity_lower, series_id) not in mapped:
                        mapped.append((entity_lower, series_id))
                    break
        return mapped

    def analyze(self, query: str) -> CausalResult:
        logger.info("\n" + "="*70)
        logger.info("📩 Query: '%s'", query)
        logger.info("="*70)

        raw_entities = self.extractor.extract_entities(query)
        logger.info("Extracted entities: %s", raw_entities)

        for abstract in ABSTRACT_CONCEPTS:
            if abstract in query.lower():
                logger.info("Abstract concept '%s' detected → System 2", abstract)
                direction, confidence, explanation = self.semantic_cloud.reason(query, abstract, "unknown")
                return CausalResult(
                    query=query, entity_a=abstract, entity_b="unknown", direction=direction,
                    confidence=confidence, method="Semantic reasoning (abstract concept)",
                    system_used=2, data_source="LLM World Knowledge", explanation=explanation,
                    warnings=["Result is semantic inference, not empirically validated."],
                )

        mapped = self._map_to_registry(raw_entities)

        if len(mapped) < 2:
            logger.warning("Could not ground two empirical entities → System 2")
            e_a = mapped[0][0] if mapped else "concept_a"
            e_b = "concept_b"
            direction, confidence, explanation = self.semantic_cloud.reason(query, e_a, e_b)
            return CausalResult(
                query=query, entity_a=e_a, entity_b=e_b, direction=direction, confidence=confidence,
                method="Semantic reasoning (empirical grounding failed)", system_used=2,
                data_source="LLM World Knowledge", explanation=explanation,
            )

        var_a_name, var_a_id = mapped[0]
        var_b_name, var_b_id = mapped[1]

        logger.info("⚡ System 1: Grounding '%s' ↔ '%s' to FRED...", var_a_name, var_b_name)
        series_a = fetch_fred_series(var_a_id, var_a_name)
        series_b = fetch_fred_series(var_b_id, var_b_name)

        if series_a is None or series_b is None:
            logger.warning("Data fetch failed → escalating to System 2")
            direction, confidence, explanation = self.semantic_cloud.reason(query, var_a_name, var_b_name)
            return CausalResult(
                query=query, entity_a=var_a_name, entity_b=var_b_name, direction=direction,
                confidence=confidence, method="Semantic reasoning (data unavailable)", system_used=2,
                data_source="LLM World Knowledge", explanation=explanation,
            )

        logger.info("│  Running empirical causal engine...")
        
        # 🚀 CALLING THE IMPORTED PURE MATH ENGINE (LTI)
        direction, confidence, method = robust_causal_direction(var_a_name, series_a, var_b_name, series_b)
        
        logger.info("│  ├─ Direction   : %s", direction)
        logger.info("│  ├─ Confidence  : %.1f%%", confidence * 100)

        s1_result = CausalResult(
            query=query, entity_a=var_a_name, entity_b=var_b_name, direction=direction,
            confidence=confidence, method=method, system_used=1,
            data_source=f"FRED Synthetic Sandbox ({var_a_id}, {var_b_id})",
            time_range=f"{series_a.index[0].date()} to {series_a.index[-1].date()}",
        )

        if confidence >= self.threshold:
            logger.info("Router ──→ RETURN empirical (conf %.1f%% ≥ %.0f%%)", confidence * 100, self.threshold * 100)
            return s1_result

        logger.info("Router ──→ ESCALATE to System 2 (conf %.1f%% < %.0f%%)", confidence * 100, self.threshold * 100)
        direction, confidence, explanation = self.semantic_cloud.reason(
            query, var_a_name, var_b_name, s1_hint=(s1_result.direction, s1_result.confidence),
        )

        return CausalResult(
            query=query, entity_a=var_a_name, entity_b=var_b_name, direction=direction,
            confidence=confidence, method="System 2 + empirical cross-check", system_used=2,
            data_source="LLM + FRED", explanation=explanation,
            warnings=[
                f"System 1 confidence was {s1_result.confidence:.0%} (below {self.threshold:.0%} threshold).",
                "Result combines semantic reasoning with empirical hint.",
            ],
        )

if __name__ == "__main__":
    engine = NCS_Engine(llm_provider="anthropic")
    test_queries = [
        "Does inflation cause interest rates to rise?",
        "Do oil prices affect producer prices?",
        "Does freedom lead to happiness?",
    ]
    for q in test_queries:
        result = engine.analyze(q)
        print(result)
