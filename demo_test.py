#!/usr/bin/env python3
"""
NCS Engine v2.0 — Comprehensive Demo & Integration Test
"""

import json
import sys
from typing import List
from ncs_core import NCS_Engine, CausalResult

def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")

def print_result_summary(result: CausalResult) -> None:
    system_label = "⚡ SYSTEM 1 (Empirical)" if result.system_used == 1 else "🧠 SYSTEM 2 (Semantic)"
    print(f"  Query           : {result.query}")
    print(f"  Variables       : {result.entity_a} ↔ {result.entity_b}")
    print(f"  Direction       : {result.direction}")
    print(f"  Confidence      : {result.confidence:.1%}")
    print(f"  System Used     : {system_label}")
    print()

def run_test_suite(llm_provider: str = "anthropic") -> None:
    print_header("NCS ENGINE v2.0 — INTEGRATION TEST SUITE")
    engine = NCS_Engine(llm_provider=llm_provider, confidence_threshold=0.80)
    
    print_header("TEST GROUP 1: System 1 Empirical Reflex")
    system1_queries = ["Does inflation cause interest rates to rise?", "Do oil prices affect producer prices?"]
    for i, query in enumerate(system1_queries, 1):
        print(f"[Test 1.{i}] {query}")
        result = engine.analyze(query)
        print_result_summary(result)
        
    print_header("TEST GROUP 2: System 2 Semantic Cloud")
    system2_direct_queries = ["Does freedom lead to happiness?", "Can love cause trust?"]
    for i, query in enumerate(system2_direct_queries, 1):
        print(f"[Test 2.{i}] {query}")
        result = engine.analyze(query)
        print_result_summary(result)

    print_header("TEST GROUP 3: Router Logic")
    router_queries = ["Does unemployment affect inflation?"]
    for i, query in enumerate(router_queries, 1):
        print(f"[Test 3.{i}] {query}")
        result = engine.analyze(query)
        print_result_summary(result)
        
    print_header("TEST SUMMARY COMPLETE")

if __name__ == "__main__":
    provider = sys.argv[1].lower() if len(sys.argv) > 1 else "anthropic"
    run_test_suite(llm_provider=provider)