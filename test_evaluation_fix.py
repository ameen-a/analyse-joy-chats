#!/usr/bin/env python3
"""
Test script to demonstrate the evaluation bug and show how the fix corrects it.
"""

import pandas as pd
import numpy as np
from src.evaluation_metrics import SessionBoundaryEvaluator

def create_test_data():
    """Create test data that mimics your real data structure."""
    # Simulate data with 1.0 for session starts, NaN for non-session starts
    n_messages = 100
    
    # Create ground truth: 10% are session starts (1.0), 90% are NaN (representing 0s)
    ground_truth_raw = np.full(n_messages, np.nan)
    session_indices = np.random.choice(n_messages, size=10, replace=False)
    ground_truth_raw[session_indices] = 1.0
    
    # Create predictions: model predicts session starts on 7 of the 10 true positives
    # Plus 5 false positives
    predictions = np.zeros(n_messages)
    true_positives = np.random.choice(session_indices, size=7, replace=False)
    false_positives = np.random.choice(
        [i for i in range(n_messages) if i not in session_indices], 
        size=5, replace=False
    )
    predictions[true_positives] = 1
    predictions[false_positives] = 1
    
    return ground_truth_raw, predictions

def buggy_evaluation(ground_truth_raw, predictions):
    """Simulate the buggy evaluation logic."""
    print("=== BUGGY EVALUATION (Only evaluating on non-NaN values) ===")
    
    # This is what the buggy code was doing
    labeled_mask = pd.notna(ground_truth_raw)
    ground_truth_filtered = ground_truth_raw[labeled_mask]
    predictions_filtered = predictions[labeled_mask]
    
    # Convert to binary
    ground_truth_binary = np.array([1 if x in [1, 1.0] else 0 for x in ground_truth_filtered])
    predictions_binary = predictions_filtered.astype(int)
    
    print(f"Messages evaluated: {len(predictions_binary)} (only non-NaN labels)")
    print(f"Ground truth positives: {np.sum(ground_truth_binary)}")
    print(f"Predicted positives: {np.sum(predictions_binary)}")
    
    evaluator = SessionBoundaryEvaluator()
    results = evaluator.evaluate_comprehensive(predictions_binary, ground_truth_binary)
    evaluator.print_evaluation_report(results, "Buggy Evaluation")
    
    return results

def correct_evaluation(ground_truth_raw, predictions):
    """Simulate the corrected evaluation logic."""
    print("\n=== CORRECTED EVALUATION (Including NaN as 0s) ===")
    
    # Convert all: 1.0 -> 1, everything else (including NaN) -> 0
    ground_truth_binary = np.array([
        1 if (pd.notna(x) and x in [1, 1.0]) else 0 
        for x in ground_truth_raw
    ])
    predictions_binary = predictions.astype(int)
    
    print(f"Messages evaluated: {len(predictions_binary)} (all messages)")
    print(f"Ground truth positives: {np.sum(ground_truth_binary)}")
    print(f"Ground truth negatives: {len(ground_truth_binary) - np.sum(ground_truth_binary)}")
    print(f"Predicted positives: {np.sum(predictions_binary)}")
    print(f"Predicted negatives: {len(predictions_binary) - np.sum(predictions_binary)}")
    
    evaluator = SessionBoundaryEvaluator()
    results = evaluator.evaluate_comprehensive(predictions_binary, ground_truth_binary)
    evaluator.print_evaluation_report(results, "Corrected Evaluation")
    
    return results

def main():
    print("Demonstrating the evaluation bug in session boundary detection\n")
    
    # Create test data
    ground_truth_raw, predictions = create_test_data()
    
    print(f"Dataset: {len(ground_truth_raw)} messages")
    print(f"True session starts: {np.sum(pd.notna(ground_truth_raw))} (marked as 1.0)")
    print(f"Non-session starts: {np.sum(pd.isna(ground_truth_raw))} (marked as NaN)")
    print(f"Model predictions: {np.sum(predictions)} session starts predicted\n")
    
    # Run buggy evaluation
    buggy_results = buggy_evaluation(ground_truth_raw, predictions)
    
    # Run corrected evaluation  
    corrected_results = correct_evaluation(ground_truth_raw, predictions)
    
    # Compare key metrics
    print("\n=== COMPARISON ===")
    print(f"Precision - Buggy: {buggy_results['precision']:.3f}, Corrected: {corrected_results['precision']:.3f}")
    print(f"False Positives - Buggy: {buggy_results['confusion_matrix']['fp']}, Corrected: {corrected_results['confusion_matrix']['fp']}")
    print(f"True Negatives - Buggy: {buggy_results['confusion_matrix']['tn']}, Corrected: {corrected_results['confusion_matrix']['tn']}")
    
    print("\nThe bug caused artificially perfect precision by excluding negative examples!")

if __name__ == "__main__":
    main() 