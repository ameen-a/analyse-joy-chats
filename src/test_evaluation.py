import pandas as pd
import numpy as np
from datetime import datetime
from evaluation_metrics import SessionBoundaryEvaluator

def test_evaluation_pipeline():
    """Test the evaluation pipeline with the new session detection logic."""
    
    # load the test results
    df = pd.read_excel('../data/test_session_detection_results_20250707_104921.xlsx')
    
    print("=== TESTING EVALUATION PIPELINE ===")
    print(f"Total messages: {len(df)}")
    print(f"Messages with predictions: {df['is_session_start_pred'].notna().sum()}")
    print(f"Messages without predictions: {df['is_session_start_pred'].isna().sum()}")
    
    # check ground truth distribution
    if 'is_session_start' in df.columns:
        # convert ground truth labels: 1.0 or '[START]' -> 1, everything else (including NaN) -> 0
        ground_truth = df['is_session_start'].apply(
            lambda x: 1 if (pd.notna(x) and x in [1, 1.0, '[START]']) else 0
        ).values
        
        predictions = df['is_session_start_pred'].astype(int).values
        
        print(f"\nGround truth distribution: {np.sum(ground_truth)} positives, {len(ground_truth) - np.sum(ground_truth)} negatives")
        print(f"Prediction distribution: {np.sum(predictions)} positives, {len(predictions) - np.sum(predictions)} negatives")
        
        # run evaluation
        evaluator = SessionBoundaryEvaluator()
        evaluation_results = evaluator.evaluate_comprehensive(predictions, ground_truth)
        evaluator.print_evaluation_report(evaluation_results, "TestSessionDetector")
        
        # save evaluation results
        import json
        eval_path = f'../data/test_evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(eval_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        print(f"\nEvaluation results saved to {eval_path}")
        
        # check if first 9 positions are properly handled
        print(f"\nFirst 10 positions:")
        print(f"Ground truth: {ground_truth[:10]}")
        print(f"Predictions:  {predictions[:10]}")
        
        # check for any positions that have ground truth but no predictions
        missing_predictions = (ground_truth == 1) & (predictions == 0)
        print(f"\nPositions with ground truth=1 but prediction=0: {np.sum(missing_predictions)}")
        if np.sum(missing_predictions) > 0:
            missing_positions = np.where(missing_predictions)[0]
            print(f"First 10 missing positions: {missing_positions[:10]}")
    
    else:
        print("No ground truth labels found in dataset")

if __name__ == "__main__":
    test_evaluation_pipeline() 