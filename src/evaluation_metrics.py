import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class SessionBoundaryEvaluator:
    """
    Evaluation metrics for session boundary detection with tolerance for noisy labels.
    
    Designed for binary sequence labeling where exact boundary placement may vary
    due to subjective labeling, but approximate boundaries are valuable.
    """
    
    def __init__(self, tolerance_levels: List[int] = None):
        """
        Initialise evaluator with tolerance levels for within-k metrics.
        
        Args:
            tolerance_levels: List of tolerance distances (e.g., [1, 2, 3])
        """
        self.tolerance_levels = tolerance_levels or [1, 2, 3, 5]
        
    def exact_accuracy(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Standard exact match accuracy."""
        return (predictions == ground_truth).mean()
    
    def within_k_accuracy(self, 
                         predictions: np.ndarray, 
                         ground_truth: np.ndarray, 
                         k: int,
                         message_ids: Optional[np.ndarray] = None) -> float:
        """
        Calculate within-k accuracy for boundary predictions.
        
        A prediction is correct if there's a true boundary within k positions
        of the predicted boundary (and vice versa).
        
        Args:
            predictions: Binary array of predicted boundaries
            ground_truth: Binary array of true boundaries  
            k: Tolerance distance
            message_ids: Optional message IDs for channel grouping
            
        Returns:
            Within-k accuracy score
        """
        pred_boundaries = np.where(predictions == 1)[0]
        true_boundaries = np.where(ground_truth == 1)[0]
        
        if len(pred_boundaries) == 0 and len(true_boundaries) == 0:
            return 1.0
        
        if len(pred_boundaries) == 0 or len(true_boundaries) == 0:
            return 0.0
        
        # For each predicted boundary, check if there's a true boundary within k
        pred_correct = 0
        for pred_idx in pred_boundaries:
            distances = np.abs(true_boundaries - pred_idx)
            if np.min(distances) <= k:
                pred_correct += 1
        
        # For each true boundary, check if there's a predicted boundary within k
        true_correct = 0
        for true_idx in true_boundaries:
            distances = np.abs(pred_boundaries - true_idx)
            if np.min(distances) <= k:
                true_correct += 1
        
        # Harmonic mean of precision and recall style metrics
        if len(pred_boundaries) == 0:
            pred_precision = 0
        else:
            pred_precision = pred_correct / len(pred_boundaries)
            
        if len(true_boundaries) == 0:
            true_recall = 0
        else:
            true_recall = true_correct / len(true_boundaries)
        
        if pred_precision + true_recall == 0:
            return 0.0
        
        return 2 * (pred_precision * true_recall) / (pred_precision + true_recall)
    
    def boundary_distance_metrics(self, 
                                 predictions: np.ndarray, 
                                 ground_truth: np.ndarray) -> Dict[str, float]:
        """
        Calculate distance-based metrics between predicted and true boundaries.
        
        Returns:
            Dictionary with mean/median distances and other boundary-specific metrics
        """
        pred_boundaries = np.where(predictions == 1)[0]
        true_boundaries = np.where(ground_truth == 1)[0]
        
        if len(pred_boundaries) == 0 or len(true_boundaries) == 0:
            return {
                'mean_distance': np.inf if len(pred_boundaries) != len(true_boundaries) else 0,
                'median_distance': np.inf if len(pred_boundaries) != len(true_boundaries) else 0,
                'boundary_count_diff': abs(len(pred_boundaries) - len(true_boundaries)),
                'boundary_precision': 0 if len(pred_boundaries) > 0 else 1,
                'boundary_recall': 0 if len(true_boundaries) > 0 else 1
            }
        
        # Find closest matches
        distances = []
        matched_true = set()
        
        # For each predicted boundary, find closest true boundary
        for pred_idx in pred_boundaries:
            true_distances = np.abs(true_boundaries - pred_idx)
            closest_true_idx = np.argmin(true_distances)
            closest_distance = true_distances[closest_true_idx]
            distances.append(closest_distance)
            matched_true.add(closest_true_idx)
        
        # Count how many true boundaries were matched
        boundary_precision = len([d for d in distances if d <= max(self.tolerance_levels)]) / len(pred_boundaries)
        boundary_recall = len(matched_true) / len(true_boundaries)
        
        return {
            'mean_distance': np.mean(distances),
            'median_distance': np.median(distances), 
            'boundary_count_diff': abs(len(pred_boundaries) - len(true_boundaries)),
            'boundary_precision': boundary_precision,
            'boundary_recall': boundary_recall,
            'min_distance': np.min(distances),
            'max_distance': np.max(distances)
        }
    
    def windowed_accuracy(self, 
                         predictions: np.ndarray, 
                         ground_truth: np.ndarray,
                         window_size: int = 5) -> float:
        """
        Calculate accuracy using sliding windows around boundaries.
        
        More forgiving metric that looks at whether the model captures
        the general vicinity of session changes.
        """
        correct_windows = 0
        total_windows = 0
        
        # Create windows around each true boundary
        true_boundaries = np.where(ground_truth == 1)[0]
        
        for boundary_idx in true_boundaries:
            start = max(0, boundary_idx - window_size // 2)
            end = min(len(predictions), boundary_idx + window_size // 2 + 1)
            
            # Check if there's any predicted boundary in this window
            window_predictions = predictions[start:end]
            if np.any(window_predictions == 1):
                correct_windows += 1
            total_windows += 1
        
        return correct_windows / total_windows if total_windows > 0 else 1.0
    
    def evaluate_comprehensive(self, 
                             predictions: np.ndarray, 
                             ground_truth: np.ndarray,
                             message_ids: Optional[np.ndarray] = None,
                             return_details: bool = False) -> Dict:
        """
        Run comprehensive evaluation with all metrics.
        
        Args:
            predictions: Binary array of predicted session starts
            ground_truth: Binary array of true session starts
            message_ids: Optional message identifiers for grouping
            return_details: Whether to return detailed breakdown
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        
        # Basic classification metrics
        accuracy = self.exact_accuracy(predictions, ground_truth)
        
        # Handle edge case where no positive examples exist
        if np.sum(ground_truth) == 0 or np.sum(predictions) == 0:
            precision = recall = f1 = 0.0
        else:
            precision, recall, f1, _ = precision_recall_fscore_support(
                ground_truth, predictions, average='binary', zero_division=0
            )
        
        # Within-k accuracies
        within_k_scores = {}
        for k in self.tolerance_levels:
            within_k_scores[f'within_{k}_accuracy'] = self.within_k_accuracy(
                predictions, ground_truth, k, message_ids
            )
        
        # Boundary-specific metrics
        boundary_metrics = self.boundary_distance_metrics(predictions, ground_truth)
        
        # Windowed accuracy
        windowed_acc = self.windowed_accuracy(predictions, ground_truth)
        
        # Confusion matrix details
        cm = confusion_matrix(ground_truth, predictions, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        results = {
            # Standard metrics
            'exact_accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            
            # Tolerance-based metrics
            **within_k_scores,
            'windowed_accuracy': windowed_acc,
            
            # Boundary-specific metrics
            **boundary_metrics,
            
            # Additional stats
            'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'predicted_boundaries': np.sum(predictions),
            'actual_boundaries': np.sum(ground_truth),
            
            # Confusion matrix
            'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
        }
        
        if return_details:
            results['predictions'] = predictions
            results['ground_truth'] = ground_truth
            results['boundary_positions'] = {
                'predicted': np.where(predictions == 1)[0].tolist(),
                'actual': np.where(ground_truth == 1)[0].tolist()
            }
        
        return results
    
    def print_evaluation_report(self, results: Dict, model_name: str = "Model"):
        """Print a comprehensive evaluation report."""
        print(f"\n{'='*60}")
        print(f"SESSION BOUNDARY EVALUATION REPORT - {model_name}")
        print(f"{'='*60}")
        
        print(f"\n📊 STANDARD CLASSIFICATION METRICS:")
        print(f"   Exact Accuracy:    {results['exact_accuracy']:.3f}")
        print(f"   Precision:         {results['precision']:.3f}")
        print(f"   Recall:            {results['recall']:.3f}")
        print(f"   F1-Score:          {results['f1_score']:.3f}")
        
        print(f"\n🎯 TOLERANCE-BASED METRICS:")
        for k in self.tolerance_levels:
            key = f'within_{k}_accuracy'
            if key in results:
                print(f"   Within-{k} Accuracy: {results[key]:.3f}")
        print(f"   Windowed Accuracy: {results['windowed_accuracy']:.3f}")
        
        print(f"\n📏 BOUNDARY DISTANCE METRICS:")
        print(f"   Mean Distance:     {results['mean_distance']:.2f}")
        print(f"   Median Distance:   {results['median_distance']:.2f}")
        print(f"   Boundary Precision:{results['boundary_precision']:.3f}")
        print(f"   Boundary Recall:   {results['boundary_recall']:.3f}")
        
        print(f"\n📈 BOUNDARY STATISTICS:")
        print(f"   Predicted Boundaries: {results['predicted_boundaries']}")
        print(f"   Actual Boundaries:    {results['actual_boundaries']}")
        print(f"   Count Difference:     {results['boundary_count_diff']}")
        
        cm = results['confusion_matrix']
        print(f"\n🔢 CONFUSION MATRIX:")
        print(f"   True Positives:  {cm['tp']:4d}   False Positives: {cm['fp']:4d}")
        print(f"   False Negatives: {cm['fn']:4d}   True Negatives:  {cm['tn']:4d}")
        
        print(f"\n{'='*60}")
    
    def plot_boundary_comparison(self, 
                               predictions: np.ndarray, 
                               ground_truth: np.ndarray,
                               sample_range: Tuple[int, int] = None,
                               figsize: Tuple[int, int] = (15, 6)):
        """
        Visualise predicted vs actual boundaries for a sample range.
        
        Args:
            predictions: Binary array of predicted boundaries
            ground_truth: Binary array of true boundaries  
            sample_range: (start, end) indices to plot (None for full range)
            figsize: Figure size tuple
        """
        if sample_range:
            start, end = sample_range
            pred_sample = predictions[start:end]
            truth_sample = ground_truth[start:end]
            x_positions = np.arange(start, end)
        else:
            pred_sample = predictions
            truth_sample = ground_truth
            x_positions = np.arange(len(predictions))
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Plot ground truth boundaries
        boundary_positions = x_positions[truth_sample == 1]
        ax1.vlines(boundary_positions, 0, 1, colors='green', linewidth=2, label='True Boundaries')
        ax1.set_ylabel('Ground Truth')
        ax1.set_ylim(-0.1, 1.1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot predicted boundaries
        pred_positions = x_positions[pred_sample == 1]
        ax2.vlines(pred_positions, 0, 1, colors='red', linewidth=2, label='Predicted Boundaries')
        ax2.set_ylabel('Predictions')
        ax2.set_ylim(-0.1, 1.1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot comparison
        ax3.vlines(boundary_positions, -0.1, 0.9, colors='green', linewidth=2, alpha=0.7, label='True')
        ax3.vlines(pred_positions, 0.1, 1.1, colors='red', linewidth=2, alpha=0.7, label='Predicted')
        ax3.set_ylabel('Comparison')
        ax3.set_xlabel('Message Position')
        ax3.set_ylim(-0.2, 1.2)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def evaluate_session_predictions(predictions: np.ndarray,
                               ground_truth: np.ndarray, 
                               tolerance_levels: List[int] = None,
                               verbose: bool = True) -> Dict:
    """
    Convenience function for quick evaluation.
    
    Args:
        predictions: Binary array of predicted session boundaries
        ground_truth: Binary array of true session boundaries
        tolerance_levels: List of tolerance levels for within-k metrics
        verbose: Whether to print detailed report
        
    Returns:
        Dictionary with all evaluation metrics
    """
    evaluator = SessionBoundaryEvaluator(tolerance_levels)
    results = evaluator.evaluate_comprehensive(predictions, ground_truth)
    
    if verbose:
        evaluator.print_evaluation_report(results)
    
    return results 