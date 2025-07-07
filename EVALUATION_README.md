# Enhanced Session Boundary Evaluation Metrics

This document explains the new evaluation metrics system designed for session boundary detection with noisy labels.

## Problem Context

Session boundary detection is a binary sequence labelling task where you predict whether each message starts a new conversation session. However, labelling these boundaries can be subjective, leading to labels that might be "correct" but off by 1-2 positions.

Traditional accuracy metrics are too strict for this problem - a prediction that's off by 1 message is much better than one that's completely wrong.

## New Metrics Added

### 1. **Within-k Accuracy Metrics**
- `within_1_accuracy`: Correct if prediction is within 1 position of true boundary
- `within_2_accuracy`: Correct if prediction is within 2 positions of true boundary  
- `within_3_accuracy`: Correct if prediction is within 3 positions of true boundary
- `within_5_accuracy`: Correct if prediction is within 5 positions of true boundary

**Use case**: Account for subjective labelling where the exact boundary position might vary.

### 2. **Boundary Distance Metrics**
- `mean_distance`: Average distance between predicted and true boundaries
- `median_distance`: Median distance between predicted and true boundaries
- `boundary_precision`: Fraction of predicted boundaries that are close to true boundaries
- `boundary_recall`: Fraction of true boundaries that have nearby predictions

**Use case**: Understand how far off your predictions typically are.

### 3. **Windowed Accuracy**
- `windowed_accuracy`: Uses sliding windows around true boundaries to check if any prediction falls within the window

**Use case**: Very forgiving metric for detecting general vicinity of session changes.

### 4. **Standard Classification Metrics**
- `exact_accuracy`: Traditional exact match accuracy
- `precision`, `recall`, `f1_score`: Standard binary classification metrics
- Confusion matrix breakdown

## Quick Start

### Basic Usage
```python
from evaluation_metrics import evaluate_session_predictions
import numpy as np

# your binary arrays
predictions = np.array([1, 0, 0, 1, 0, 1, 0, 0])
ground_truth = np.array([1, 0, 0, 0, 1, 1, 0, 0])

# get comprehensive evaluation
results = evaluate_session_predictions(
    predictions=predictions,
    ground_truth=ground_truth,
    tolerance_levels=[1, 2, 3, 5],
    verbose=True  # prints detailed report
)
```

### Advanced Usage
```python
from evaluation_metrics import SessionBoundaryEvaluator

evaluator = SessionBoundaryEvaluator(tolerance_levels=[1, 2, 3, 5])
results = evaluator.evaluate_comprehensive(predictions, ground_truth)

# print report
evaluator.print_evaluation_report(results, "My Model")

# visualise boundaries
evaluator.plot_boundary_comparison(predictions, ground_truth)
```

## Integration with Your Session Detector

The evaluation is now automatically integrated into your `session_detector.py`. When you run it, you'll get:

1. **Comprehensive evaluation report** with all metrics
2. **JSON file** with detailed results saved to `/data/`
3. **Tolerance-based metrics** that account for label noise

## Interpreting the Metrics

### When Labels Are Noisy
- **Focus on within-k metrics** rather than exact accuracy
- **within_1_accuracy** and **within_2_accuracy** are most important
- **Mean distance < 2** indicates good boundary detection

### When Labels Are Clean
- **Exact accuracy** and **F1-score** are primary metrics
- **Within-k metrics** should be close to exact accuracy

### Example Interpretation
```
Exact Accuracy:    0.750  ← Traditional metric (too strict for noisy labels)
Within-1 Accuracy: 0.900  ← Much better! Accounts for ±1 position errors
Within-2 Accuracy: 0.950  ← Even better! Accounts for ±2 position errors
Mean Distance:     1.2    ← On average, predictions are 1.2 positions off
```

## Running the Demo

Test the new metrics with the provided example:

```bash
cd src/
python example_evaluation.py
```

This will show you:
- How different metrics behave with various error types
- Visualisation of predicted vs actual boundaries
- Integration examples for your workflow

## Key Benefits

1. **Accounts for Label Noise**: Within-k metrics are forgiving of small positional errors
2. **Better Model Comparison**: More nuanced evaluation than simple accuracy  
3. **Detailed Insights**: Distance metrics show prediction quality patterns
4. **Visual Analysis**: Plots help understand model behaviour
5. **Automatic Integration**: Works seamlessly with your existing pipeline

## Recommended Usage

For your session detection task with noisy labels:

1. **Primary metrics**: `within_1_accuracy`, `within_2_accuracy`
2. **Secondary metrics**: `mean_distance`, `boundary_precision`, `boundary_recall`
3. **Tolerance levels**: `[1, 2, 3, 5]` covers typical labelling variations
4. **Threshold for "good" model**: `within_2_accuracy > 0.8` and `mean_distance < 3`

This evaluation system gives you much better insight into your model's performance while accounting for the inherent subjectivity in session boundary labelling. 