# Classification Rule Simulator

A local Flask + HTML/CSS/JavaScript classification rule-mining dashboard.

## Features
- CSV upload and dataset preview
- Basic preprocessing summary
- Automatic high-cardinality ID-column exclusion
- Decision Tree and Random Forest
- Decision Tree criteria: Gini, Entropy, Log Loss
- Test-size and max-depth controls
- Classification rules with confidence and support
- Macro Accuracy/Precision/Recall/F1 evaluation
- Confusion matrix
- Feature importance
- New-record prediction with calibrated class probabilities (avoids misleading 0%/100% tree confidence)
- Visualization tab comparing criterion, test size, max depth and algorithm

## Run on Windows
Double-click `run_windows.bat` or run:

```bat
python -m pip install -r requirements.txt
python backend\app.py
```

Then open http://127.0.0.1:5000

## Recommended student dataset settings
For the included 50k student dataset:
- Target: `Performance`
- Algorithm: Decision Tree
- Criterion: Gini
- Test size: 20%
- Max depth: 5
- Min samples split: 10

The supplied dataset is intentionally noisy and multi-factor, so it is designed to produce a realistic, non-perfect classification result rather than a 100% score.


## Prediction probability behavior
The Decision Tree is still used for the predicted class, rules, and held-out accuracy. For a new student, the displayed class probabilities are produced by a separate sigmoid-calibrated probability model. This is important because a Decision Tree can create a pure leaf and report 100% raw leaf probability even when the overall test accuracy is much lower. The calibration step makes the displayed probabilities reflect model uncertainty more realistically without changing the reported test accuracy.
