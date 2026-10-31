
# Himalayan Expedition Success Prediction

INT234 Predictive Analytics project by Ruhani (Reg. No. 12405148), Lovely Professional University.

## About
Predicts whether a Himalayan expedition reaches the summit (classification) and how long it lasts (regression), using 9,747 expeditions from 1905 to 2019 (duplicates removed).

## Models
- Classification: Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM
- Regression: Simple, Multiple and Polynomial (degree 2 and 3)

## Key results
- SVM was the best classifier on the test set: 68.0% accuracy, AUC 0.735
- Oxygen use is the strongest predictor: 79.2% success with it versus 48.2% without
- Multiple linear regression predicts duration best: R2 = 0.407

## Files
- CA2_project.py: full analysis and models
- extra_experiments.py: cross-validation and sensitivity checks
- himalayan_joined.csv: dataset
- Report and IEEE paper (PDF)

## Run
pip install pandas numpy matplotlib seaborn scikit-learn
python CA2_project.py
