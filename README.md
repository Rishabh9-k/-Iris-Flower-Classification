# Iris-Flower-Classification>


Here is a complete walkthrough and Python code to train, evaluate, and classify the Iris flowers using the dataset you provided (Iris.csv). We will use Scikit-learn, a robust and widely used library for this task.

Since your data includes an unnecessary Id column, we will make sure to drop it so it doesn't interfere with the model training.


## Breakdown of the Process
Data Preparation: The script reads your Iris.csv file, drops the irrelevant Id column, and separates the features (SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm) from the label (Species).

Stratified Splitting: stratify=y ensures that both your training and testing sets get an equal, balanced distribution of all three species (Setosa, Versicolor, and Virginica).

Feature Scaling: Standardizing features ensures that features with larger numerical ranges don't unfairly dominate the model's math.

Model Choice: A Random Forest Classifier is used here because it is incredibly robust, requires almost no hyperparameter tuning to get great results, and handles tabular datasets like Iris effortlessly.

Evaluation: The script outputs the overall accuracy, a classification report (showing precision, recall, and F1-score for each species individually), and a confusion matrix to highlight exactly where the model might have misclassified any samples.
