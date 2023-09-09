import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to perform logistic regression and display results
def perform_logistic_regression():
    # Get selected features
    selected_features = [var.get() for var in feature_vars]

    # Load a sample dataset (you can replace this with your own data)
    data = load_iris()
    X = data.data[:, selected_features]  # Selected predictor variables
    y = (data.target == 2).astype(int)  # Binary outcome (e.g., 0 or 1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the logistic regression model
    model = LogisticRegression()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Display results in a messagebox
    messagebox.showinfo("Logistic Regression Results", f"Accuracy: {accuracy:.2f}")

# Create the main window
root = tk.Tk()
root.title("Logistic Regression Calibration")

# Create a frame for the feature selection
feature_frame = ttk.LabelFrame(root, text="Select Features")
feature_frame.grid(row=0, column=0, padx=10, pady=10, sticky="w")

# Load a sample dataset to get feature names (you can replace this with your own data)
data = load_iris()
feature_names = data.feature_names

# Create variables to store the state of feature selection
feature_vars = []
for i, feature_name in enumerate(feature_names):
    var = tk.IntVar()
    ttk.Checkbutton(feature_frame, text=feature_name, variable=var).grid(row=i, column=0, sticky="w")
    feature_vars.append(var)

# Create a button to perform logistic regression
calibrate_button = ttk.Button(root, text="Calibrate Logistic Regression", command=perform_logistic_regression)
calibrate_button.grid(row=1, column=0, padx=10, pady=10)

# Start the GUI main loop
root.mainloop()
