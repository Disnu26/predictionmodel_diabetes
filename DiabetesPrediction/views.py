import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from django.shortcuts import render
def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    # Load the dataset
    data = pd.read_csv(r"C:\\Users\\disnu\\Downloads\\diabetes\\diabetes.csv")
    x = data.drop("Outcome", axis=1)
    y = data['Outcome']

    # Standardize the data
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)  # x_scaled is now a NumPy array

    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2)

    # Train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    # Retrieve and validate input values
    try:
        input_data = {
            'Pregnancies': float(request.GET.get('n1', 0)),
            'Glucose': float(request.GET.get('n2', 0)),
            'BloodPressure': float(request.GET.get('n3', 0)),
            'SkinThickness': float(request.GET.get('n4', 0)),
            'Insulin': float(request.GET.get('n5', 0)),
            'BMI': float(request.GET.get('n6', 0)),
            'DiabetesPedigreeFunction': float(request.GET.get('n7', 0)),
            'Age': float(request.GET.get('n8', 0))
        }
    except ValueError:
        result1 = "Invalid input. Please enter valid numbers."
        return render(request, "predict.html", {"result": result1})

    # Convert input_data to DataFrame with same columns as training data
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)  # Standardize the input data

    # Make prediction
    pred = model.predict(input_scaled)

    # Determine result
    result1 = "Positive" if pred == [1] else "Negative"

    return render(request, "predict.html", {"result": result1})
