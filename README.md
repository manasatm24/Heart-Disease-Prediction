# Heart Disease Prediction Web App

A machine learning-powered web application for predicting heart disease risk based on medical parameters. Built as a mini project using Flask and scikit-learn.

## Features

- **User Authentication**: Secure login and registration system
- **Heart Disease Prediction**: Machine learning model predicts risk based on 13 medical features
- **Interactive Web Interface**: Clean, responsive UI built with Bootstrap
- **Prediction History**: View logs of all predictions made
- **PDF Reports**: Generate downloadable PDF reports of predictions
- **Database Storage**: SQLite database for storing user data and predictions

## Technologies Used

- **Backend**: Flask (Python web framework)
- **Machine Learning**: scikit-learn (Random Forest classifier)
- **Database**: SQLite
- **Frontend**: HTML, CSS, Bootstrap 5
- **PDF Generation**: ReportLab
- **Data Processing**: pandas, NumPy

## Dataset

The model is trained on the Cleveland Heart Disease dataset, which includes:
- Age, Sex, Chest Pain Type
- Resting Blood Pressure, Serum Cholesterol
- Fasting Blood Sugar, Resting ECG
- Maximum Heart Rate, Exercise Induced Angina
- ST Depression, Slope of Peak Exercise ST Segment
- Number of Major Vessels, Thalassemia

## Installation

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install flask scikit-learn pandas numpy werkzeug reportlab
   ```

3. **Train the model** (if not already trained):
   ```bash
   python train_model.py
   ```
   This will create `heart-disease-prediction-knn-model.pkl`

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Open your browser** and navigate to `http://localhost:5000`

## Usage

1. **Register**: Create a new account or login with existing credentials
2. **Predict**: Fill out the prediction form with medical parameters
3. **Results**: View prediction results with risk probability
4. **Logs**: Access prediction history from the logs page
5. **Reports**: Download PDF reports of individual predictions

## Project Structure

```
AK MINI PROJECT/
├── app.py                 # Main Flask application
├── train_model.py         # Model training script
├── heart_cleveland_upload.csv  # Dataset
├── heart_predictions.db   # SQLite database (created automatically)
├── heart-disease-prediction-knn-model.pkl  # Trained model
├── scripts/
│   ├── insert_test.py     # Test data insertion
│   └── inspect_db.py      # Database inspection
├── static/
│   └── style.css          # Additional CSS styles
└── templates/
    ├── index.html         # Landing page
    ├── login.html         # Login page
    ├── register.html      # Registration page
    ├── main.html          # Prediction form
    ├── result.html        # Results page
    └── logs.html          # Prediction logs
```

## Model Performance

The Random Forest model achieves approximately 85-90% accuracy on the test set (results may vary based on random seed).

## Security Notes

- Change the `app.secret_key` in `app.py` before deploying to production
- This is a demonstration project - implement proper security measures for real-world use
- Passwords are hashed using Werkzeug's security functions

## Contributing

This is an academic mini project. Feel free to fork and modify for educational purposes.

## License

Academic project - no specific license applied.
