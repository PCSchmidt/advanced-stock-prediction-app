# Advanced Stock Price Prediction App

This is a sophisticated Python-based web application that pulls stock data using `yfinance` and performs time series analysis using various advanced models (ARIMA, SARIMA, GARCH, and LSTM) to predict future stock prices. The backend is built with `FastAPI`, and the frontend is developed using HTML, CSS, and JavaScript with Chart.js for visualization.

## Table of Contents

1. [Project Setup](#project-setup)
2. [Running the Application](#running-the-application)
3. [Using the Application](#using-the-application)
4. [Project Structure](#project-structure)
5. [Technologies Used](#technologies-used)
6. [Time Series Models](#time-series-models)
7. [Key Features](#key-features)
8. [Recent Improvements](#recent-improvements)
9. [Future Enhancements](#future-enhancements)

## Project Setup

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/advanced-stock-prediction-app.git
   cd advanced-stock-prediction-app
   ```
2. Create and activate a virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```
3. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

   Note: Make sure to use the latest requirements.txt file, as it has been updated to include all necessary dependencies and resolve potential conflicts.

## Running the Application

1. Start the FastAPI backend:

   ```
   uvicorn main:app --reload
   ```

   The API will be available at `http://127.0.0.1:8000`.
2. Open the frontend:
   Navigate to the `frontend` directory and open `index.html` in your web browser.

## Using the Application

1. Enter a stock symbol (e.g., AAPL for Apple Inc.) in the input field.
2. Select a start date and end date for historical data.
3. Choose a prediction model (ARIMA, SARIMA, GARCH, or LSTM).
4. Click the "Predict" button to fetch historical data and generate predictions.
5. The chart will display historical stock prices and predictions for the next 30 days.
6. Below the chart, you'll find information about the selected model and prediction results.

## Project Structure

- `main.py`: FastAPI backend application
- `requirements.txt`: Python dependencies
- `frontend/`
  - `index.html`: Main HTML file for the frontend
  - `styles.css`: CSS styles for the frontend
  - `script.js`: JavaScript for handling user interactions and API calls

## Technologies Used

- Backend:
  - Python
  - FastAPI
  - yfinance
  - pandas
  - numpy
  - statsmodels (ARIMA, SARIMA)
  - arch (GARCH)
  - scikit-learn
  - TensorFlow/Keras (LSTM)
- Frontend:
  - HTML5
  - CSS3
  - JavaScript
  - Chart.js

## Time Series Models

1. **ARIMA (AutoRegressive Integrated Moving Average)**:
   A statistical method for time series forecasting that combines autoregression, differencing, and moving average components.
2. **SARIMA (Seasonal ARIMA)**:
   An extension of ARIMA that supports seasonal time series data, capturing both trend and seasonal components.
3. **GARCH (Generalized AutoRegressive Conditional Heteroskedasticity)**:
   Used to model time series where the variance of the error term is not constant, making it particularly useful for financial time series with volatility clustering.
4. **LSTM (Long Short-Term Memory)**:
   A type of recurrent neural network capable of learning long-term dependencies, making it well-suited for time series prediction tasks.

## Key Features

1. **Multiple Prediction Models**: The application supports four different time series models (ARIMA, SARIMA, GARCH, and LSTM) for stock price prediction.
2. **Interactive User Interface**: The frontend provides an intuitive interface for users to input stock symbols, select date ranges, and choose prediction models.
3. **Data Visualization**: Historical and predicted stock prices are visualized using Chart.js, providing a clear representation of the data and predictions.
4. **Caching Mechanism**: The backend implements a simple in-memory cache to improve performance for frequently requested stocks. Cached results are valid for one hour.
5. **Comprehensive Error Handling**: Both frontend and backend incorporate error handling to provide informative feedback to users in case of issues.
6. **Responsive Design**: The frontend is designed to be responsive, ensuring a good user experience across different device sizes.
7. **Well-Commented Codebase**: All major components of the application (backend, frontend HTML, JavaScript, and CSS) are thoroughly commented, making it easier for developers to understand and maintain the code.

## Recent Improvements

1. **Enhanced GARCH Model**: Improved error handling and prediction generation for the GARCH model to ensure more reliable results.
2. **LSTM Model Upgrade**: Updated the LSTM model to generate a full 30-day forecast, providing more comprehensive predictions.
3. **ARIMA and SARIMA Model Enhancements**:

   - Implemented more robust error handling for ARIMA and SARIMA models.
   - Added warning suppression to handle potential warnings during model fitting.
   - Improved exception handling to provide more informative error messages for these models.
4. **Dependency Management**: Updated the requirements.txt file to resolve potential conflicts and ensure compatibility across all dependencies.
5. **Error Handling**: Implemented more specific error messages for different types of exceptions, improving debugging and user feedback.
6. **Code Documentation**: Added comprehensive comments throughout the codebase to enhance readability and maintainability.

## Future Enhancements

1. Implement model parameter optimization to improve prediction accuracy.
2. Add more advanced charting options and technical indicators for better data analysis.
3. Implement user authentication and the ability to save favorite stocks.
4. Provide model comparison and ensemble methods to leverage the strengths of different prediction models.
5. Integrate real-time data updates for live stock price tracking.
6. Implement a more robust caching system, potentially using Redis or a similar technology.
7. Add unit tests and integration tests to ensure code reliability and ease of maintenance.
8. Implement a backend database to store historical predictions and user preferences
   =================================================================================

   [... Keep the existing content up to the "Deploying to Heroku" section ...]

## Deploying to Heroku

To deploy this application to Heroku, follow these steps:

1. Make sure you have a Heroku account and the Heroku CLI installed.
2. Log in to Heroku CLI:

   ```
   heroku login
   ```
3. Create a new Heroku app:

   ```
   heroku create your-app-name
   ```
4. Set the Python buildpack:

   ```
   heroku buildpacks:set heroku/python
   ```
5. Choose one of the following deployment methods:

   Option A: Deploy from the main branch (recommended for production)

   - If you're currently on the dev branch, first merge your changes to the main branch:

     ```
     git checkout main
     git merge dev
     git push origin main
     ```
   - Then deploy to Heroku:

     ```
     git push heroku main
     ```

   Option B: Deploy directly from the dev branch

   - If you want to deploy your current dev branch without merging to main:

     ```
     git push heroku dev:main
     ```
6. Scale the web dyno:

   ```
   heroku ps:scale web=1
   ```
7. Open the deployed application in your browser:

   ```
   heroku open
   ```

Note: Make sure all the necessary files (Procfile, runtime.txt, requirements.txt) are present and committed in your current branch before deploying to Heroku.

Feel free to contribute to this project by submitting pull requests or opening issues for any bugs or feature requests.
