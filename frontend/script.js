// Global variables
let stockChart;
let componentCharts = {};

// Model descriptions
const modelDescriptions = {
    arima: "ARIMA (AutoRegressive Integrated Moving Average) is a popular statistical method for time series forecasting. It combines autoregression (AR), differencing (I), and moving average (MA) components. ARIMA models work best for data showing evidence of non-stationarity, where an initial differencing step can be applied to remove the non-stationarity.",
    sarima: "SARIMA (Seasonal ARIMA) extends the ARIMA model to support seasonal time series data. It adds seasonal terms to capture repeating patterns or cycles in the data. SARIMA is particularly useful for data that exhibits both trend and seasonality.",
    garch: "GARCH (Generalized AutoRegressive Conditional Heteroskedasticity) is used to model time series where the variance of the error term is not constant. It's particularly useful for financial time series that exhibit volatility clustering, where periods of high volatility tend to be grouped together.",
    lstm: "LSTM (Long Short-Term Memory) is a type of recurrent neural network capable of learning long-term dependencies in time series data. LSTMs are well-suited for capturing complex patterns and long-range dependencies in sequential data, making them effective for stock price prediction."
};

// Glossary terms
const glossaryTerms = {
    "Time Series": "A sequence of data points indexed in time order, often collected at regular intervals.",
    "Stationarity": "A property of a time series where statistical properties such as mean, variance, and autocorrelation are constant over time.",
    "Autocorrelation": "The correlation of a signal with a delayed copy of itself, used to find repeating patterns in a time series.",
    "Trend": "The long-term movement or change in the data over time, which can be upward, downward, or flat.",
    "Seasonality": "Regular and predictable patterns that repeat over fixed intervals of time.",
    "Volatility": "A statistical measure of the dispersion of returns for a given security or market index.",
    "Forecasting": "The process of making predictions about future values based on past and present data."
};

/**
 * Fetches stock prediction data from the backend API and updates the UI
 */
async function fetchData(retryCount = 0) {
    const symbol = document.getElementById('symbol').value;
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;
    const model = document.getElementById('model').value;

    if (!symbol || !startDate || !endDate || !model) {
        alert('Please fill in all fields');
        return;
    }

    document.getElementById('stockChart').style.display = 'none';
    document.getElementById('modelInfo').style.display = 'none';
    document.getElementById('loading').style.display = 'block';

    try {
        const response = await fetch(`http://localhost:8000/predict/${symbol}?start_date=${startDate}&end_date=${endDate}&model=${model}`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Received data:', data);
        
        if (!data.historical || !data.predictions || !Array.isArray(data.historical) || !Array.isArray(data.predictions)) {
            throw new Error('Invalid data structure received from server');
        }
        
        updateChart(data, model);
        updateModelInfo(data, model);
        updateModelComponents(data, model);
        updatePerformanceMetrics(data, model);
        
        document.getElementById('loading').style.display = 'none';
        document.getElementById('stockChart').style.display = 'block';
        document.getElementById('modelInfo').style.display = 'block';
    } catch (error) {
        console.error('Error:', error);
        if (retryCount < 3) {
            console.log(`Retrying... (${retryCount + 1}/3)`);
            setTimeout(() => fetchData(retryCount + 1), 2000);
        } else {
            alert('An error occurred while fetching data. Please try again.');
            document.getElementById('loading').style.display = 'none';
        }
    }
}

/**
 * Updates the chart with historical and predicted stock price data
 */
function updateChart(data, model) {
    const ctx = document.getElementById('stockChart').getContext('2d');

    if (stockChart) {
        stockChart.destroy();
    }

    const historicalDates = data.historical.map((_, index) => `Day ${index + 1}`);
    const futureDates = data.predictions.map((_, index) => `Future Day ${index + 1}`);

    stockChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [...historicalDates, ...futureDates],
            datasets: [{
                label: 'Historical Data',
                data: data.historical,
                borderColor: 'blue',
                fill: false
            }, {
                label: `${model.toUpperCase()} Predictions`,
                data: [...Array(data.historical.length).fill(null), ...data.predictions],
                borderColor: 'red',
                fill: false
            }]
        },
        options: {
            responsive: true,
            title: {
                display: true,
                text: `Stock Price for ${data.symbol} - ${model.toUpperCase()} Model`
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Days'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Price'
                    },
                    ticks: {
                        callback: function(value, index, values) {
                            return '$' + value.toFixed(2);
                        }
                    }
                }
            },
            tooltips: {
                mode: 'index',
                intersect: false,
                callbacks: {
                    label: function(tooltipItem, data) {
                        var label = data.datasets[tooltipItem.datasetIndex].label || '';
                        if (label) {
                            label += ': ';
                        }
                        label += '$' + tooltipItem.yLabel.toFixed(2);
                        return label;
                    }
                }
            }
        }
    });
}

/**
 * Updates the model information section with prediction results
 */
function updateModelInfo(data, model) {
    const modelInfo = document.getElementById('modelInfo');
    const lastHistorical = data.historical[data.historical.length - 1];
    const firstPrediction = data.predictions[0];
    const lastPrediction = data.predictions[data.predictions.length - 1];
    
    const percentChange = ((lastPrediction - lastHistorical) / lastHistorical) * 100;
    
    modelInfo.innerHTML = `
        <h2>${model.toUpperCase()} Model Results</h2>
        <p>${modelDescriptions[model]}</p>
        <p>Last Historical Price: $${lastHistorical ? lastHistorical.toFixed(2) : 'N/A'}</p>
        <p>First Predicted Price: $${firstPrediction ? firstPrediction.toFixed(2) : 'N/A'}</p>
        <p>Last Predicted Price: $${lastPrediction ? lastPrediction.toFixed(2) : 'N/A'}</p>
        <p>Predicted Change: ${!isNaN(percentChange) ? percentChange.toFixed(2) : 'N/A'}%</p>
    `;
}

/**
 * Updates the model explanation section based on the selected model
 */
function updateModelInfo() {
    const model = document.getElementById('model').value;
    const modelDescription = document.getElementById('modelDescription');
    modelDescription.innerHTML = `<p>${modelDescriptions[model]}</p>`;
}

/**
 * Updates the model components section with visualizations
 */
function updateModelComponents(data, model) {
    const componentCharts = document.getElementById('componentCharts');
    componentCharts.innerHTML = ''; // Clear previous charts

    // This is a simplified example. In a real application, you would need to
    // decompose the time series into trend, seasonal, and residual components.
    const trend = data.historical.map((value, index) => value * (1 + index * 0.001));
    const seasonal = data.historical.map((_, index) => Math.sin(index * 0.1) * 10);
    const residual = data.historical.map((value, index) => value - trend[index] - seasonal[index]);

    createComponentChart('trendChart', 'Trend', trend);
    createComponentChart('seasonalChart', 'Seasonal', seasonal);
    createComponentChart('residualChart', 'Residual', residual);
}

/**
 * Creates a chart for a model component
 */
function createComponentChart(id, label, data) {
    const canvas = document.createElement('canvas');
    canvas.id = id;
    document.getElementById('componentCharts').appendChild(canvas);

    new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: {
            labels: data.map((_, index) => `Day ${index + 1}`),
            datasets: [{
                label: label,
                data: data,
                borderColor: getRandomColor(),
                fill: false
            }]
        },
        options: {
            responsive: true,
            title: {
                display: true,
                text: `${label} Component`
            }
        }
    });
}

/**
 * Updates the performance metrics section
 */
function updatePerformanceMetrics(data, model) {
    const metricsContainer = document.getElementById('metricsContainer');
    
    // These are placeholder metrics. In a real application, you would calculate these based on the model's performance.
    const mse = Math.random() * 10;
    const mae = Math.random() * 5;
    const mape = Math.random() * 10;

    metricsContainer.innerHTML = `
        <p>Mean Squared Error (MSE): ${mse.toFixed(4)}</p>
        <p>Mean Absolute Error (MAE): ${mae.toFixed(4)}</p>
        <p>Mean Absolute Percentage Error (MAPE): ${mape.toFixed(2)}%</p>
    `;
}

/**
 * Populates the glossary section
 */
function populateGlossary() {
    const glossaryList = document.getElementById('glossaryList');
    for (const [term, definition] of Object.entries(glossaryTerms)) {
        glossaryList.innerHTML += `
            <dt>${term}</dt>
            <dd>${definition}</dd>
        `;
    }
}

/**
 * Generates a random color for charts
 */
function getRandomColor() {
    const letters = '0123456789ABCDEF';
    let color = '#';
    for (let i = 0; i < 6; i++) {
        color += letters[Math.floor(Math.random() * 16)];
    }
    return color;
}

/**
 * Toggles dark mode
 */
function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');
    localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
}

/**
 * Loads user's dark mode preference
 */
function loadDarkModePreference() {
    const darkMode = localStorage.getItem('darkMode');
    if (darkMode === 'true') {
        document.body.classList.add('dark-mode');
        document.getElementById('darkModeToggle').checked = true;
    }
}

// Set default dates, populate glossary, and set up dark mode when the DOM content is loaded
document.addEventListener('DOMContentLoaded', (event) => {
    const today = new Date();
    const oneYearAgo = new Date(today.getFullYear() - 1, today.getMonth(), today.getDate());
    
    document.getElementById('startDate').value = oneYearAgo.toISOString().split('T')[0];
    document.getElementById('endDate').value = today.toISOString().split('T')[0];

    populateGlossary();
    updateModelInfo();

    // Set up dark mode toggle
    const darkModeToggle = document.getElementById('darkModeToggle');
    darkModeToggle.addEventListener('change', toggleDarkMode);

    // Load user's dark mode preference
    loadDarkModePreference();
});