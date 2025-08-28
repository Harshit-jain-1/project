# Stock Price Trend Prediction Using LSTM – Project Report
________________________________________
# Introduction
In the world of financial markets, predicting stock prices is a challenging yet crucial task. With the rise of machine learning and deep learning, time-series models like Long Short-Term Memory (LSTM) networks have proven effective in modeling sequential data, such as stock prices. This project aims to leverage LSTM networks to predict future stock price trends based on historical data.
________________________________________
# Tools Used
•	Programming Language: Python
•	Libraries & Frameworks:
o	yfinance – For historical stock data fetching
o	pandas, numpy – Data manipulation
o	matplotlib, seaborn – Visualization
o	scikit-learn – Data normalization
o	TensorFlow/Keras – Building and training the LSTM model
o	Streamlit (optional) – For web dashboard deployment
________________________________________
# Steps Involved in Building the Project
1.	Data Collection:
o	Used the yfinance API to fetch historical stock data (e.g., AAPL from 2015–2025).
2.	Data Preprocessing:
o	Selected the Close price for prediction.
o	Normalized the data using MinMaxScaler.
o	Created sequences (e.g., 60-day windows) for time-series training.
3.	Model Building:
o	Built an LSTM model with 2 LSTM layers, dropout layers to prevent overfitting, and a final dense output layer.
4.	Model Training and Evaluation:
o	Split the dataset into training and testing sets.
o	Trained the model for 20 epochs using MSE loss and Adam optimizer.
o	Evaluated model performance on unseen data.
5.	Visualization:
o	Plotted actual vs. predicted stock prices to assess performance.
o	Computed and plotted 50-day Moving Average and RSI for added market insight.
6.	Saving the Model:
o	Saved the trained model weights using model.save_weights('lstm_stock_model.h5') for future reuse.
7.	Optional Deployment:
o	A basic Streamlit app was prepared for visualizing predictions and market indicators interactively.
# Live Graph in Streamlit (Optional Deployment Feature)
To enhance user experience and provide real-time stock insights, a live interactive graph was added to the Streamlit dashboard. This feature allows users to visualize stock trends dynamically, including actual prices, predicted prices, moving averages, and RSI, all within a user-friendly web interface.
How It Works:
•	The Streamlit app fetches live stock data using the yfinance API.
•	Real-time charts are generated using matplotlib or plotly.
•	The LSTM model loads saved weights and predicts the next price values based on the most recent data.
________________________________________
# Conclusion
The project successfully demonstrates the use of LSTM neural networks in predicting stock prices based on historical trends. While the model does not guarantee exact price values (as markets are influenced by numerous unpredictable factors), it effectively captures general trends. Integrating technical indicators like MA and RSI provides additional insights, making this system potentially useful for traders and analysts. Further improvements can include tuning hyperparameters, incorporating news sentiment analysis, or expanding to multivariate input features.
