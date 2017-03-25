# Kaktusio
Go-Search - Machine learning powered Heatmap to predict supply-demand gap for ride hailing services across city.

Play Go-Search.mp4 video file to peek the result! 

### Everyone hates waiting... or, even worse, surge pricing.
In ride-hailing service, nobody likes to get a price increase, especially in the time when you need the service the most. The idea behind surge pricing is to attract more drivers go down on roads when the demand arises by offering them a better price. It is indeed a sensible approach from a basic supply-demand model to keep the market in equilibrium. But on the other side, it is also a turn-off for some of customers.

### So, what if you can pre-position the supply before the demand arises and avoid surge price?
We intent to build Go-Search, a live heatmap which shows live prediction of supply-demand gap across area in the city. By viewing the heatmap, drivers can search which area that will need their service the most. Therefore, they can make decision to reposition themselves before the demands arises and maximize their revenues. In this way, the supply will be ready for the demand in advance so the waiting time can be minimized and the surge pricing never even has to happen.

In order to achieve this, we attempt build a predictive model powered by machine learning techniques that analyze historical demand pattern and current drivers position to predict the the supply and demand gap, 1 hour in advance. This model will serve as a backend to a city heatmap showing a supply-demand gap prediction for the next hour.

## Data
Go-Search requires two data as raw input:
1. Demand Data: contains customer request for ride hailing services in specific time and location (lat, long)
2. Supply Data: contains iddle active drivers in specific time and location (lat, long)

Since we don't have direct access to that kind of data, we will create a dummy data, processed from Kaggle datasets (https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i). The data provides a dataset describing the trajectories for all the 442 taxis running in the city of Porto, in Portugal (around 75km2 area). We treat the taxi's starting point as demand data (customer requests), and taxi end point (taxi availability) as supply data.


##Our Approach
###Data Processing
We process the data by dividing Porto city into grids, with 1km2 wide for each. We then divide the day into time slots with 10 minutes length. In each location and time slots, we define the number of customer request (demand) and the number of iddle drivers (supply).

Every 10 minutes, we attempt to predict supply and demand gap for the next 30 minutes for all location. Thus the heatmap will be updated with a new forecast for every 10 minutes. 

We use Random Forest Regressor from Scikit-Learn library to serve as machine learning model. This model is chosen due to its relatively simple and powerful to learn the inter-relationship (linearity and nonlinearity) between features.

### Training and Test Results
We use Root Mean Squared Error as evaluation metric. This metric measures the error between the forecasted and actual value. The result shows that our model can give RMSE lower than 1 (around 0.7 - 0.8) which means in average the model can predict with error of 1 request gap per location. This is a quite good result from RMSE point of view. However from geo-spatial perspective this metric may misleadig since they only measure the aggregate value but not the distribution form across the area.

### Conclusion
The data processing and modeling approach for Go-Search has shown a good initial result. This model can be developed further to improve its accuracy. For example by testing the model with a real ride-hailing service data, such as from Go-Jek, Grab, and Uber. One can also develop the model to be able to prescript the drivers to move to specific area to maximize revenue and optimize the supply and demand for entire city.

##Running the Code
###Dependency
- Python 2.7
- Numpy
- Scipy
- Matplotlib
- Pandas
- Seaborn
- Sklearn

###Steps to Run
1. git clone (this repository)
2. install dependency: pip install -r requirements.txt
3. run data_process.py: python data_process.py
	- output: training_data.csv and test_data.csv in folder processed_data
4. run prediction_model.py: python prediction_model.py
	- output: test_result.csv in folder processed_data
5. run print_result.py: python print_result.py
	- output: series of image containing model visualization 
