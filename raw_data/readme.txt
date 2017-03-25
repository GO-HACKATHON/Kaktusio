Go-Search requires two data as raw input:
1. Demand Data: contains customer request for ride hailing services in specific time and location (lat, long)
2. Supply Data: contains iddle active drivers in specific time and location (lat, long)

Since we don't have direct access to such a kind of data, we will create a dummy data processed from Kaggle (https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i).

The data provides a dataset describing the trajectories for all the 442 taxis running in the city of Porto, in Portugal (around 75km2 area). We treat the taxi's starting point as demand data (customer requests), and taxi end point (drivers availability) as supply data.
