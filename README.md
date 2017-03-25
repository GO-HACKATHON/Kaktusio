# Kaktusio
Go-Search - Machine learning powered Heatmap to predict supply-demand gap for ride hailing services across city. 

## Everyone hates waiting... or, even worse, surge pricing.

In ride-hailing service, nobody likes to get a price increase, especially in the time when you need the service the most. The idea behind surge pricing is to serve as incentives to attract more drivers go down on roads when the demand arises, so that the customer would not have to wait a long time to get a service. It is indeed a sensible approach from a basic supply-demand model to keep the market in equilibrium and improve the service rate. But on the other side, it also a turn-off for some customers.

## What if you can pre-position the supply before the demand arises and avoid surge price?

We intent to build Go-Search, a live heatmap which shows live prediction of supply-demand gap across area in the city. By viewing the heatmap, drivers can search which area that will need their service the most. Therefore, they can make decision to reposition themselves before the demands arises and maximize their revenues. In this way, the supply will be ready for the demand in advance so the waiting time can be minimized and the surge pricing never even has to happen.

In order to achieve this, we will build a predictive model powered by machine learning techniques that analyze historical demand pattern and current drivers position to predict the the supply and demand gap, 1 hour in advance. This model will serve as a backend to a city heatmap showing a supply-demand gap prediction for the next hour.

This heatmap is not only to show a prediction, but also can be developed further (as future work) into a prescriptive model which able to recommend drivers to move to specific area in advance, in order to maximize their revenue, minimize customer waiting time, and also optimize the supply-demand market for the entire city.
