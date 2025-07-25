# DSI - Project 

# Car sales forecasting - ML Team 3 

Team members: 

- Artem Kostiuk
- Julio Socher
- Palash Bose
- Sakibal Sadat 
- Victor Lopez
- Zoe Ackah 

# Business Problem and Objective 
#MLTeam3 is a market leading reserach group focused on automative supply chain with data backed reporting and forecasting services for auto manufacturers, parts suppliers and dealership groups across the globe. #MLTeam3 provides AI backed market predictions and forecasts to optimize demand based decision making for all verticles in the industry. 

Business Problem:
A large dealership network is seeking to optimize their sales forecast and engaged #MLTeam3 to boost their inventory management by identifying best selling vehicles within their competitve markets. 

Objective:
#MLTeam3 will develop a Machine Learning model to predict the top 3 best sellers and aid the dealer's inventory decsions to ensure model avilablity for customers to purchase with the aim of increasing sales volume and driving profitability. 

# Note about the dataset
#MLTeam3 is leveraging the Car Sales Report data from Kaggle for training and testing the ML model. The base dataset contains sales data with over 20000 samples within 16 features. The dataset contains historical sales data for peroid of 24 months starting from Jan 1, 2022 and concluding by Dec 2023. The key features in the data are:

- Car_ID - A unique identifer for each car 
- Date - Date of the sale transaction 
- Customer Name - Name of the purchaser
- Gender - Gender of the purchaser
- Annual Income - Declared income of the purchaser
- Dealer Name - Name of the sales dealer
- Dealer No - Dealer id  
- Company - Brand/make of the car
- Model - Vehicle model 
- Engine - Engine specification 
- Transmission - Transmission of the vehicle 
- Color - Paint characteristics
- Body Style - Type of vehicle 
- Phone - Phone number of the purchaser
- Dealer Region - Geographic region of the dealer 

## Syntetic Additions
To enhance the predictive power and real-world relevance of our dataset, we have introduced synthetic features that simulate critical variables often considered by consumers in the vehicle purchasing process. These additions aim to supplement the original data with practical and human-centered context.

- Family Size
Reflects the number of people in the purchaser’s household. This is a key factor influencing the type of vehicle chosen (e.g., sedan vs. SUV). Estimated based on vehicle category and type. For example, larger vehicles like SUVs and minivans are assigned larger family sizes, while compact cars typically serve smaller households or individuals. This variable simulates one of the key factors in buyer decision-making.

- Gas Mileage
Represents the distance the vehicle has been driven so far, one of the most important criteria for buyers when selecting a car. Assigned in relation to the vehicle’s brand, type, and price, we have determined a arbitrary start point and generated a random number based on that start point using a random function.

- Crash Test Score
Indicates the vehicle’s safety rating, typically based on standardized crash testing. Safety is also a major component in purchase decisions. Generated within a realistic range based on assumed safety performance, based on the type of brand - which was split between luxury and safe brands and also using the price in order to randomly generate a classification. Note: This does not reflect the real crash test score of those vehicles and it can be improved by extracting real data from the government departments in charge of this measure.

These synthetic enhancements were designed to align the dataset more closely with real-world buyer behavior, ultimately improving the performance and explainability of downstream forecasting models but are not trying to get a 100% accuracy to the data as all those synthetic data is randomly created based on values assigned by the group.

# Methodology
- (TBD) How are we going to perfrom each of the following

# Data Preperation 
- (TBD) Along with data preperation techniques we also need to highlight how we plan to populate data for the three new features 

# Model selection 
- (TBD) 

# Model Training and Evaluation 
- (TBD)

# Key Findings 
- (TBD)

# Visualization and Observation 
- (TBD)

# Conclusion 
- (TBD)

