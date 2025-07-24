# DSI - Project 

# Car sales forecasting - ML Team 3 

![Dataset Cover](./images/dataset-cover.png)

## Table of Contents

1. [Team Members](#team-members)
2. [Business Problem and Objective](#business-problem-and-objective)
3. [Note About the Dataset](#note-about-the-dataset)
4. [Synthetic Additions](#synthetic-additions)
5. [Methodology](#methodology)
    - [Data Preparation](#data-preparation)
    - [Model Selection](#model-selection)
    - [Model Training and Evaluation](#model-training-and-evaluation)
6. [Key Findings](#key-findings)
7. [Visualization and Observation](#visualization-and-observation)
8. [Conclusion](#conclusion)

## Team members: 

- [Artem Kostiuk](https://github.com/postatum)
- [Julio Socher](https://github.com/juliosocher)
- [Palash Bose](https://github.com/pala2003)
- [Sakibal Sadat](https://github.com/ssadat92)
- [Victor Lopez](https://github.com/vhlopezch)
- [Zoe Ackah](https://github.com/zoeackah)

## Business Problem and Objective 
#MLTeam3 is a market leading research group focused on automotive supply chain with data backed reporting and forecasting services for auto manufacturers, parts suppliers and dealership groups across the globe. #MLTeam3 provides AI backed market predictions and forecasts to optimize demand based decision making for all vehicles in the industry. 

### Business Problem
A large dealership network is seeking to optimize their sales forecast and engaged #MLTeam3 to boost their inventory management by identifying best selling vehicles within their competitve markets. 

### Objective
#MLTeam3 will develop a Machine Learning model to predict the top 3 best sellers and aid the dealer's inventory decisions to ensure model availability for customers to purchase with the aim of increasing sales volume and driving profitability. Target variable for the machine learning model is binary classification of "best seller" - whether a car will be a best seller in a particular month based on the car's features and historical sales data. Car dealers will be able to use the model to make predictions about best sellers to adjust inventory.

### Risks and unknowns
Some of the risks we have identified are:
* We have a limited amount of data - only ~28 thousand rows and 2 years of data - which limits our ability to well-tune a model. This will result in a sub-optimal model which will be prone to making mistakes;
* Technological limitations (e.g. slow machines) may prevent us from choosing the best classification model, because training such a model would either be impossible or take hours. This will results is us choosing a simpler model that is realistic to train using our technological resources.
 
## Note about the dataset
#MLTeam3 is leveraging the Car Sales Report data from Kaggle for training and testing the ML model. The base dataset contains sales data with over 20000 samples within 16 features. The dataset contains historical sales data for period of 24 months starting from Jan 1, 2022 and concluding by Dec 2023. The key features in the data are:

- Car_ID - A unique identifier for each car 
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

You can find the original dataset on Kaggle here: [Car Sales Report Dataset](https://www.kaggle.com/datasets/missionjee/car-sales-report/data)

## Synthetic Additions
Along with the base dataset, we have chosen to include additional features to enhance the dataset focused on real-world metrics. The features we added are completely synthetic, they do not represent the actual data and are not related to it. The only purpose they serve is making data richer.

After much consideration, we have chosen to add the following features:
- Family Size - size of the purchasers family as that is a key buying consideration 
- Gas Milage - Fuel efficiency is one of the top 5 decision factors when buying a vehicle 
- Crash Test Score - Along with fuel efficiency, safety rating of a vehicle is a strong driver of purchasing decision

## Methodology

### Data Preparation 
- (TBD) Along with data preparation techniques we also need to highlight how we plan to populate data for the three new features 

### Model selection 
Considering we are trying to solve a classification problem, we considered the following classification models:
* LogisticRegression;
* RandomForestClassifier;
* XGBClassifier;
* LGBMClassifier;

The reason we chose these particular models is - they were easy to understand, had a variety of hyperparameters to tune and our compute resources were enough to train them in a reasonable time.

### Model Training and Evaluation 

All the models we were working with were trained and evaluated in a similar manner.

First we split all the features we had into numeric and categorical to apply distinct processing to each category down the line. In particular we had:
* numeric features, e.g. year, month, number of sales in a current month, number of sales in previous 1-12 months;
* categorical features, e.g. engine type, transmission type, color, body style.

During data exploration and augmentation we noticed a imbalance in the data - non-best-sellers class was present in about 93% of the data, while best-seller only in around 7%. To address the issue and make sure models consider the imbalance, we calculated weights of the classes using sklearn tools and used them when training all the models.

To train the models we split available data into training and test sets by choosing the last 3 month of available data to be test data. This made sure model evaluation represented a real world scenario - model predicting future events based on past data.

Then we developed reusable functions to perform grid search of a model and evaluate a model. These functions accept parameters so they can be used to tune and evaluate every model in a similar manner.

The function that performs grid search uses the same one-hot encoding hyperparameters for all the models, allows to specify more hyperparameters to tune, prints some metrics and returns a best-performing model. It also aims to maximize F1 score, because this metric represents a balance of precision and recall.

The function that evaluates a model performance, predicts classes using test dataset, compares them to actual classes and outputs a few metrics, such as: accuracy, precision, recall, F1 score, AUC and a confusion matrix. The matrix played a crucial role in our model selection process, because it allowed us to choose a model that satisfied business requirements the best. Here's an example of what it looks like:

![Confusion matrix example](./confusion-matrix.png)

Next we configured a preprocessor pipeline, which is used by all the model. The pipeline does two things:
1. Performs standard scaling of numeric features to make sure they all contribute equally to the model performance. We had to do this, because some of our features have very different scale;
2. Performs One-Hot encoding of categorical features.

Then we created pipelines using each type of classifier we previously selected and the preprocessor we developed previously. Next, for each model we ran grid search using a variety of hyperparameters and evaluated the results. Again, we were aiming for maximizing F1 score and a confusion matrix with the least confusion.

Based on the evaluation we chose model using LGBMClassifier and exported it into a pickle file so it can be loaded and reused down the line. The output model had the following performance metrics:

    * Accuracy: 0.95132186319765
    * Precision: 0.5534441805225653
    * Recall: 0.8411552346570397
    * F1 Score: 0.667621776504298
    * AUC: 0.9706575158047791


## Key Findings 
- (TBD)

## Visualization and Observation 
- (TBD)

## Conclusion 
- (TBD)

