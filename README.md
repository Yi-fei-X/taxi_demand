# Taxi Demand

In this assignment, I implemented several time series models to predict taxi demand (i.e., #pick-ups)  in New York City.

#### Dataset description:

The dataset consists of taxi trips spanning a period of days in New York City. The training data records normalized taxi pick up counts for 100 regions at each hour of the day. We provide a sequence of 8 hours for each sample.

The feature matrix has dimensions (72000 x 8 x 49), where 72,000 = 30 days * 24 hours * 100 regions, 8 is the sequence size, and 49 features extracted from neighbors’ information. The dimension of the label is (72000, 1). In addition, we provide the region location as (72000, 2), where each row represents the location of each grid (e.g., (7,7), (8,7), (6,7) are nearby grids). The time matrix is also provided as a (72000, 1) matrix, where each row represents the hour index of the time.

To load the training data, use the numpy.load function:  
data = np.load('train.npz')  
x = data['x'] #feature matrix  
y = data['y'] #label matrix  
location = data['locations'] #location matrix  
times = data['times'] #time matrix

#### Evaluation metrics:

For all experiments, use the following metrics to compare your models:

Root Mean Squared Error (RMSE):  
https://en.wikipedia.org/wiki/Root-mean-square_deviation

#### Instructions:

Use the training data to train your models and validation set to compare them. Create a table detailing all of your results.

Q1) Implement a simple baseline where the historical average for each region is predicted against the validation set. Report the RMSE

Q2) Extract features from the temporal data and use linear regression to predict demand on the validation set. Report the RMSE

Q3) Extract features from the temporal data and use XGBOOST to predict demand on the validation set. Report the RMSE

Q4) Implement a basic Recurrent Neural Network (RNN) to predict demand on the validation set. Report the RMSE.

Q5) Implement a basic Long-short Term Memory (LSTM) network to predict demand on the validation set. Report the RMSE.

Q6) Choose the best performing model and predict taxi volume using the test data.
