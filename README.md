# Prediction-with-Neural-Network

Using publicly available insurance data [data](https://www.kaggle.com/datasets/easonlai/sample-insurance-claim-prediction-dataset) ([data](https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv)), I wanted to build a neural network model which is able to predict price/charges based on the age, sex, bmi, childre, smoker, region using [TensorFlow](https://www.tensorflow.org/)

Here I compare two differnt models; one with, and one without activation [ReLu](https://www.tensorflow.org/api_docs/python/tf/keras/activations/relu) after transforming dataset by [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) and [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html), splitting data to test and train sets.
Same dataset was used on both models to compare them. Model_1 has no activation meanwhile model_2 consists of 'relu' in two layers as well. Beside of that different, both contains 3 layers with 100, 10 then 1 neuron. As optimizer, [Adam](https://keras.io/api/optimizers/adam/) has been introduced with learning_rate=0.1. Furthermore [mae](https://en.wikipedia.org/wiki/Mean_absolute_error) is the metrics.



