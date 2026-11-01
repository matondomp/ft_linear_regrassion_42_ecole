import csv
import math
import json
import random

def evaluate_model(x_test, y_test, theta0, theta1):
    errors = []

    for i in range(len(x_test)):
        prediction = theta0 + theta1 * x_test[i]
        error = prediction - y_test[i]
        errors.append(error ** 2)

    mse = sum(errors) / len(errors)
    rmse = math.sqrt(mse)

    return mse, rmse

def convert_parameters(theta_0, theta_1, mean_x, std):
    theta_1 = theta_1 / std
    theta_0 = theta_0 - (theta_1 * mean_x)

    return theta_0, theta_1 

def gradient_descent(x, y, learning_rate=0.1, iterations= 500):
    m = len(x)
    theta_0 = 0
    theta_1 = 0

    for epoch in range(iterations):
        aux_theta_0 = 0
        aux_theta_1 = 0

        for i in range(m):
            
            y_pred = theta_0 + theta_1 * x[i]
            error = y_pred - y[i]

            aux_theta_0 += error
            aux_theta_1 += error * x[i]
        if epoch % 100 == 0:
            print('epoch =>',epoch, theta_0, theta_1)  

        aux_theta_0 /= m
        aux_theta_1 /= m
        
        theta_0 -= learning_rate * aux_theta_0
        theta_1 -= learning_rate * aux_theta_1

    return theta_0, theta_1

def standardize_x(x_train, x_test):
    mean_x = sum(x_train) / len(x_train)
    variance_x = sum((km - mean_x) ** 2 for km in x_train) / len(x_train)
    std = math.sqrt(variance_x)

    train_scaled = [(km - mean_x) / std for km in x_train]
    test_scaled = [(km -mean_x) / std for km in x_test]
    return train_scaled, test_scaled, mean_x, std

def extract_features(data):
    features = []
    labels = []

    for km, price in data:
        features.append(km)
        labels.append(price)

    return features, labels

def split_data(data):
    random.seed(42)
    random.shuffle(data)
    index = int(len(data) * 0.8)

    return data[:index], data[index:]

def load_data(file_name):
    data = []

    with open(file_name, 'r') as file:
        reader = csv.DictReader(file)

        for row in reader:
            km = float(row['km'])
            price = float(row['price'])

            data.append((km, price))
    
    return data

data = load_data('./data.csv')
train_data, test_data = split_data(data)
x_train, y_train = extract_features(train_data)
x_test, y_test = extract_features(test_data)

standardized_train, standardized_test, mean_x, std = standardize_x(
    x_train, x_test
)

b_0, b_1 = gradient_descent(standardized_train, y_train)

theta_0, theta_1 = convert_parameters(b_0, b_1, mean_x, std)

mse, rmse = evaluate_model(
    x_test, y_test, theta_0, theta_1
)

print("Test MSE:", mse)
print("Test RMSE:", rmse)

parameters = {
    "theta_0": theta_0,
    "theta_1": theta_1
}

with open('theta.json', 'w') as file:
    json.dump(parameters, file, indent=4)

