import json

def get_mileage_input():

    while True:
        try:
            mileage = float(input("Enter the mileage (in km): "))
            if mileage < 0:
                print("Mileage cannot be negative. Please enter a valid value.")
                continue
            return mileage
        except ValueError:
            print("Invalid input. Please enter a numeric value for mileage.")
        

def predict(mileage, theta_0, theta_1):
    y_pred = theta_0 + theta_1 * mileage
    
    return y_pred

def load_parameters():

    with open('theta.json', 'r') as file:
        parameters = json.load(file)
        
    return parameters["theta_0"], parameters["theta_1"]

theta_0, theta_1 = load_parameters()
mileage = get_mileage_input()

if mileage < 22899 or mileage > 176000:
    print("Warning: Mileage is outside the training data range.")
    
predicted_price = predict(mileage, theta_0, theta_1)

print("Predicted price:", "{:.2f} kz".format(predicted_price))