# Libraries
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor


# Import dataset
instagram_data = pd.read_csv("datasets/instagram_data.csv", encoding = 'latin1')

def predict_instagram_model(data):
    x = np.array(data[['Likes', 'Saves', 'Comments', 'Shares', 
                   'Profile Visits', 'Follows']])
    y = np.array(data["Impressions"])

    # Split dataset (80% train + 20% test)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

    model = PassiveAggressiveRegressor()
    model.fit(xtrain, ytrain)
    model.score(xtest, ytest)

    return model


if __name__ == "__main__":
    instagram_model = predict_instagram_model(instagram_data)

    # Features = [['Impressions','Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']]
    test_features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
    prediction = instagram_model.predict(test_features)
    print("Predicted Impressions:", round(prediction[0]))
