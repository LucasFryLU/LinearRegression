import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("Starting")

    data = pd.read_csv('salarydata.csv')
    data = data.dropna()

    m = 0
    b = 0
    L = 0.0001
    epochs = 10000

    for i in range(epochs):
        m, b = gradientDST(m, b, data, L)
        print(i)

    print(m, b)

    x_data = data['Years of Experience']
    y_data = data['Salary']

    plt.scatter(x_data, y_data, color='blue', label='Data Points')

    x_line = np.linspace(x_data.min(), x_data.max(), 100)
    y_line = m * x_line + b

    plt.plot(x_line, y_line, color='red', linewidth=2)
    plt.show()


def gradientDST(mcurrent, bcurrent, points, L):
    x = points['Years of Experience'].values  #Takes Years of Exp as a 1D array
    y = points['Salary'].values
    n = len(points)

    #Calculate prediction and error for ALL rows at once
    y_pred = mcurrent * x + bcurrent #Creates a 1D y-predicion array using matrix mult and addition
    error = y - y_pred #1D Prediciton array minus 1D Actual Array 

    #Calculate gradients
    mgrad = (-2/n) * sum(x * error)
    bgrad = (-2/n) * sum(error)

    #Update parameters
    m = mcurrent - mgrad * L #Updating Weights
    b = bcurrent - bgrad * L #Updating Bias (Machine Learning Bias is the Same!)
    
    return m, b


if __name__ == "__main__":
    main()
