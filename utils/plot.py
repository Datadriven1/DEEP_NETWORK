import matplotlib.pyplot as plt
import time
import os


def get_unique_filename(filename):
    unique_filename = time.strftime(f"%Y%m%d_%H%M%S_{filename}")
    return unique_filename

def save_plot(y_test,pred):
    """
    Save plot to file
    """
    #unique_filename = get_unique_filename(plot)
    plt.figure(figsize=(8,8))
    plt.scatter(y_test, pred)
    plt.scatter(y_test, y_test, c='r')
    y_plus = []
    for i in y_test["pIC50"]:
        a = i+1
        y_plus.append(a)
    plt.scatter(y_test, y_plus, c='r')
    y_minus = []
    for i in y_test["pIC50"]:
        a = i-1
        y_minus.append(a)
    plt.scatter(y_test,y_minus, c='r')

    plt.xlim(2, 11)
    plt.ylim(2, 11)
    plt.xlabel('y_test')
    plt.ylabel('Predicted y')
    plt.title('Test data and predicted data')
    
    plt.savefig("unique_filename.png")
