import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_gesture_data(filename, gesture_name):
    """Plot acceleration and gyroscope data for a gesture."""
    # Read the data
    df = pd.read_csv("./data/" + filename)
    index = range(1, len(df['aX']) + 1)
    
    # Set figure size
    plt.rcParams["figure.figsize"] = (20, 10)
    
    # Plot acceleration data
    plt.figure()
    for axis, color in [('aX', 'g'), ('aY', 'b'), ('aZ', 'r')]:
        plt.plot(index, df[axis], f'{color}.', label=axis[1].lower(), 
                linestyle='solid', marker=',')
    plt.title(f"{gesture_name} Acceleration")
    plt.xlabel(f"{gesture_name} Sample #")
    plt.ylabel(f"{gesture_name} Acceleration (G)")
    plt.legend()
    plt.show()
    
    # Plot gyroscope data
    plt.figure()
    for axis, color in [('gX', 'g'), ('gY', 'b'), ('gZ', 'r')]:
        plt.plot(index, df[axis], f'{color}.', label=axis[1].lower(), 
                linestyle='solid', marker=',')
    plt.title(f"{gesture_name} Gyroscope")
    plt.xlabel(f"{gesture_name} Sample #")
    plt.ylabel(f"{gesture_name} Gyroscope (deg/sec)")
    plt.legend()
    plt.show()

# List of gestures to plot
gestures = [
    ("sidelift.csv", "Sidelift Curl"),
    ("rotcurl.csv", "Rotating Curl"),
    ("curl.csv", "Curl"),
    ("punch.csv", "Punch"),
    ("flex.csv", "Flex")
]

# Plot data for each gesture
for filename, gesture_name in gestures:
    plot_gesture_data(filename, gesture_name)

