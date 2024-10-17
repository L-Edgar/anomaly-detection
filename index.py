import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from collections import deque

class AnomalyDetector:
    def __init__(self, window_size=50, threshold=3):
        self.window_size = window_size
        self.threshold = threshold
        self.data_window = deque(maxlen=window_size)
        self.mean = 0
        self.std_dev = 0
        self.ema = 0
        self.alpha = 0.1  # EMA weighting factor

    def update_stats(self, new_value):
        if len(self.data_window) == self.window_size:
            old_value = self.data_window.popleft()
            self.mean -= old_value / self.window_size

        self.data_window.append(new_value)
        self.mean += new_value / self.window_size
        self.ema = (self.alpha * new_value) + (1 - self.alpha) * self.ema

        # Calculate standard deviation
        if len(self.data_window) > 1:
            self.std_dev = np.std(list(self.data_window))

    def is_anomaly(self, value):
        if self.std_dev == 0:
            return False
        z_score = abs((value - self.mean) / self.std_dev)
        return z_score > self.threshold

def generate_data():
    t = 0
    while True:
        base_value = 10 * np.sin(0.1 * t) + 20
        noise = random.normalvariate(0, 1)
        value = base_value + noise
        # Introduce occasional anomaly
        if random.random() < 0.05:
            value += random.choice([15, -15])
        yield value
        t += 1

def visualize(detector):
    fig, ax = plt.subplots()
    x_data, y_data = [], []
    anomalies_x, anomalies_y = [], []
    stream = generate_data()

    def animate(i):
        value = next(stream)
        x_data.append(i)
        y_data.append(value)
        detector.update_stats(value)

        if detector.is_anomaly(value):
            anomalies_x.append(i)
            anomalies_y.append(value)

        ax.clear()
        ax.plot(x_data, y_data, label="Data Stream")
        ax.scatter(anomalies_x, anomalies_y, color="red", label="Anomalies", zorder=5)
        ax.set_title("Real-time Data Stream with Anomaly Detection")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()

    ani = animation.FuncAnimation(fig, animate, interval=100)
    plt.show()

if __name__ == "__main__":
    detector = AnomalyDetector(window_size=50, threshold=3)
    visualize(detector)