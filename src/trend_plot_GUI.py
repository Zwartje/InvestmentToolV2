import tkinter as tk
from tkinter import ttk
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import numpy as np
import trend_identification as trend
import data_sourcer as ds

class PlottingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Trend identification visualizer")

        # Create input fields
        self.label_code = ttk.Label(root, text="Enter price code:")
        self.label_code.pack()
        self.entry_code = ttk.Entry(root)
        self.entry_code.pack()

        self.label_start_date = ttk.Label(root, text="Enter start date (yyyy-mm-dd):")
        self.label_start_date.pack()
        self.entry_start_date = ttk.Entry(root)
        self.entry_start_date.pack()

        self.label_window = ttk.Label(root, text="Enter window in days:")
        self.label_window.pack()
        self.entry_window = ttk.Entry(root)
        self.entry_window.pack()

        # Create plot button
        self.plot_button_curve = ttk.Button(root, text="Plot the entire curve with identified trend", command=self.plot_trend_curve)
        self.plot_button_curve.pack()

        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack()

    def plot_trend_curve(self):

        window_in_days = int(self.entry_window.get())
        price_raw = ds.download_stock_price_daily_close(code=self.entry_code.get(), start_date=self.entry_start_date.get(), end_date='2023-08-01')
        RP_vector, RP_summary = trend.trend_identification_main(price_raw, False, window_in_days)

        matplotlib.use('Agg')  # Set the non-interactive backend, in order to prevent the extra popup window when plotting
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        trend.trend_plot_curve(RP_vector, RP_summary, window_in_days)
        self.canvas.figure = plt.gcf()
        self.canvas.draw()
        matplotlib.use('TkAgg')  # Return to the interactive backend

    def show_error(self, message):
        error_popup = tk.Toplevel(self.root)
        error_popup.title("Error")
        error_label = tk.Label(error_popup, text=message)
        error_label.pack()
        ok_button = tk.Button(error_popup, text="OK", command=error_popup.destroy)
        ok_button.pack()


if __name__ == "__main__":
    root = tk.Tk()
    app = PlottingApp(root)
    root.mainloop()
