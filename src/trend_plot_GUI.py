import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from datetime import datetime

import numpy as np
import trend_identification as trend
import data_sourcer as ds


class PlottingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Trend identification visualizer")

        # Create input fields
        self.label_code = ttk.Label(root, text="Enter price code:")
        self.entry_code = ttk.Entry(root)

        self.label_start_date = ttk.Label(root, text="Enter start date (yyyy-mm-dd):")
        self.entry_start_date = ttk.Entry(root)

        self.label_window = ttk.Label(root, text="Enter window in days:")
        self.entry_window = ttk.Entry(root)

        # Create plot button
        self.plot_button_curve = ttk.Button(root, text="Plot the entire curve with identified trend", command=self.plot_trend_curve)
        self.plot_button_scatter = ttk.Button(root, text="Plot the scatter plots of trend P&L and duration", command=self.plot_trend_scatter)
        self.plot_button_clear = ttk.Button(root, text="Clear figure", command=self.clear_canvas)

        # Create plot canvas
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()

        # Arrange the elements mentioned above
        self.label_code.grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        self.entry_code.grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)
        self.label_start_date.grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
        self.entry_start_date.grid(row=1, column=1, padx=10, pady=5, sticky=tk.W)
        self.label_window.grid(row=0, column=2, padx=10, pady=5, sticky=tk.W)
        self.entry_window.grid(row=0, column=3, padx=10, pady=5, sticky=tk.W)
        self.plot_button_curve.grid(row=3, column=0, padx=10, pady=10, sticky=tk.W)
        self.plot_button_scatter.grid(row=3, column=1, padx=10, pady=10, sticky=tk.W)
        self.plot_button_clear.grid(row=3, column=2, padx=10, pady=10, sticky=tk.W)
        self.canvas_widget.grid(row=4, column=0, rowspan=12, columnspan=4, padx=10, pady=5, sticky=tk.W+tk.E)

        self.output_state_wrt_1st_last_RP_return = ttk.Label(root, text="Return w.r.t. the last RP: ")
        self.output_state_wrt_1st_last_RP_duration = ttk.Label(root, text="Duration w.r.t. the last RP: ")
        self.output_state_wrt_1st_last_RP_return_pct = ttk.Label(root, text="Return percentile w.r.t. the last RP: ")
        self.output_state_wrt_1st_last_RP_duration_pct = ttk.Label(root, text="Duration percentile w.r.t. the last RP: ")

        self.output_state_wrt_1st_last_RP_return.grid(row=4, column=4, padx=10, pady=5, sticky=tk.W)
        self.output_state_wrt_1st_last_RP_duration.grid(row=5, column=4, padx=10, pady=5, sticky=tk.W)
        self.output_state_wrt_1st_last_RP_return_pct.grid(row=6, column=4, padx=10, pady=5, sticky=tk.W)
        self.output_state_wrt_1st_last_RP_duration_pct.grid(row=7, column=4, padx=10, pady=5, sticky=tk.W)

        self.output_state_wrt_1st_last_RP_return_value = ttk.Label(root, text=" ")
        self.output_state_wrt_1st_last_RP_duration_value = ttk.Label(root, text=" ")
        self.output_state_wrt_1st_last_RP_return_pct_value = ttk.Label(root, text=" ")
        self.output_state_wrt_1st_last_RP_duration_pct_value = ttk.Label(root, text=" ")

        self.output_state_wrt_1st_last_RP_return_value.grid(row=4, column=5, padx=10, pady=5, sticky=tk.W)
        self.output_state_wrt_1st_last_RP_duration_value.grid(row=5, column=5, padx=10, pady=5, sticky=tk.W)
        self.output_state_wrt_1st_last_RP_return_pct_value.grid(row=6, column=5, padx=10, pady=5, sticky=tk.W)
        self.output_state_wrt_1st_last_RP_duration_pct_value.grid(row=7, column=5, padx=10, pady=5, sticky=tk.W)

        self.output_state_wrt_2nd_last_RP_return = ttk.Label(root, text="Return w.r.t. the 2nd last RP: ")
        self.output_state_wrt_2nd_last_RP_duration = ttk.Label(root, text="Duration w.r.t. the 2nd last RP: ")
        self.output_state_wrt_2nd_last_RP_return_pct = ttk.Label(root, text="Return percentile w.r.t. the 2nd last RP: ")
        self.output_state_wrt_2nd_last_RP_duration_pct = ttk.Label(root, text="Duration percentile w.r.t. the 2nd last RP: ")

        self.output_state_wrt_2nd_last_RP_return.grid(row=8, column=4, padx=10, pady=5, sticky=tk.W)
        self.output_state_wrt_2nd_last_RP_duration.grid(row=9, column=4, padx=10, pady=5, sticky=tk.W)
        self.output_state_wrt_2nd_last_RP_return_pct.grid(row=10, column=4, padx=10, pady=5, sticky=tk.W)
        self.output_state_wrt_2nd_last_RP_duration_pct.grid(row=11, column=4, padx=10, pady=5, sticky=tk.W)

        self.output_state_wrt_2nd_last_RP_return_value = ttk.Label(root, text=" ")
        self.output_state_wrt_2nd_last_RP_duration_value = ttk.Label(root, text=" ")
        self.output_state_wrt_2nd_last_RP_return_pct_value = ttk.Label(root, text=" ")
        self.output_state_wrt_2nd_last_RP_duration_pct_value = ttk.Label(root, text=" ")

        self.output_state_wrt_2nd_last_RP_return_value.grid(row=8, column=5, padx=10, pady=5, sticky=tk.W)
        self.output_state_wrt_2nd_last_RP_duration_value.grid(row=9, column=5, padx=10, pady=5, sticky=tk.W)
        self.output_state_wrt_2nd_last_RP_return_pct_value.grid(row=10, column=5, padx=10, pady=5, sticky=tk.W)
        self.output_state_wrt_2nd_last_RP_duration_pct_value.grid(row=11, column=5, padx=10, pady=5, sticky=tk.W)

        self.output_state_wrt_3rd_last_RP_return = ttk.Label(root, text="Return w.r.t. the 3rd last RP: ")
        self.output_state_wrt_3rd_last_RP_duration = ttk.Label(root, text="Duration w.r.t. the 3rd last RP: ")
        self.output_state_wrt_3rd_last_RP_return_pct = ttk.Label(root, text="Return percentile w.r.t. the 3rd last RP: ")
        self.output_state_wrt_3rd_last_RP_duration_pct = ttk.Label(root, text="Duration percentile w.r.t. the 3rd last RP: ")

        self.output_state_wrt_3rd_last_RP_return.grid(row=12, column=4, padx=10, pady=5, sticky=tk.W)
        self.output_state_wrt_3rd_last_RP_duration.grid(row=13, column=4, padx=10, pady=5, sticky=tk.W)
        self.output_state_wrt_3rd_last_RP_return_pct.grid(row=14, column=4, padx=10, pady=5, sticky=tk.W)
        self.output_state_wrt_3rd_last_RP_duration_pct.grid(row=15, column=4, padx=10, pady=5, sticky=tk.W)

        self.output_state_wrt_3rd_last_RP_return_value = ttk.Label(root, text=" ")
        self.output_state_wrt_3rd_last_RP_duration_value = ttk.Label(root, text=" ")
        self.output_state_wrt_3rd_last_RP_return_pct_value = ttk.Label(root, text=" ")
        self.output_state_wrt_3rd_last_RP_duration_pct_value = ttk.Label(root, text=" ")

        self.output_state_wrt_3rd_last_RP_return_value.grid(row=12, column=5, padx=10, pady=5, sticky=tk.W)
        self.output_state_wrt_3rd_last_RP_duration_value.grid(row=13, column=5, padx=10, pady=5, sticky=tk.W)
        self.output_state_wrt_3rd_last_RP_return_pct_value.grid(row=14, column=5, padx=10, pady=5, sticky=tk.W)
        self.output_state_wrt_3rd_last_RP_duration_pct_value.grid(row=15, column=5, padx=10, pady=5, sticky=tk.W)

    def plot_trend_curve(self):

        window_in_days = int(self.entry_window.get())
        price_raw = ds.download_stock_price_daily_close(code=self.entry_code.get(), start_date=self.entry_start_date.get(), end_date=datetime.now().strftime("%Y-%m-%d"))
        RP_vector, RP_summary = trend.trend_identification_main(price_raw, False, window_in_days)

        # Here the canvas needs to be cleared
        self.canvas.get_tk_widget().delete("all")
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=4, column=0, rowspan=12, columnspan=4, padx=10, pady=5, sticky=tk.W+tk.E)

        self.update_current_state(RP_vector, RP_summary)

        matplotlib.use('Agg')  # Set the non-interactive backend, in order to prevent the extra popup window when plotting
        ax = self.figure.add_subplot(111)
        trend.trend_plot_curve(RP_vector, RP_summary, window_in_days)
        self.canvas.figure = plt.gcf()
        self.canvas.draw()
        self.canvas_widget.grid(row=4, column=0, rowspan=12, columnspan=4, padx=10, pady=5, sticky=tk.W+tk.E)

        matplotlib.use('TkAgg')  # Return to the interactive backend

    def plot_trend_scatter(self):

        window_in_days = int(self.entry_window.get())
        price_raw = ds.download_stock_price_daily_close(code=self.entry_code.get(), start_date=self.entry_start_date.get(), end_date=datetime.now().strftime("%Y-%m-%d"))
        RP_vector, RP_summary = trend.trend_identification_main(price_raw, False, window_in_days)

        # Here the canvas needs to be cleared
        self.canvas.get_tk_widget().delete("all")
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=4, column=0, rowspan=12, columnspan=4, padx=10, pady=5, sticky=tk.W+tk.E)

        self.update_current_state(RP_vector, RP_summary)

        matplotlib.use('Agg')  # Set the non-interactive backend, in order to prevent the extra popup window when plotting
        ax = self.figure.add_subplot(111)
        trend.trend_plot_scatter(RP_summary, window_in_days)
        self.canvas.figure = plt.gcf()
        self.canvas.draw()
        matplotlib.use('TkAgg')  # Return to the interactive backend

    def clear_canvas(self):
        # Clear the canvas
        self.canvas.get_tk_widget().delete("all")
        self.figure.clf()
        self.canvas.draw()

    def update_current_state(self, RP_vector, RP_summary):
        current_state_stat_1st = trend.current_state_in_trend(datetime.now().strftime("%Y-%m-%d"), 1, RP_vector, RP_summary)
        current_state_stat_2nd = trend.current_state_in_trend(datetime.now().strftime("%Y-%m-%d"), 2, RP_vector, RP_summary)
        current_state_stat_3rd = trend.current_state_in_trend(datetime.now().strftime("%Y-%m-%d"), 3, RP_vector, RP_summary)

        # Next, update the label text by the variables.
        # I know it's ugly but this is simpler to code...
        # wrt 1st last
        value_to_update = "{:.2%}".format(current_state_stat_1st[0])
        updated_text = f"{value_to_update}"
        # Update the label with the new text
        self.output_state_wrt_1st_last_RP_return_value.config(text=updated_text)

        value_to_update = current_state_stat_1st[1]
        updated_text = f"{value_to_update}"
        self.output_state_wrt_1st_last_RP_duration_value.config(text=updated_text)

        value_to_update = "{:.2%}".format(current_state_stat_1st[2])
        updated_text = f"{value_to_update}"
        self.output_state_wrt_1st_last_RP_return_pct_value.config(text=updated_text)

        value_to_update = "{:.2%}".format(current_state_stat_1st[3])
        updated_text = f"{value_to_update}"
        self.output_state_wrt_1st_last_RP_duration_pct_value.config(text=updated_text)

        # wrt 2nd last
        value_to_update = "{:.2%}".format(current_state_stat_2nd[0])
        updated_text = f"{value_to_update}"
        # Update the label with the new text
        self.output_state_wrt_2nd_last_RP_return_value.config(text=updated_text)

        value_to_update = current_state_stat_2nd[1]
        updated_text = f"{value_to_update}"
        self.output_state_wrt_2nd_last_RP_duration_value.config(text=updated_text)

        value_to_update = "{:.2%}".format(current_state_stat_2nd[2])
        updated_text = f"{value_to_update}"
        self.output_state_wrt_2nd_last_RP_return_pct_value.config(text=updated_text)

        value_to_update = "{:.2%}".format(current_state_stat_2nd[3])
        updated_text = f"{value_to_update}"
        self.output_state_wrt_2nd_last_RP_duration_pct_value.config(text=updated_text)

        # wrt 3rd last
        value_to_update = "{:.2%}".format(current_state_stat_3rd[0])
        updated_text = f"{value_to_update}"
        # Update the label with the new text
        self.output_state_wrt_3rd_last_RP_return_value.config(text=updated_text)

        value_to_update = current_state_stat_3rd[1]
        updated_text = f"{value_to_update}"
        self.output_state_wrt_3rd_last_RP_duration_value.config(text=updated_text)

        value_to_update = "{:.2%}".format(current_state_stat_3rd[2])
        updated_text = f"{value_to_update}"
        self.output_state_wrt_3rd_last_RP_return_pct_value.config(text=updated_text)

        value_to_update = "{:.2%}".format(current_state_stat_3rd[3])
        updated_text = f"{value_to_update}"
        self.output_state_wrt_3rd_last_RP_duration_pct_value.config(text=updated_text)

    # def show_error(self, message):
    #     error_popup = tk.Toplevel(self.root)
    #     error_popup.title("Error")
    #     error_label = tk.Label(error_popup, text=message)
    #     error_label.pack()
    #     ok_button = tk.Button(error_popup, text="OK", command=error_popup.destroy)
    #     ok_button.pack()


if __name__ == "__main__":
    # root = tk.Tk()
    root = ThemedTk(theme="elegance")  # Specify the theme name here
    app = PlottingApp(root)
    root.mainloop()
