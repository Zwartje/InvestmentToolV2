import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

import trend_identification as trend
import data_sourcer as ds

# Function to create a sample figure
def create_figure_1():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plt.figure(figsize=(5, 3))
    plt.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Sample Figure 1')
    plt.grid(True)

    # Update the figure in the embedded canvas
    canvas.figure = plt.gcf()
    canvas.draw()

def create_figure_2():
    x = np.linspace(0, 10, 100)
    y = np.cos(x)
    plt.figure(figsize=(5, 3))
    plt.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Sample Figure 2')
    plt.grid(True)

    # Update the figure in the embedded canvas
    canvas.figure = plt.gcf()
    canvas.draw()

# Function to handle button click event
def button_click():
    input_text = entry.get()
    if input_text:
        try:
            value = int(input_text)
            messagebox.showinfo("Success", f"Input value: {value}")
        except ValueError:
            messagebox.showerror("Error", "Invalid input! Please enter an integer.")
    else:
        messagebox.showerror("Error", "Input cannot be empty!")

# Create the main window
root = tk.Tk()
root.title("GUI with Figures")

# Create an input field
entry = tk.Entry(root)
entry.pack(pady=10)

# Create a button to trigger the function
button = tk.Button(root, text="Submit", command=button_click)
button.pack()

# Create buttons to display figures
figure_button_1 = tk.Button(root, text="Show Figure 1", command=create_figure_1)
figure_button_1.pack(pady=5)

figure_button_2 = tk.Button(root, text="Show Figure 2", command=create_figure_2)
figure_button_2.pack(pady=5)

# Create a FigureCanvasTkAgg widget to embed the figure in the main window
fig, ax = plt.subplots(figsize=(5, 3))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# Run the GUI main loop
root.mainloop()
