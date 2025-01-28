import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Define constants
COLOR_PALETTE = [
    "#88CCEE", "#CC6677", "#DDCC77", "#117733",
    "#332288", "#AA4499", "#44AA99", "#999933",
    "#882255", "#661100", "#6699CC", "#888888"
]
MARKER_STYLES = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']

# Function to load and prepare data
def load_and_prepare_data(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Ensure columns have the expected names
    if not {'accuracy', 'reaction_time', 'n'}.issubset(data.columns):
        raise ValueError("CSV must contain 'accuracy', 'reaction_time', and 'n' columns.")

    # Filter out rows where data points are missing
    data = data.dropna(subset=['accuracy', 'reaction_time', 'n'])

    # Increase reaction time values by 0.25 ms
    data['reaction_time'] = data['reaction_time'] + 0.25

    # Create a unique identifier for consecutive n values
    data['n_group'] = (data['n'] != data['n'].shift()).cumsum()

    # Group data by the unique n_group
    groups = data.groupby('n_group')

    return data, groups

# Function to plot the data on a Tkinter canvas
def plot_in_canvas(parent_frame, data, groups):
    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = COLOR_PALETTE * (len(groups) // len(COLOR_PALETTE) + 1)
    markers = MARKER_STYLES * (len(groups) // len(MARKER_STYLES) + 1)

    x_min, x_max = data['reaction_time'].min() - 0.1, data['reaction_time'].max() + 0.1
    y_min, y_max = data['accuracy'].min() - 5, data['accuracy'].max() + 5

    marker_size = 80

    for (n_group, group), color, marker in zip(groups, colors, markers):
        label = f'n = {group["n"].iloc[0]} (group {n_group})'
        ax.scatter(
            group['reaction_time'],
            group['accuracy'],
            label=label,
            color=color,
            alpha=0.8,
            s=marker_size,
            edgecolor='k',
            marker=marker
        )

    ax.set_title("Relationship Between Accuracy and Reaction Time", fontsize=16, fontweight='bold')
    ax.set_xlabel("Reaction Time (ms)", fontsize=14)
    ax.set_ylabel("Accuracy (%)", fontsize=14)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(
        title="n Value Groups",
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        fontsize=10,
        title_fontsize=12
    )

    # Add the plot to the Tkinter canvas
    canvas = FigureCanvasTkAgg(fig, master=parent_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

# Main application
def main():
    # Load the data
    file_path = "graph.csv"  # Replace with the path to your CSV file
    try:
        data, groups = load_and_prepare_data(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Create the Tkinter app
    root = tk.Tk()
    root.title("Scatter Plot Viewer")
    root.geometry("900x700")

    # Create a frame for the plot
    frame = ttk.Frame(root)
    frame.pack(fill="both", expand=True, padx=10, pady=10)

    # Plot the data
    plot_in_canvas(frame, data, groups)

    # Run the application
    root.mainloop()

if __name__ == "__main__":
    main()
