import tkinter as tk
import customtkinter as ctk
import random
import math
import time
import csv
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.exceptions import NotFittedError
import re
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter.messagebox as messagebox  
from tkinter import messagebox
from matplotlib.backend_bases import MouseEvent

# Configure CustomTkinter appearance
ctk.set_appearance_mode("dark")  # Modes: "light", "dark", "system"
ctk.set_default_color_theme("dark-blue")
        

class GraphWindow(tk.Toplevel):
    def __init__(self, csv_filepath, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Graph Window")
        self.geometry("1024x768")  # Set the initial window size to a larger size
        self.resizable(True, True)  # Allow the window to be resizable

        self.csv_filepath = csv_filepath

        # Check if the file exists
        if not os.path.exists(self.csv_filepath):
            messagebox.showerror("Error", f"File not found: {self.csv_filepath}")
            self.destroy()
            return

        # Read and prepare data from the CSV file
        data, groups = self.read_and_prepare_csv_data()
        if data is None or groups is None:
            messagebox.showerror("Error", "CSV file is empty or invalid.")
            self.destroy()
            return

        self.data = data

        # Main frame to hold canvas and controls
        self.main_frame = tk.Frame(self)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=10)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Create the matplotlib figure
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)

        # Plot the scatter plot
        self.plot_scatter(data, groups)

        # Embed the matplotlib figure into the Canvas
        self.canvas_widget = FigureCanvasTkAgg(self.figure, self.main_frame)
        self.canvas_widget.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Add interactivity for point click
        self.canvas_widget.mpl_connect('button_press_event', self.on_click)

        # Adjust canvas size dynamically when the window is resized
        self.bind("<Configure>", self.resize_canvas)

        # Close button
        close_button = tk.Button(self.main_frame, text="Close", command=self.destroy_window)
        close_button.grid(row=1, column=0, pady=10)

        self.protocol("WM_DELETE_WINDOW", self.destroy_window)

    def destroy_window(self):
        self.canvas_widget.get_tk_widget().destroy()
        super().destroy()
        self.quit()

    def read_and_prepare_csv_data(self):
        try:
            # Load the CSV file
            data = pd.read_csv(self.csv_filepath)

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
        except Exception as e:
            print(f"Error reading or preparing CSV data: {e}")
            return None, None

    def plot_scatter(self, data, groups):
        color_palette = [
            "#88CCEE", "#CC6677", "#DDCC77", "#117733",
            "#332288", "#AA4499", "#44AA99", "#999933",
            "#882255", "#661100", "#6699CC", "#888888"
        ]
        marker_styles = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']

        colors = color_palette * (len(groups) // len(color_palette) + 1)
        markers = marker_styles * (len(groups) // len(marker_styles) + 1)

        x_min, x_max = data['reaction_time'].min() - 0.1, data['reaction_time'].max() + 0.1
        y_min, y_max = data['accuracy'].min() - 5, data['accuracy'].max() + 5

        marker_size = 100

        for (n_group, group), color, marker in zip(groups, colors, markers):
            label = f'n = {group["n"].iloc[0]} (group {n_group})'
            self.ax.scatter(
                group['reaction_time'],
                group['accuracy'],
                label=label,
                color=color,
                alpha=0.8,
                s=marker_size,
                edgecolor='k',
                marker=marker
            )

        self.ax.set_title("Relationship Between Accuracy and Reaction Time", fontsize=16, fontweight='bold')
        self.ax.set_xlabel("Reaction Time (ms)", fontsize=14)
        self.ax.set_ylabel("Accuracy (%)", fontsize=14)
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.legend(
            title="n Value Groups",
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
            fontsize=10,
            title_fontsize=12
        )

    def on_click(self, event: MouseEvent):
        if event.inaxes == self.ax:
            x, y = event.xdata, event.ydata
            tolerance = 0.1

            # Find the closest data point within tolerance
            for _, row in self.data.iterrows():
                if abs(row['reaction_time'] - x) <= tolerance and abs(row['accuracy'] - y) <= tolerance:
                    messagebox.showinfo("Data Point Info", f"Reaction Time: {row['reaction_time']:.2f}\nAccuracy: {row['accuracy']:.2f}\nn: {row['n']}")
                    break

    def resize_canvas(self, event):
        self.figure.set_size_inches(event.width / 100, event.height / 100)
        self.canvas_widget.draw()

class ModelPredictor:
    def __init__(self, model_path, required_columns=None, features=None):
        self.model_path = model_path
        #csv_file = 'metrics.csv'
        self.required_columns = required_columns or ["accuracy", "reaction_time"]
        self.features = features or ["accuracy", "reaction_time"]
        
        # Load the trained model
        self.model = self.load_model()
        
    def load_model(self):
        """Load the trained model from the specified path."""
        try:
            return joblib.load(self.model_path)
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")
    
    def predict_from_csv(self, csv_file):
        try:
            # Check if the CSV file exists
            if not os.path.isfile(csv_file):
                raise FileNotFoundError(f"CSV file '{csv_file}' does not exist.")

            # Load the CSV data
            data = pd.read_csv(csv_file)

            # Check if the CSV contains data
            if data.empty:
                raise ValueError("The CSV file is empty.")

            # Check if required columns are present
            missing_columns = [col for col in self.required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in CSV: {', '.join(missing_columns)}")

            # Drop rows with missing values in required columns
            data = data.dropna(subset=self.required_columns)

            # Ensure at least one row remains after dropping missing values
            if data.empty:
                raise ValueError("No data left after removing rows with missing values in required columns.")

            # Check if the required features are present
            missing_features = [feature for feature in self.features if feature not in data.columns]
            if missing_features:
                raise ValueError(f"Missing required features for prediction: {', '.join(missing_features)}")

            # Extract the feature data
            feature_data = data[self.features].mean()
            print(f'mean of Feature data: {feature_data}')
            feature_data = pd.DataFrame([feature_data])
            # Ensure the columns are numeric
            feature_data  = feature_data.apply(pd.to_numeric)

            # Predict using the model
            predictions = self.model.predict(feature_data)

            # Determine the mode of the predictions
            #predicted_class = pd.Series(predictions).mode()[0]
            print(f'Predictions: {predictions}')
            return predictions[0]
            #return predicted_class

        except FileNotFoundError as fnf_error:
            return f"Error: {str(fnf_error)}"
        except pd.errors.EmptyDataError:
            return "Error: The CSV file contains no data."
        except pd.errors.ParserError:
            return "Error: The CSV file could not be parsed."
        except ValueError as ve:
            return f"Error: {str(ve)}"
        except NotFittedError:
            return "Error: The model is not properly fitted. Ensure the model is trained before using it for predictions."
        except Exception as e:
            return f"Unexpected error: {str(e)}"

class IntroWindow(ctk.CTk):
    user_id = None
    n_value = None
    def __init__(self, n_value):
        super().__init__()
        self.title("Welcome")
        self.minsize(800, 600)
        #ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 3)
        #hwnd = win32gui.FindWindow(None, "Welcome")
        #win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)

        self.after_id = None
        self.filepath = "users.csv"

        self.label = ctk.CTkLabel(
            self,
            text=(
                "Welcome to the N-back task!\n\n"
                "You will see a sequence of shapes. This will show how exactly this will work.\n\n"
                "We will start with some practice. Press 'Space' to continue."
            ),
            font=("Arial", 16),
            justify="center",
            wraplength=500
        )
        self.label.pack(expand=True, padx=20, pady=20)

        self.bind("<space>", self.on_space_press)
        self.protocol("WM_DELETE_WINDOW", self.close_window)

        self.buttons_frame = None
        self.input_frame = None

        
        # Ensure the CSV file exists with headers
        if not os.path.exists(self.filepath):
            with open(self.filepath, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Name", "Email", "DoB", "User ID", "N Value"])

    def close_window(self):
        self.quit()
        self.destroy()

    def on_space_press(self, event):
         # Create a frame for buttons
        if self.buttons_frame is None:  # Ensure buttons are only created once
            self.buttons_frame = ctk.CTkFrame(self)
            self.buttons_frame.pack(pady=20)

            # Login Button
            self.button_1 = ctk.CTkButton(
                self.buttons_frame,
                text="Login",
                command=self.show_login_fields
            )
            self.button_1.grid(row=0, column=0, padx=10)

            # Sign Up Button
            self.button_2 = ctk.CTkButton(
                self.buttons_frame,
                text="Sign Up",
                command=self.show_register_fields
            )
            self.button_2.grid(row=0, column=1, padx=10)

    def show_login_fields(self):
        # Clear previous input fields if any
        if self.input_frame:
            self.input_frame.destroy()

        # Create a frame for login inputs
        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.pack(pady=20)

        # User ID Field
        user_id_label = ctk.CTkLabel(self.input_frame, text="User ID:")
        user_id_label.grid(row=0, column=0, padx=5, pady=5)

        user_id_entry = ctk.CTkEntry(self.input_frame)
        user_id_entry.grid(row=0, column=1, padx=5, pady=5)

        # Submit Button
        submit_button = ctk.CTkButton(
            self.input_frame,
            text="Enter",
            command=lambda: self.submit_login(user_id_entry)
        )
        submit_button.grid(row=1, column=0, columnspan=2, pady=10)

        # Add any additional logic for login button press

    def show_register_fields(self):
        # Clear previous input fields if any
        if self.input_frame:
            self.input_frame.destroy()

        # Create a frame for register inputs
        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.pack(pady=20)

        # Name Field
        name_label = ctk.CTkLabel(self.input_frame, text="Name:")
        name_label.grid(row=0, column=0, padx=5, pady=5)

        name_entry = ctk.CTkEntry(self.input_frame)
        name_entry.grid(row=0, column=1, padx=5, pady=5)

        # Email Field
        email_label = ctk.CTkLabel(self.input_frame, text="Email:")
        email_label.grid(row=1, column=0, padx=5, pady=5)

        email_entry = ctk.CTkEntry(self.input_frame)
        email_entry.grid(row=1, column=1, padx=5, pady=5)

        dob_label = ctk.CTkLabel(self.input_frame, text="Date of Birth (DDMMYYYY):")
        dob_label.grid(row=2, column=0, padx=5, pady=5)

        dob_entry = ctk.CTkEntry(self.input_frame)
        dob_entry.grid(row=2, column=1, padx=5, pady=5)

        # Submit Button
        submit_button = ctk.CTkButton(
            self.input_frame,
            text="Enter",
            command=lambda: self.submit_register(name_entry, email_entry, dob_entry)
        )
        submit_button.grid(row=3, column=0, columnspan=2, pady=10)

        # Add any additional logic for register button press

    def submit_login(self, user_id_entry):
        user_id = user_id_entry.get().strip()
        user_exists = False

        with open(self.filepath, mode="r") as file:
            reader = csv.DictReader(file)
            print(reader)
            for row in reader:
                if row["User ID"] == user_id:
                    user_exists = True
                    print(f'User ID: {user_id} exists')
                    IntroWindow.n_value = row["N Value"]
                    #print(f'row N value is {row["N Value"]}')
                    print(f'Introwindow N value is {IntroWindow.n_value}')
                    break

        if user_exists:
            IntroWindow.user_id = user_id
            print(f'IntroWindow user id is {IntroWindow.user_id}')
            print(f"Login successful for User ID: {user_id}")
            ctk.CTkLabel(self.input_frame, text="Login successful!", fg_color="green").grid(row=2, column=0, columnspan=2, pady=5)
            self.destroy()
            if int(self.n_value) >1:
                start_main_app()
            else:
                start_practice()
        else:
            ctk.CTkLabel(self.input_frame, text="User ID not found!", fg_color="red").grid(row=2, column=0, columnspan=2, pady=5)

    def submit_register(self, name_entry, email_entry, dob_entry):
        name = name_entry.get().strip()
        email = email_entry.get().strip()
        dob = dob_entry.get().strip()

        # Validate Name
        if not name.isalpha():
            ctk.CTkLabel(self.input_frame, text="Invalid Name! Only alphabets allowed.", fg_color="red").grid(row=4, column=0, columnspan=2, pady=5)
            return

        # Validate Email
        if not email.endswith("@gmail.com"):
            ctk.CTkLabel(self.input_frame, text="Invalid Email! Must end with @gmail.com", fg_color="red").grid(row=5, column=0, columnspan=2, pady=5)
            return

        # Validate Date of Birth
        if not re.match(r"^\d{8}$", dob):  # Check if DoB is exactly 8 digits
            ctk.CTkLabel(self.input_frame, text="Invalid DoB! Format: DDMMYYYY", fg_color="red").grid(row=6, column=0, columnspan=2, pady=5)
            return

        # Generate User ID
        user_id = f"{name}_{dob}"

        # Check if user already exists
        user_exists = False
        with open(self.filepath, mode="r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row["User ID"] == user_id or row["Email"] == email:
                    user_exists = True
                    break
        if user_exists:
            ctk.CTkLabel(self.input_frame, text="User already exists!", fg_color="red").grid(row=7, column=0, columnspan=2, pady=5)
            return
        
        # Save details to CSV if user doesn't exist
        with open(self.filepath, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([name, email, dob, user_id, 1])

        print(f"Register successful for User ID: {user_id}")
        ctk.CTkLabel(self.input_frame, text="Registration successful!", fg_color="green").grid(row=7, column=0, columnspan=2, pady=5)

        self.input_frame.destroy()
        self.show_login_fields()

class MainApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Shape Display Application")
        self.geometry("800x600")
        self.minsize(600, 500)

        # initalize list
        #self.shape_list = []
        self.shape_history = []  # List to store shape history
        self.start_time = None   # For reaction time
        self.n_value = 1
        if IntroWindow.n_value is not None:
            self.n_value = int(IntroWindow.n_value)         # Initial N value
            print(f'InroWindow n_value from MainApp:', self.n_value)
        self.correct_count = 0
        self.incorrect_responses = 0
        self.total_count = 0
        self.accuracy_treds_reference = []

        print(f'IntroWindow n_value:', self.n_value)
        print(type(self.n_value))
        print(f'IntroWindow user_id:', IntroWindow.user_id)
        
        self.csv_file_path = os.path.abspath("metrics.csv")
        self.csv_file = open(self.csv_file_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['accuracy', 'reaction_time', 'error_ratio', 'accuracy_trend','incorrect_responses','n'])
        
        # Initialize variables
        self.previous_shape = None
        self.shape_size = 150  # Default size
        self.rotation_angle = 0  # For rotating shapes

        # Configure grid for responsive design
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Instructions
        self.instructions = ctk.CTkLabel(
            self,
            text="Welcome to the Shape Display Application.",
            font=("Arial", 16)
        )
        self.instructions.grid(row=0, column=0, pady=10, padx=10, sticky="n")

        # Main frame to hold canvas and controls
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Shape Display Canvas
        self.canvas = tk.Canvas(
            self.main_frame,
            bg="black",
            highlightthickness=0
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Controls Frame (contains buttons and slider)
        self.controls_frame = ctk.CTkFrame(self.main_frame)
        self.controls_frame.grid(row=1, column=0, pady=20, sticky="ew")

        # Same Button
        self.same_button = ctk.CTkButton(
            self.controls_frame,
            text="Same",
            command=self.display_same_shape,
            width=150
        )
        self.same_button.grid(row=0, column=0, padx=10, pady=10)

        # Different Button
        self.not_same_button = ctk.CTkButton(
            self.controls_frame,
            text="Different",
            command=self.display_not_same_shape,
            width=150
        )
        self.not_same_button.grid(row=0, column=1, padx=10, pady=10)

        
        # Size Slider Label
        self.size_slider_label = ctk.CTkLabel(
            self.controls_frame,
            text="Shape Size:",
            font=("Arial", 12)
        )
        self.size_slider_label.grid(row=0, column=2, padx=10)

        # Size Slider
        self.size_slider = ctk.CTkSlider(
            self.controls_frame,
            from_=50,
            to=300,
            command=self.update_size,
            width=150
        )
        self.size_slider.set(self.shape_size)
        self.size_slider.grid(row=0, column=3, padx=10)

        # Bind resize event for responsive design
        self.bind("<Configure>", self.on_resize)
        self.protocol("WM_DELETE_WINDOW", self.destroy_window)

        # Initial Shape Display
        self.display_random_shape()

    def on_resize(self, event):
        """Handle window resize to make the canvas responsive."""
        self.canvas_width = self.canvas.winfo_width()
        self.canvas_height = self.canvas.winfo_height()
        self.canvas.delete("all")
        if self.previous_shape:
            self.display_shape(self.previous_shape, animate=False)

    def update_size(self, value):
        """Update the size of the shape based on the slider."""
        self.shape_size = int(float(value))
        self.canvas.delete("all")
        if self.previous_shape:
            self.display_shape(self.previous_shape, animate=False)

    def generate_random_shape(self):
        """Generate a random shape and color."""
        shape_type = random.choice([
            "circle", "square", "triangle", "star", "pentagon", "rectangle", "line"
        #shape_type = random.choice([
        #    "circle", "square", "triangle", "arc", "star", "pentagon", "ellipse", "rectangle", "hexagon", "line"
        ])
        self.shape_history.append(shape_type)
        return shape_type

    def generate_random_gradient(self, shape_type):
        """
        Simulate a gradient by creating multiple overlapping shapes with varying colors.
        This is a simple approximation as Tkinter doesn't support gradients natively.
        """
        gradient_steps = 10
        base_color = random.choice([
            "#FF5733", "#33FF57", "#3357FF", "#F5FF33", "#33FFF5",
            "#FF33A8", "#A833FF", "#33FFD5", "#FF8C33", "#8C33FF"
        ])
        # Convert hex to RGB
        base_color_rgb = self.hex_to_rgb(base_color)
        gradient_colors = [
            self.rgb_to_hex([
                min(255, base_color_rgb[0] + i * 15),
                min(255, base_color_rgb[1] + i * 15),
                min(255, base_color_rgb[2] + i * 15)
            ]) for i in range(gradient_steps)
        ]
        return gradient_colors

    def hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    def rgb_to_hex(self, rgb_color):
        """Convert RGB tuple to hex color."""
        return "#{:02x}{:02x}{:02x}".format(*rgb_color)

    def animate_shape(self, shape_type, gradient_colors, rotation=False):
        """Animate the drawing of a shape."""
        self.canvas.delete("all")
        steps = len(gradient_colors)
        delay = 50  # milliseconds

        def draw_step(step):
            if step > steps:
                return
            color = gradient_colors[step - 1]
            size = int((self.shape_size / steps) * step)
            if rotation:
                angle = (self.rotation_angle / steps) * step
                self.draw_shape(shape_type, size, angle, color)
            else:
                self.draw_shape(shape_type, size, 0, color)
            self.after(delay, lambda: draw_step(step + 1))

        draw_step(1)

    def draw_shape(self, shape_type, size, angle, color):
        """Draw the specified shape with given size, rotation, and color."""
        self.canvas.create_text(
                400, 20,  # Coordinates for the text (centered)
                text=f"Current N Value {self.n_value} ",
                fill="white",
                font=("Arial", 14)
            )
        cx, cy = self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2
        x1, y1 = cx - size, cy - size
        x2, y2 = cx + size, cy + size

        # Calculate rotated coordinates if angle is provided
        def rotate_point(x, y, cx, cy, angle):
            radians = math.radians(angle)
            cos_val = math.cos(radians)
            sin_val = math.sin(radians)
            x -= cx
            y -= cy
            x_new = x * cos_val - y * sin_val
            y_new = x * sin_val + y * cos_val
            x_new += cx
            y_new += cy
            return x_new, y_new

        if shape_type == "circle":
            self.canvas.create_oval(x1, y1, x2, y2, fill=color, outline="")
        elif shape_type == "square":
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
        elif shape_type == "triangle":
            points = [
                (cx, y1),
                (x1, y2),
                (x2, y2)
            ]
            if angle != 0:
                points = [rotate_point(x, y, cx, cy, angle) for x, y in points]
            self.canvas.create_polygon(points, fill=color, outline="")
        elif shape_type == "arc":
            self.canvas.create_arc(x1, y1, x2, y2, start=0, extent=150, fill=color, outline="")
        elif shape_type == "star":
            points = self.calculate_star_points(cx, cy, size, size / 2, 5, angle)
            self.canvas.create_polygon(points, fill=color, outline="")
        elif shape_type == "pentagon":
            points = self.calculate_polygon_points(cx, cy, size, 5, angle)
            self.canvas.create_polygon(points, fill=color, outline="")
        elif shape_type == "ellipse":
            self.canvas.create_oval(x1 + size * 0.2, y1, x2 - size * 0.2, y2, fill=color, outline="")
        elif shape_type == "rectangle":
            self.canvas.create_rectangle(x1 + size * 0.2, y1, x2 - size * 0.2, y2, fill=color, outline="")
        elif shape_type == "hexagon":
            points = self.calculate_polygon_points(cx, cy, size, 6, angle)
            self.canvas.create_polygon(points, fill=color, outline="")
        elif shape_type == "line":
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=5)

    def calculate_polygon_points(self, cx, cy, size, num_sides, angle):
        """Calculate the vertices of a regular polygon."""
        points = []
        for i in range(num_sides):
            theta = (2 * math.pi / num_sides) * i + math.radians(angle)
            x = cx + size * math.cos(theta)
            y = cy + size * math.sin(theta)
            points.extend([x, y])
        return points

    def calculate_star_points(self, cx, cy, outer_size, inner_size, num_points, angle):
        """Calculate the vertices of a star shape."""
        points = []
        for i in range(2 * num_points):
            r = outer_size if i % 2 == 0 else inner_size
            theta = (math.pi / num_points) * i + math.radians(angle)
            x = cx + r * math.cos(theta)
            y = cy + r * math.sin(theta)
            points.extend([x, y])
        return points

    def display_shape(self, shape_data, animate=True, rotate=False):
        """Display a shape on the canvas, optionally with animation."""
        shape_type, _ = shape_data
        gradient_colors = self.generate_random_gradient(shape_type)
        if animate:
            self.animate_shape(shape_type, gradient_colors, rotation=rotate)
        else:
            # Display without animation
            color = gradient_colors[-1]
            self.draw_shape(shape_type, self.shape_size, self.rotation_angle if rotate else 0, color)
        self.previous_shape = shape_data
        
        #self.shape_list.append(self.previous_shape)

    def display_random_shape(self):
        self.start_time = time.time() 
        shape_type = self.generate_random_shape()
        shape_data = (shape_type, None)
        self.display_shape(shape_data, animate=True, rotate=False)

    def calculate_accuracy_trend(self, accuracies):
        """
        Calculate the accuracy trend using linear regression.
        :param accuracies: List of accuracy values over trials.
        :return: Slope of the trend (positive = improving, negative = declining).
        """
        trials = np.arange(1, len(accuracies) + 1)  # Trial numbers
        slope, _ = np.polyfit(trials, accuracies, 1)  # Linear regression
        return slope

    def check_response(self, is_correct, is_match):
        reaction_time = time.time() - self.start_time
        #print(reaction_time)
        self.total_count += 1

        if is_correct:
            self.correct_count += 1
        else:
            self.incorrect_responses += 1

        # Calculate metrics
        accuracy = (self.correct_count / self.total_count) * 100 if self.total_count > 0 else 0
        self.accuracy_treds_reference.append(accuracy)
        accuracy_treds = self.calculate_accuracy_trend(self.accuracy_treds_reference)

        #error_ratio = 1 - (self.correct_count / self.total_count)
        error_ratio = self.incorrect_responses / self.total_count if self.total_count > 0 else 0

        print(f"Accuracy: {accuracy:.2f}%, Reaction Time: {reaction_time:.4f}s, Error Ratio: {error_ratio:.2f}, accuracy_treds: {accuracy_treds : .2f}, incorrect_responses:{self.incorrect_responses}, N vlaue: {self.n_value}")

        self.csv_file_path = os.path.abspath("metrics.csv")
        self.csv_file = open(self.csv_file_path, "a", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([accuracy, reaction_time, error_ratio,  accuracy_treds, self.incorrect_responses,self.n_value])


        self.model_path=os.path.abspath("decision_tree_model.pkl")
        self.model_predictor = ModelPredictor(self.model_path)
        # Proceed to the next N value if a trial is completed
        if self.total_count >= 25:  
            result = self.model_predictor.predict_from_csv(self.csv_file_path)
            print("Predicted class for the dataset:", result)
            self.n_value = self.n_value + int(result)
            if self.n_value ==  0 :
                self.n_value = 1
            self.correct_count = 0
            self.total_count = 0
            self.incorrect_responses = 0
            self.copy_csv_contents(self.csv_file_path, 'graph.csv')
            self.clear_csv()
            
    def copy_csv_contents(self, source_file, target_file):
        """
        Copy contents of source_file to target_file, skipping the header if necessary.
        """
        # Read data from the source file
        source_data = self._read_csv(source_file)

        # Check if the target file has a header
        source_data = source_data[1:] if self.has_header_row(target_file) else source_data

        # Append data to the target file
        self._write_csv(target_file, source_data)

    def _read_csv(self, file_path):
        try:
            with open(file_path, "r", newline="", encoding="utf-8") as file:
                reader = csv.reader(file)
                return list(reader)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return []

    def _write_csv(self, file_path, data):
        try:
            with open(file_path, "a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerows(data)
        except Exception as e:
            print(f"Error writing to file: {file_path}. Error: {e}")

    def has_header_row(self, file_path):
        """
        Check if the file contains a header row.
        """
        try:
            with open(file_path, "r") as file:
                reader = csv.reader(file)
                first_row = next(reader, None)  # Read the first row
                return first_row is not None and any(cell.isalpha() for cell in first_row)
        except FileNotFoundError:
            # If the file doesn't exist, assume no header
            return False

    def clear_csv(self):
        # Read the header first
        with open(self.csv_file_path, "r") as file:
            reader = csv.reader(file)
            header = next(reader, None)  # Extract the header row

        # Overwrite the file, writing back only the header
        with open(self.csv_file_path, "w", newline="") as file:
            writer = csv.writer(file)
            if header:
                writer.writerow(header)

    def check_shape_match(self, n_value, match):
        if len(self.shape_history) < self.n_value + 1:
            return False  # Not enough history to compare
        current_shape = self.shape_history[-1]
        comparison_shape = self.shape_history[-(self.n_value + 1)]  # Shape N steps back
        # Compare shape types
        print(current_shape[0] == comparison_shape[0])
        return current_shape[0] == comparison_shape[0]

    def check_shape_non_match(self, n_value, match):
        if len(self.shape_history) < self.n_value + 1:
            return True  # If there's not enough history, assume no match
        current_shape = self.shape_history[-1]
        comparison_shape = self.shape_history[-(self.n_value + 1)]  # Shape N steps back
        # Return True if the shapes do not match
        print(current_shape[0] != comparison_shape[0])
        return current_shape[0] != comparison_shape[0]

    def display_same_shape(self):
        #self.n_value = n_value
        self.start_time = time.time()
        #self.play_sound("wrong.mp3")  # Ensure 'not_same.mp3' exists
        self.rotation_angle = random.randint(0, 360)  # Random rotation angle
        self.display_random_shape()
        is_correct = self.check_shape_match(self.n_value, match=True)
        print(f'is_correct: {is_correct} from display_same_shape')
        self.check_response(is_correct, is_match=True)

    def display_not_same_shape(self):
        #self.n_value = n_value
        self.start_time = time.time()
        #self.play_sound("wrong.mp3")  # Ensure 'not_same.mp3' exists
        self.rotation_angle = random.randint(0, 360)  # Random rotation angle
        self.display_random_shape()
        is_correct = self.check_shape_non_match(self.n_value, match=False)
        print(f'is_correct: {is_correct} from display_not_same_shape')
        self.check_response(is_correct, is_match=False)

    def destroy_window(self):
        self.csv_file.close()
        with open('users.csv', mode="r", newline="") as file:
            reader = csv.DictReader(file)
            rows = list(reader)  # Read all rows into a list
            fieldnames = reader.fieldnames  # Save the field names

        # Modify the desired row
        for row in rows:
            if row["User ID"] == IntroWindow.user_id:
                row["N Value"] = self.n_value

        # Write the updated rows back
        with open('users.csv', mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()  # Write the header first
            writer.writerows(rows)  # Write the modified rows

        #GraphWindow(self, self.graph_csv_filepath)
        super().destroy()
        self.quit()

class P(MainApp):
    def __init__(self,main_app_instance, on_close_callback, n_value):
        super().__init__()
        self.n_value = n_value
        self.title(f"Practice N={n_value}")
        self.geometry("800x600")
        self.minsize(600, 500)

        # Configure grid for responsive design
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.on_close_callback = on_close_callback
        self.after_id = None
        self.shape_history_practice = []  # List to store shape history

        self.instructions = ctk.CTkLabel(
            self,
            text=f"N Value: {n_value}",
            font=("Arial", 16)
        )
        self.instructions.grid(row=0, column=0, pady=10, padx=10, sticky="n")

        # Main frame to hold canvas and controls
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Shape Display Canvas
        self.canvas = tk.Canvas(
            self.main_frame,
            bg="black",
            highlightthickness=0
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        

        self.main_app = main_app_instance
        print(self.shape_history)
        self.bind("<space>", self.on_space_press)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.shape_count = 0
        self.display_shapes(20)

    def display_shapes(self,total_shapes):
        if self.shape_count < total_shapes:
            self.shape_count += 1
            self.canvas.delete("all")
            shape_type = self.main_app.generate_random_shape()
            shape_data = (shape_type, None)
            self.shape_history_practice.append(shape_type)
            cx, cy = self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2
            size = 150
            angle = 0
            color = "#FF5733"
            x1, y1 = cx - size, cy - size
            x2, y2 = cx + size, cy + size

            # Calculate rotated coordinates if angle is provided
            def rotate_point(x, y, cx, cy, angle=10):
                radians = math.radians(angle)
                cos_val = math.cos(radians)
                sin_val = math.sin(radians)
                x -= cx
                y -= cy
                x_new = x * cos_val - y * sin_val
                y_new = x * sin_val + y * cos_val
                x_new += cx
                y_new += cy
                return x_new, y_new

            if shape_type == "circle":
                self.canvas.create_oval(x1, y1, x2, y2, fill=color, outline="")
            elif shape_type == "square":
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
            elif shape_type == "triangle":
                points = [
                    (cx, y1),
                    (x1, y2),
                    (x2, y2)
                ]
                if angle != 0:
                    points = [rotate_point(x, y, cx, cy, angle) for x, y in points]
                self.canvas.create_polygon(points, fill=color, outline="")
            elif shape_type == "arc":
                self.canvas.create_arc(x1, y1, x2, y2, start=0, extent=150, fill=color, outline="")
            elif shape_type == "star":
                points = self.calculate_star_points(cx, cy, size, size / 2, 5, angle)
                self.canvas.create_polygon(points, fill=color, outline="")
            elif shape_type == "pentagon":
                points = self.calculate_polygon_points(cx, cy, size, 5, angle)
                self.canvas.create_polygon(points, fill=color, outline="")
            elif shape_type == "ellipse":
                self.canvas.create_oval(x1 + size * 0.2, y1, x2 - size * 0.2, y2, fill=color, outline="")
            elif shape_type == "rectangle":
                self.canvas.create_rectangle(x1 + size * 0.2, y1, x2 - size * 0.2, y2, fill=color, outline="")
            elif shape_type == "hexagon":
                points = self.calculate_polygon_points(cx, cy, size, 6, angle)
                self.canvas.create_polygon(points, fill=color, outline="")
            elif shape_type == "line":
                self.canvas.create_line(x1, y1, x2, y2, fill=color, width=5)

            print(f"Shape {self.shape_count} displayed.")
            # Optionally, add labels or other visuals
            self.after(1500, self.display_shapes, total_shapes)
            print(f'self.shape_history_practice: {self.shape_history_practice}')
            res = self.check_shape_match( self.n_value, match=True)
            print(f'res: {res}')
            self.canvas.create_text(
                400, 30,  # Coordinates for the text (centered)
                text=f"Displaying Shape:{self.shape_count} And MatchIng with N=1: {res}",
                fill="white",
                font=("Arial", 20)
            )
        else:
            #self.destroy()
            self.on_close()
            
    def check_shape_match(self, n_value, match):
        if len(self.shape_history_practice) < self.n_value + 1:
            return False  # Not enough history to compare
        current_shape = self.shape_history_practice[-1]
        comparison_shape = self.shape_history_practice[-(self.n_value + 1)]  # Shape N steps back
        # Compare shape types
        print(current_shape[0] == comparison_shape[0])
        return current_shape == comparison_shape

    def on_space_press(self, event):
        self.on_close()

    def on_close(self):
        if self.after_id:
            self.after_cancel(self.after_id)
        self.destroy()
        if self.on_close_callback:
            self.on_close_callback()

if __name__ == "__main__":
    def start_practice():
        practice_window = P(main_app_instance=main_app, on_close_callback=start_practice_2, n_value=1)
        practice_window.mainloop()
        practice_window.quit()

    def start_practice_2():
        practice_window = P(main_app_instance=main_app, on_close_callback=start_main_app, n_value=2)
        practice_window.mainloop()
        main_app.quit()

    def start_main_app():
        app = MainApp()
        app.mainloop()
        app.quit()

        window = GraphWindow("graph.csv")
        window.mainloop()
        window.quit()
        window.master.destroy()
        window.master.quit()
        exit()

    main_app = MainApp()
    intro_window = IntroWindow(n_value=1)
    intro_window.mainloop()