import os
import re
import csv
import math
import time
import joblib
import random
import numpy as np
import pandas as pd
import tkinter as tk
import seaborn as sns
from fpdf import FPDF
import customtkinter as ctk
from datetime import datetime
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import tkinter.messagebox as messagebox  
from sklearn.exceptions import NotFittedError
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

ctk.set_appearance_mode("dark")  # Modes: "light", "dark", "system"
ctk.set_default_color_theme("dark-blue")

Time_limit = 25*60
Time_out_time = time.time()
check_Time = None


class CSVReportGenerator2:
    def __init__(self, csv_filepath_og, Total_time):
        self.Total_time = Total_time
        self.csv_filepath = csv_filepath_og

    def load_data(self):
        if not os.path.exists(self.csv_filepath):
            print(f"Error: File not found: {self.csv_filepath}")
            return False
        print(f"File loaded successfully: {self.csv_filepath}")
        self.data = pd.read_csv(self.csv_filepath)

        required_columns = {'n', 'accuracy', 'reaction_time', 'incorrect_responses'}
        if not required_columns.issubset(self.data.columns):
            print(f"Error: CSV must contain {required_columns} columns.")
            return False

        self.data.dropna(subset=required_columns, inplace=True)
        return True

    def find_last_n1_sequence(self):
        """Find the last complete sequence of n=1 trials (25 rows) and return all rows after it."""
        n_value_to_find = 1
        last_n1_index = None

        # Find all indexes where n=1
        n1_indices = self.data[self.data['n'] == n_value_to_find].index.tolist()

        if len(n1_indices) < 25:
            print("No valid sequence of 25 trials with n=1 found.")
            return pd.DataFrame()  # Return an empty DataFrame if not enough trials

        # Find the last complete set of 25 trials where n=1
        for i in range(len(n1_indices) - 24):  # Check sequences of 25 trials
            if n1_indices[i + 24] - n1_indices[i] == 24:  # Ensure consecutive rows
                last_n1_index = n1_indices[i]  # Start index of the last valid set

        if last_n1_index is None:
            print("No valid sequence of 25 trials with n=1 found.")
            return pd.DataFrame()  # Return an empty DataFrame if not found

        # Extract all rows from this last valid `n = 1` sequence onward
        return self.data.iloc[last_n1_index:]

    def add_title(self, pdf, title):
        pdf.set_fill_color(0, 102, 204)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 18)
        pdf.cell(0, 12, title, ln=True, align="C", fill=True)
        pdf.ln(10)
    
    def add_summary(self, pdf):
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}  |  Total Duration: {self.Total_time:.2f} sec", ln=True)
        pdf.ln(5)
    
    def add_metrics_table(self, pdf, user_data):
        pdf.set_fill_color(50, 150, 50)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "overall Performance", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        self.create_table(pdf, ["Metric", "Value"], [
            ["Total Trials", str(len(user_data))],
            ["Correct %", f"{user_data['accuracy'].mean():.2f}"],
            ["Incorrect %", f"{100 - user_data['accuracy'].mean():.2f}"]
        ])
    
    def add_performance_table(self, pdf, user_data):
        pdf.set_fill_color(255, 165, 0)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Performance Metrics", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        headers = ["N-Level", "Accuracy %", "RT (ms)", "Correct %", "Incorrect %"]
        rows = []
        for n in sorted(user_data['n'].unique()):
            subset = user_data[user_data['n'] == n]
            rows.append([
                str(n),
                f"{subset['accuracy'].mean():.2f}",
                f"{subset['reaction_time'].sum():.2f}",
                f"{subset['accuracy'].mean():.2f}",
                f"{100 - subset['accuracy'].mean():.2f}"
            ])
        self.create_table(pdf, headers, rows)
    
    def add_reaction_time_analysis(self, pdf, user_data):
        pdf.set_fill_color(128, 0, 128)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Reaction Time Analysis", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 10, f"Correct Trials: {user_data[user_data['incorrect_responses'] == 0]['reaction_time'].sum():.2f} ms", ln=True)
        pdf.cell(0, 10, f"Incorrect Trials: {user_data[user_data['incorrect_responses'] > 0]['reaction_time'].sum():.2f} ms", ln=True)
        pdf.ln(5)

    def add_reaction_time_analysis(self, pdf, user_data):
        pdf.set_fill_color(128, 0, 128)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Reaction Time Analysis", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", "", 10)

        correct_trials = user_data[user_data['incorrect_responses'] == 0]['reaction_time']
        incorrect_trials = user_data[user_data['incorrect_responses'] > 0]['reaction_time']

        correct_reaction_time_sum = correct_trials.sum()
        correct_reaction_time_avg = correct_trials.mean() if not correct_trials.empty else 0  # Avoid NaN

        incorrect_reaction_time_sum = incorrect_trials.sum()
        incorrect_reaction_time_avg = incorrect_trials.mean() if not incorrect_trials.empty else 0  # Avoid NaN

        pdf.cell(0, 10, f"Correct Trials: {correct_reaction_time_sum:.2f} ms (Avg: {correct_reaction_time_avg:.2f} ms)", ln=True)
        pdf.cell(0, 10, f"Incorrect Trials: {incorrect_reaction_time_sum:.2f} ms (Avg: {incorrect_reaction_time_avg:.2f} ms)", ln=True)
        pdf.ln(5)

    def create_table(self, pdf, headers, rows):
        pdf.set_font("Arial", "B", 10)
        col_widths = [30] * len(headers)
        pdf.set_fill_color(180, 180, 180)
        pdf.set_text_color(0, 0, 0)
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 10, header, 1, 0, "C", fill=True)
        pdf.ln()
        pdf.set_font("Arial", "", 10)
        for row in rows:
            for i, cell in enumerate(row):
                pdf.cell(col_widths[i], 10, str(cell), 1, 0, "C")
            pdf.ln()
        pdf.ln(5)
    
    def plot_correct_incorrect_responses(self, user_data):
        plt.figure(figsize=(8, 5))
        correct = user_data.groupby('n')['accuracy'].mean()
        incorrect = 100 - correct
        
        plt.bar(correct.index, correct, color='green', label='Correct %')
        plt.bar(incorrect.index, incorrect, bottom=correct, color='red', label='Incorrect %')
        plt.xlabel('N-Level')
        plt.ylabel('Percentage')
        plt.title('Correct vs Incorrect Responses by N-Level')
        plt.xticks(correct.index)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join('correct_incorrect_responses.png'))
        plt.close()
    
    def plot_reaction_time(self, user_data):
        plt.figure(figsize=(8, 5))
        correct_rt = user_data[user_data['incorrect_responses'] == 0].groupby('n')['reaction_time'].mean()
        incorrect_rt = user_data[user_data['incorrect_responses'] > 0].groupby('n')['reaction_time'].mean()
        
        plt.plot(correct_rt.index, correct_rt, marker='o', linestyle='-', color='blue', label='Correct Trials')
        plt.plot(incorrect_rt.index, incorrect_rt, marker='s', linestyle='-', color='orange', label='Incorrect Trials')
        plt.xlabel('N-Level')
        plt.ylabel('Reaction Time (ms)')
        plt.title('Reaction Time for Correct vs Incorrect Trials')
        plt.xticks(correct_rt.index)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join('reaction_time_analysis.png'))
        plt.close()
    
    def add_graphs_to_pdf(self, pdf):
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Graphical Analysis", ln=True, align='C')
        pdf.ln(5)
        pdf.image(os.path.join('correct_incorrect_responses.png'), x=10, w=190)
        pdf.ln(10)
        pdf.image(os.path.join('reaction_time_analysis.png'), x=10, w=190)
    
    def create_pdf(self, user_data):
        self.plot_correct_incorrect_responses(user_data)
        self.plot_reaction_time(user_data)
        
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        self.add_title(pdf, "N-Back Task Report")
        self.add_summary(pdf)
        self.add_metrics_table(pdf, user_data)
        self.add_performance_table(pdf, user_data)
        self.add_reaction_time_analysis(pdf, user_data)
        self.add_graphs_to_pdf(pdf)
        
        pdf.output(f"report_of_Simple_Task.pdf")
        print(f"Report saved as report_of_Simple_Task.pdf")

    def generate_reports(self):
        if not self.load_data():
            return

        user_data = self.find_last_n1_sequence()
        if user_data.empty:
            print("No valid data found for n=1.")
            return

        self.create_pdf(user_data)


class CSVReportGenerator:
    def __init__(self, csv_filepath_og, Total_time):
        self.Total_time = Total_time
        self.csv_filepath_og = csv_filepath_og
        self.data = None
        self.keep_last_user_consecutive_data("filtered_file.csv")
        self.csv_filepath = 'filtered_file.csv'

    def load_data(self):
        if not os.path.exists(self.csv_filepath):
            print(f"Error: File not found: {self.csv_filepath}")
            return False
        
        self.data = pd.read_csv(self.csv_filepath)
        required_columns = {'User ID', 'n', 'accuracy', 'reaction_time', 'incorrect_responses'}
        if not required_columns.issubset(self.data.columns):
            print(f"Error: CSV must contain {required_columns} columns.")
            return False

        self.data.dropna(subset=required_columns, inplace=True)
        return True
    
    def keep_last_user_consecutive_data(self, output_filepath="filtered_data.csv"):
        df = pd.read_csv(self.csv_filepath_og)

        if "User ID" not in df.columns:
            raise ValueError("The CSV file must have a 'user' column.")

        last_user = df.iloc[-1]["User ID"]

        last_index = len(df) - 1
        while last_index >= 0 and df.iloc[last_index]["User ID"] == last_user:
            last_index -= 1
        
        filtered_df = df.iloc[last_index + 1:]

        filtered_df.to_csv(output_filepath, index=False)
        print(f"Filtered data saved to {output_filepath}")

    def add_title(self, pdf, title):
        pdf.set_fill_color(0, 102, 204)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 18)
        pdf.cell(0, 12, title, ln=True, align="C", fill=True)
        pdf.ln(10)
    
    def add_summary(self, pdf, user_id):
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}  |  ID: {user_id}  |  Total Duration: {self.Total_time:.2f} sec", ln=True)
        pdf.ln(5)
    
    def add_metrics_table(self, pdf, user_data):
        pdf.set_fill_color(50, 150, 50)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "overall Performance", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        self.create_table(pdf, ["Metric", "Value"], [
            ["Total Trials", str(len(user_data))],
            ["Correct %", f"{user_data['accuracy'].mean():.2f}"],
            ["Incorrect %", f"{100 - user_data['accuracy'].mean():.2f}"]
        ])
    
    def add_performance_table(self, pdf, user_data):
        pdf.set_fill_color(255, 165, 0)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Performance Metrics", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        headers = ["N-Level", "Accuracy %", "RT (ms)", "Correct %", "Incorrect %"]
        rows = []
        for n in sorted(user_data['n'].unique()):
            subset = user_data[user_data['n'] == n]
            rows.append([
                str(n),
                f"{subset['accuracy'].mean():.2f}",
                f"{subset['reaction_time'].sum():.2f}",
                f"{subset['accuracy'].mean():.2f}",
                f"{100 - subset['accuracy'].mean():.2f}"
            ])
        self.create_table(pdf, headers, rows)
    
    def add_reaction_time_analysis(self, pdf, user_data):
        pdf.set_fill_color(128, 0, 128)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Reaction Time Analysis", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 10, f"Correct Trials: {user_data[user_data['incorrect_responses'] == 0]['reaction_time'].sum():.2f} ms", ln=True)
        pdf.cell(0, 10, f"Incorrect Trials: {user_data[user_data['incorrect_responses'] > 0]['reaction_time'].sum():.2f} ms", ln=True)
        pdf.ln(5)
    
    def track_n_sequence(self, user_data):
        n_sequence_summary = []
        count = 0
        prev_n = None
        temp_data = []

        for index, row in user_data.iterrows():
            if prev_n is None or count == 25:
                if temp_data:
                    avg_accuracy = sum(d['accuracy'] for d in temp_data) / len(temp_data)
                    avg_rt = sum(d['reaction_time'] for d in temp_data) / len(temp_data)
                    sum_rt = sum(d['reaction_time'] for d in temp_data)
                    correct_pct = avg_accuracy
                    incorrect_pct = 100 - avg_accuracy
                    n_sequence_summary.append([prev_n, f"{avg_accuracy:.2f}", f"{avg_rt:.2f}", f"{sum_rt:.2f}", f"{correct_pct:.2f}", f"{incorrect_pct:.2f}"])
                temp_data = []
                count = 1
            else:
                count += 1
            
            temp_data.append({'accuracy': row['accuracy'], 'reaction_time': row['reaction_time']})
            prev_n = row['n']
        
        if temp_data:
            avg_accuracy = sum(d['accuracy'] for d in temp_data) / len(temp_data)
            avg_rt = sum(d['reaction_time'] for d in temp_data) / len(temp_data)
            sum_rt = sum(d['reaction_time'] for d in temp_data)
            correct_pct = avg_accuracy
            incorrect_pct = 100 - avg_accuracy
            n_sequence_summary.append([prev_n, f"{avg_accuracy:.2f}", f"{avg_rt:.2f}", f"{sum_rt:.2f}", f"{correct_pct:.2f}", f"{incorrect_pct:.2f}"])
        
        return n_sequence_summary

    def add_n_sequence_table(self, pdf, user_data):
        pdf.set_fill_color(0, 0, 255)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "N-Level Sequence Summary", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        headers = ["N-Level", "Accuracy %", "Avg RT (ms)", "Sum RT (ms)", "Correct %", "Incorrect %"]
        rows = self.track_n_sequence(user_data)
        self.create_table(pdf, headers, rows)

    def add_reaction_time_analysis(self, pdf, user_data):
        pdf.set_fill_color(128, 0, 128)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Reaction Time Analysis", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", "", 10)

        correct_trials = user_data[user_data['incorrect_responses'] == 0]['reaction_time']
        incorrect_trials = user_data[user_data['incorrect_responses'] > 0]['reaction_time']

        correct_reaction_time_sum = correct_trials.sum()
        correct_reaction_time_avg = correct_trials.mean() if not correct_trials.empty else 0  # Avoid NaN

        incorrect_reaction_time_sum = incorrect_trials.sum()
        incorrect_reaction_time_avg = incorrect_trials.mean() if not incorrect_trials.empty else 0  # Avoid NaN

        pdf.cell(0, 10, f"Correct Trials: {correct_reaction_time_sum:.2f} ms (Avg: {correct_reaction_time_avg:.2f} ms)", ln=True)
        pdf.cell(0, 10, f"Incorrect Trials: {incorrect_reaction_time_sum:.2f} ms (Avg: {incorrect_reaction_time_avg:.2f} ms)", ln=True)
        pdf.ln(5)

    def create_table(self, pdf, headers, rows):
        pdf.set_font("Arial", "B", 10)
        col_widths = [30] * len(headers)
        pdf.set_fill_color(180, 180, 180)
        pdf.set_text_color(0, 0, 0)
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 10, header, 1, 0, "C", fill=True)
        pdf.ln()
        pdf.set_font("Arial", "", 10)
        for row in rows:
            for i, cell in enumerate(row):
                pdf.cell(col_widths[i], 10, str(cell), 1, 0, "C")
            pdf.ln()
        pdf.ln(5)
    
    def plot_correct_incorrect_responses(self, user_data):
        plt.figure(figsize=(8, 5))
        correct = user_data.groupby('n')['accuracy'].mean()
        incorrect = 100 - correct
        
        plt.bar(correct.index, correct, color='green', label='Correct %')
        plt.bar(incorrect.index, incorrect, bottom=correct, color='red', label='Incorrect %')
        plt.xlabel('N-Level')
        plt.ylabel('Percentage')
        plt.title('Correct vs Incorrect Responses by N-Level')
        plt.xticks(correct.index)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join('correct_incorrect_responses.png'))
        plt.close()
    
    def plot_reaction_time(self, user_data):
        plt.figure(figsize=(8, 5))
        correct_rt = user_data[user_data['incorrect_responses'] == 0].groupby('n')['reaction_time'].mean()
        incorrect_rt = user_data[user_data['incorrect_responses'] > 0].groupby('n')['reaction_time'].mean()
        
        plt.plot(correct_rt.index, correct_rt, marker='o', linestyle='-', color='blue', label='Correct Trials')
        plt.plot(incorrect_rt.index, incorrect_rt, marker='s', linestyle='-', color='orange', label='Incorrect Trials')
        plt.xlabel('N-Level')
        plt.ylabel('Reaction Time (ms)')
        plt.title('Reaction Time for Correct vs Incorrect Trials')
        plt.xticks(correct_rt.index)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join('reaction_time_analysis.png'))
        plt.close()
    
    def add_graphs_to_pdf(self, pdf):
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Graphical Analysis", ln=True, align='C')
        pdf.ln(5)
        pdf.image(os.path.join('correct_incorrect_responses.png'), x=10, w=190)
        pdf.ln(10)
        pdf.image(os.path.join('reaction_time_analysis.png'), x=10, w=190)
    
    def create_pdf(self, user_data, user_id):
        self.plot_correct_incorrect_responses(user_data)
        self.plot_reaction_time(user_data)
        
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        self.add_title(pdf, "N-Back Task Report")
        self.add_summary(pdf, user_id)
        self.add_metrics_table(pdf, user_data)
        self.add_performance_table(pdf, user_data)
        self.add_n_sequence_table(pdf, user_data)
        self.add_reaction_time_analysis(pdf, user_data)
        self.add_graphs_to_pdf(pdf)
        
        pdf.output(f"report_{user_id}.pdf")
        print(f"Report saved as report_{user_id}.pdf")
    
    def generate_reports(self):
        if not self.load_data():
            return
        last_user_id = self.data['User ID'].iloc[-1]
        user_data = self.data[self.data['User ID'] == last_user_id]
        self.create_pdf(user_data, last_user_id)


class ModelPredictor:
    def __init__(self, model_path, required_columns=None, features=None):
        self.model_path = model_path
        self.required_columns = required_columns or ["accuracy", "reaction_time"]
        self.features = features or ["accuracy", "reaction_time"]
        
        self.model = self.load_model()
        
    def load_model(self):
        try:
            return joblib.load(self.model_path)
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")
    
    def predict_from_csv(self, csv_file):
        try:
            if not os.path.isfile(csv_file):
                raise FileNotFoundError(f"CSV file '{csv_file}' does not exist.")

            data = pd.read_csv(csv_file)

            if data.empty:
                raise ValueError("The CSV file is empty.")

            missing_columns = [col for col in self.required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in CSV: {', '.join(missing_columns)}")

            data = data.dropna(subset=self.required_columns)

            if data.empty:
                raise ValueError("No data left after removing rows with missing values in required columns.")

            missing_features = [feature for feature in self.features if feature not in data.columns]
            if missing_features:
                raise ValueError(f"Missing required features for prediction: {', '.join(missing_features)}")

            feature_data = data[self.features].mean()
            print(f'mean of Feature data: {feature_data}')
            feature_data = pd.DataFrame([feature_data])
            feature_data  = feature_data.apply(pd.to_numeric)

            predictions = self.model.predict(feature_data)

            print(f'Predictions: {predictions}')
            return predictions[0]

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


class NBackApp(ctk.CTk):
    def __init__(self, n_value=1):
        super().__init__()
        self.title("Simple N-Back Task Application")
        self.geometry("800x600")
        self.minsize(600, 500)
        
        self.n_value = n_value
        self.output_value_n = None
        self.output_color= None

        self.accuracy_trend_array = []
        self.correct_count = 0
        self.incorrect_responses = 0
        self.total_count = 0
        self.shape_history = []
        self.button_pressed = None

        self.csv_file_path = os.path.abspath("ST_metrics.csv")
        if os.stat(self.csv_file_path).st_size == 0:
            self.csv_file = open(self.csv_file_path, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['User ID', 'accuracy', 'reaction_time','accuracy_trend', 'error_ratio', 'incorrect_responses', 'n', 'Button_pressed'])

        self.shape_size = 150 
        self.rotation_angle = 0 

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.create_widgets()
        self.bind("<Configure>", self.on_resize)
        self.protocol("WM_DELETE_WINDOW", self.show_popup_graph)
        self.display_random_shape()
        check_Time = time.time() - Time_out_time 
        if check_Time > Time_limit:
            print("Time exceeded")
            self.show_popup_time()
            self.show_popup_graph()
            #self.destroy_window()

    def show_popup_graph(self):
        if hasattr(self, 'popup_shown') and self.popup_shown:
            return
        self.popup_shown = True
        self.popup = ctk.CTkToplevel(self)
        self.popup.geometry("300x150")
        self.popup.title("Graph Window")

        label = ctk.CTkLabel(self.popup, text="Do you want to Analyze your Progress?")
        label.pack(pady=20)

        button_frame = ctk.CTkFrame(self.popup)
        button_frame.pack(pady=10)

        ok_button = ctk.CTkButton(button_frame, text="YES", command=self.happy)
        ok_button.pack(side="left", padx=10)

        cancel_button = ctk.CTkButton(button_frame, text="NO (exit)", command=self.destroy_window)
        cancel_button.pack(side="right", padx=10)

        self.popup.grab_set()  
        self.wait_window(self.popup)

    def happy(self):
        self.popup_shown = True
        self.popup.destroy()
        #self.destroy_window()
        #app.quit()
        self.check_T = time.time() - Time_out_time 
        report_generator2 = CSVReportGenerator2('ST_graph.csv', self.check_T)
        report_generator2.generate_reports()
        self.quit()
        self.destroy()
        exit()

    def create_widgets(self):
        self.instructions = ctk.CTkLabel(
            self,
            text="Testing, simple N Back Task.",
            font=("Arial", 16)
        )
        self.instructions.grid(row=0, column=0, pady=10, padx=10, sticky="n")

        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            self.main_frame,
            bg="black",
            highlightthickness=0
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.controls_frame = ctk.CTkFrame(self.main_frame)
        self.controls_frame.grid(row=1, column=0, pady=20, sticky="ew")

        self.same_button = ctk.CTkButton(
            self.controls_frame,
            text="Same",
            command=self.display_same_shape,
            width=150
        )
        self.same_button.grid(row=0, column=0, padx=10, pady=10)

        self.not_same_button = ctk.CTkButton(
            self.controls_frame,
            text="Different",
            command=self.display_not_same_shape,
            width=150
        )
        self.not_same_button.grid(row=0, column=1, padx=10, pady=10)

        self.back_button = ctk.CTkButton(
            self.controls_frame, 
            text="Back", 
            command=self.go_back,
            width=150,
        )
        self.back_button.grid(row=0, column=4, padx=10, pady=10)

        self.size_slider_label = ctk.CTkLabel(
            self.controls_frame,
            text="Shape Size:",
            font=("Arial", 12)
        )
        self.size_slider_label.grid(row=0, column=2, padx=10)

        self.size_slider = ctk.CTkSlider(
            self.controls_frame,
            from_=50,
            to=300,
            command=self.update_size,
            width=150
        )
        self.bind("<Configure>", self.on_resize)
        self.protocol("WM_DELETE_WINDOW", self.destroy_window)

        self.size_slider.set(self.shape_size)
        self.size_slider.grid(row=0, column=3, padx=10)

    def go_back(self):
        self.destroy()
        intro = IntroWindow(self.n_value == 1)
        intro.mainloop()

    def on_resize(self, event):
        self.canvas_width = self.canvas.winfo_width()
        self.canvas_height = self.canvas.winfo_height()
        self.canvas.delete("all")
        if self.previous_shape:
            self.display_shape(self.previous_shape, animate=False)

    def destroy_window(self):
        self.csv_file = open(self.csv_file_path, "w", newline="")
        self.csv_file.close()
        self.quit()
        self.destroy()
        exit()

    def update_size(self, value):
        self.shape_size = int(float(value))
        self.canvas.delete("all")
        if self.previous_shape:
            self.display_shape(self.previous_shape, animate=False)

    def generate_random_shape(self):
       
        shape_options = ["circle", "square", "triangle", "star", "pentagon", "rectangle", "arc"]
        probabilities = [0.1] * len(shape_options) 
        
        if len(self.shape_history) >= self.n_value:
            recent_shapes = self.shape_history[-self.n_value:]
            for i, shape in enumerate(shape_options):
                if shape in recent_shapes:
                    probabilities[i] = 0.3

        total_prob = sum(probabilities)
        probabilities = [p / total_prob for p in probabilities]

        shape_type = random.choices(shape_options, probabilities)[0]
        
        self.shape_history.append(shape_type)
        return shape_type

    def generate_random_gradient(self, shape_type):
        gradient_steps = 10
        base_color = random.choice([
            "#FF5733", "#33FF57", "#3357FF", "#F5FF33", "#33FFF5",
            "#FF33A8", "#A833FF", "#33FFD5", "#FF8C33", "#8C33FF"
        ])
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
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    def rgb_to_hex(self, rgb_color):
        return "#{:02x}{:02x}{:02x}".format(*rgb_color)

    def animate_shape(self, shape_type, gradient_colors, rotation=False):
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
        self.canvas.create_text(
                200, 25, 
                text=f"Current N Value : {self.n_value} ",
                fill="#1E90FF",
                font=("Arial", 14)
            )
        self.canvas.create_text(
                800, 25, 
                text=f"Response : {self.output_value_n} ",
                fill=self.output_color,
                font=("Arial", 14)
            )
        cx, cy = self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2
        x1, y1 = cx - size, cy - size
        x2, y2 = cx + size, cy + size

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
            points = main_app.calculate_star_points(cx, cy, size, size / 2, 5, angle)
            self.canvas.create_polygon(points, fill=color, outline="")
        elif shape_type == "pentagon":
            points = main_app.calculate_polygon_points(cx, cy, size, 5, angle)
            self.canvas.create_polygon(points, fill=color, outline="")
        elif shape_type == "ellipse":
            self.canvas.create_oval(x1 + size * 0.2, y1, x2 - size * 0.2, y2, fill=color, outline="")
        elif shape_type == "rectangle":
            self.canvas.create_rectangle(x1 + size * 0.2, y1, x2 - size * 0.2, y2, fill=color, outline="")

    def display_shape(self, shape_data, animate=True, rotate=False):
        shape_type, _ = shape_data
        gradient_colors = self.generate_random_gradient(shape_type)
        if animate:
            self.animate_shape(shape_type, gradient_colors, rotation=rotate)
        else:
            color = gradient_colors[-1]
            self.draw_shape(shape_type, self.shape_size, self.rotation_angle if rotate else 0, color)
        self.previous_shape = shape_data
        
    def display_random_shape(self):
        self.start_time = time.time() 
        shape_type = self.generate_random_shape()
        shape_data = (shape_type, None)
        self.display_shape(shape_data, animate=True, rotate=False)

    def check_match(self):
        if len(self.shape_history) <= self.n_value:
            return False
        return self.shape_history[-1] == self.shape_history[-(self.n_value + 1)]

    def display_same_shape(self):
        self.button_pressed = 'Same_shape'
        self.same_button.configure(state="disabled")
        self.rotation_angle = random.randint(0, 360)
        self.handle_response(self.check_match(), self.button_pressed)
        self.display_random_shape()
        check_Time = time.time() - Time_out_time 
        if check_Time > Time_limit:
            print("Time exceeded")
            self.show_popup_time()
            self.show_popup_graph()
            #self.destroy_window()
        self.canvas.after(400, lambda: self.same_button.configure(state="normal"))

    def display_not_same_shape(self):
        self.button_pressed = 'Diffrent_shape'
        self.not_same_button.configure(state="disabled")
        self.rotation_angle = random.randint(0, 360)
        self.handle_response(not self.check_match(), self.button_pressed)
        self.display_random_shape()
        check_Time = time.time() - Time_out_time
        if check_Time  > Time_limit:
            print("Time exceeded")
            self.show_popup_time()
            self.show_popup_graph()
            #self.destroy_window()
        self.canvas.after(400, lambda: self.not_same_button.configure(state="normal"))

    def handle_response(self, is_correct, button_pressed):
        print(f'IntroWindow user_id:', IntroWindow.user_id)
        reaction_time = time.time() - self.start_time
        self.total_count += 1
        if is_correct:
            self.output_value_n = "Correct"
            self.output_color = "green"
            self.correct_count += 1
        else:
            self.output_value_n = "Incorrect"
            self.output_color = "red"
            self.incorrect_responses += 1

        accuracy = (self.correct_count / self.total_count) * 100
        self.accuracy_trend_array.append(accuracy)
        accuracy_trend = self.instantaneous_accuracy_trend(self.accuracy_trend_array)
        error_ratio = self.incorrect_responses / self.total_count
        print(f"N value: {self.n_value}, Accuracy: {accuracy:.2f}%, Reaction Time: {reaction_time:.4f}s,Accuracy_trend :{accuracy_trend[-1]:.4f}, Error Ratio: {error_ratio:.2f}")

        self.csv_file = open(self.csv_file_path, "a", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([IntroWindow.user_id, accuracy, reaction_time, accuracy_trend[-1], error_ratio, self.incorrect_responses,  self.n_value, button_pressed])
        self.csv_file.flush()

        if self.total_count > 24:
            self.n_value = self.n_value + 1
            self.output_value_n = None
            self.show_popup()
            self.shape_history.clear()
            self.accuracy_trend_array.clear()
            self.correct_count = 0
            self.total_count = 0
            self.incorrect_responses = 0
            self.copy_csv_contents(self.csv_file_path, 'ST_graph.csv')
            self.csv_file = open(self.csv_file_path, "w", newline="")
            self.csv_writer.writerow(['User ID', 'accuracy', 'reaction_time','accuracy_trend', 'error_ratio', 'incorrect_responses', 'n', 'Button_pressed'])
    
    def copy_csv_contents(self, source_file, target_file):
        source_data = self._read_csv(source_file)
        source_data = source_data[1:] if self.has_header_row(target_file) else source_data
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
        try:
            with open(file_path, "r") as file:
                reader = csv.reader(file)
                first_row = next(reader, None)  
                return first_row is not None and any(cell.isalpha() for cell in first_row)
        except FileNotFoundError:
            return False

    def instantaneous_accuracy_trend(self,accuracy_values):
        return np.diff(accuracy_values, prepend=accuracy_values[0])

    def show_popup(self):
        self.popup = ctk.CTkToplevel(self)
        self.popup.geometry("300x150")
        self.popup.title("Notification")
        self.output_value = None

        if self.n_value > 5:
            self.n_value = 1
            label = ctk.CTkLabel(self.popup, text=f" Max V value Achived.Resetting N to 1. ")
            label.pack(pady=20)

        else:
            label = ctk.CTkLabel(self.popup, text=f"N value changes to : {self.n_value}")
            label.pack(pady=20)

        button = ctk.CTkButton(self.popup, text="Close", command=self.popup.destroy)
        button.pack(pady=10)

        self.popup.grab_set()
        self.wait_window(self.popup)
        
    def show_popup_time(self):
        self.popup = ctk.CTkToplevel(self)
        self.popup.geometry("300x150")
        self.popup.title("Time Out")

        label = ctk.CTkLabel(self.popup, text=f"Max Time Limit Exceed")
        label.pack(pady=20)

        button = ctk.CTkButton(self.popup, text="Close", command=self.popup.destroy)
        button.pack(pady=10)

        self.popup.grab_set() 
        self.wait_window(self.popup)


class IntroWindow(ctk.CTk):
    user_id = None
    n_value = None
    def __init__(self, n_value):
        super().__init__()
        self.title("Welcome To N Back Task Application")
        self.minsize(800, 600)

        self.after_id = None
        self.filepath = "users.csv"

        self.label = ctk.CTkLabel(
            self,
            text=(
                "✨ **Adaptive N-Back Task** ✨\n\n"
                "**How It Works:**\n"
                "—————————————\n"
                "✅ Watch the sequence of shapes.\n"
                "✅ Compare with the shape **N steps ago**.\n"
                "✅ Press **'Same'** if it matches, **'Different'** if not.\n"
                "✅ Difficulty changes based on your accuracy.\n\n"
                "🚀 **Train your memory. Stay focused. Beat the challenge!\n\n**"

                "**Instruction**\n"
                "At start of every Round (when N value Changes)"
                "Press Different Button For the Same Number of times as your N value."

            ),
            font=("Arial", 18),
            justify="center",
            wraplength=700,
            text_color="#00FF7F"
        )
        self.label.pack(expand=True, padx=20, pady=20)

        self.protocol("WM_DELETE_WINDOW", self.close_window)

        self.buttons_frame = None
        self.input_frame = None
        
        self.add_initial_buttons()

        if not os.path.exists(self.filepath):
            with open(self.filepath, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Name", "DoB", "User ID", "N Value"])

    def add_initial_buttons(self):
        self.buttons_frame = ctk.CTkFrame(self)
        self.buttons_frame.pack(pady=20)

        self.mode_label = ctk.CTkLabel(
            self.buttons_frame,
            text="Choose the mode you want to play",
            font=("Arial", 18, "bold"), 
            text_color = "#4682B4",
        )
        self.mode_label.grid(row=0, column=0, columnspan=2, pady=(0, 10),)

        self.ant_button = ctk.CTkButton(
            self.buttons_frame,
            text="Adaptive NBT",
            command=self.show_login_or_register,
            fg_color="#1E90FF",  
            hover_color="#187BCD", 
            text_color="white" 
        )
        self.ant_button.grid(row=1, column=0, padx=10,pady = 5)

        self.st_button = ctk.CTkButton(
            self.buttons_frame,
            text="Simple NBT",
            command=self.dont_add_anything,
            fg_color="#1E90FF", 
            hover_color="#187BCD",  
            text_color="white" 
        )
        self.st_button.grid(row=1, column=1, padx=10,pady = 5)

    def dont_add_anything(self):
        self.destroy()
        app = NBackApp()
        app.mainloop()

    def show_login_or_register(self):
        self.buttons_frame.destroy()
        self.buttons_frame = ctk.CTkFrame(self)
        self.buttons_frame.pack(pady=20)
        self.button_1 = ctk.CTkButton(
            self.buttons_frame,
            text="Login",
            command=self.show_login_fields
        )
        self.button_1.grid(row=0, column=0, padx=10,pady =5)
        self.button_2 = ctk.CTkButton(
            self.buttons_frame,
            text="Sign Up",
            command=self.show_register_fields
        )
        self.button_2.grid(row=0, column=1, padx=10,pady =5)

        self.back_button = ctk.CTkButton(
            self.buttons_frame, 
            text="Back", 
            command=self.go_back
        )
        self.back_button.grid(row=0, column=2, padx=10, sticky="ne", pady = 5)

    def go_back(self):
        self.buttons_frame.destroy()
        self.add_initial_buttons() 

    def close_window(self):
        self.quit()
        self.destroy()

    def show_login_fields(self):
        self.buttons_frame.destroy()
        if self.input_frame:
            self.input_frame.destroy()

        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.pack(pady=20)

        user_id_label = ctk.CTkLabel(self.input_frame, text="User ID:")
        user_id_label.grid(row=0, column=0, padx=5, pady=5)

        user_id_entry = ctk.CTkEntry(self.input_frame, )
        user_id_entry.grid(row=0, column=1, padx=5, pady=5)
        #user_id_entry.insert(0, "a_11111111") ################

        submit_button = ctk.CTkButton(
            self.input_frame,
            text="Enter",
            command=lambda: self.submit_login(user_id_entry)
        )
        submit_button.grid(row=1, column=0, pady=10,padx =5)

        self.back_button = ctk.CTkButton(
            self.input_frame, 
            text="Back", 
            command=self.go_back_1,
            width=150,
        )
        self.back_button.grid(row=1, column=1, pady=10,padx =5)

    def show_register_fields(self):
        self.buttons_frame.destroy()
        if self.input_frame:
            self.input_frame.destroy()

        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.pack(pady=20)

        name_label = ctk.CTkLabel(self.input_frame, text="Name:")
        name_label.grid(row=0, column=0, padx=5, pady=5)

        name_entry = ctk.CTkEntry(self.input_frame)
        name_entry.grid(row=0, column=1, padx=5, pady=5)

        dob_label = ctk.CTkLabel(self.input_frame, text="Date of Birth (DDMMYYYY):")
        dob_label.grid(row=2, column=0, padx=5, pady=5)

        dob_entry = ctk.CTkEntry(self.input_frame)
        dob_entry.grid(row=2, column=1, padx=5, pady=5)

        submit_button = ctk.CTkButton(
            self.input_frame,
            text="Enter",
            command=lambda: self.submit_register(name_entry, dob_entry)
        )
        submit_button.grid(row=3, column=0, pady=10)

        self.back_button = ctk.CTkButton(
            self.input_frame, 
            text="Back", 
            command=self.go_back_1,
            width=150,
        )
        self.back_button.grid(row=3, column=1, pady=10)

    def go_back_1(self):
        self.input_frame.destroy()
        self.add_initial_buttons()

    def submit_login(self, user_id_entry):
        user_id = user_id_entry.get().strip()
        user_exists = False

        with open(self.filepath, mode="r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row["User ID"] == user_id:
                    user_exists = True
                    IntroWindow.n_value = row["N Value"]
                    n_value = IntroWindow.n_value 
                    break

        if user_exists:
            IntroWindow.user_id = user_id
            ctk.CTkLabel(self.input_frame, text="Login successful!", fg_color="green").grid(row=2, column=0, columnspan=2, pady=5)
            if int(n_value) == 1:
                self.destroy()
                print('Starting practic')
                start_practice()
            if int(n_value) >1:
                self.input_frame.destroy()
                self.show_selection_buttons()
            else:
                start_main_app()
        else:
            ctk.CTkLabel(self.input_frame, text="User ID not found!", fg_color="red").grid(row=2, column=0, columnspan=2, pady=5)

    def show_selection_buttons(self):
        self.nvalue_frame = ctk.CTkFrame(self)
        self.nvalue_frame.pack(pady=20)

        self.mode_label = ctk.CTkLabel(
            self.nvalue_frame,
            text="Choose N value",
            font=("Arial", 18, "bold"),
            text_color = "#4682B4",
        )
        self.mode_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        self.button_1 = ctk.CTkButton(
            self.nvalue_frame,
            text="Previous",
            command=self.set_previous_n
        )
        self.button_1.grid(row=1, column=0, padx=10)

        self.button_2 = ctk.CTkButton(
            self.nvalue_frame,
            text="New",
            command=self.set_new_n
        )
        self.button_2.grid(row=1, column=1, padx=10)
        
    def set_previous_n(self):
        n_value = self.n_value
        self.nvalue_frame.destroy()
        self.destroy()
        self.write_n(IntroWindow.user_id, n_value)
        print(f"Selected Previous N: {self.n_value}")
        start_main_app()

    def set_new_n(self):
        print(f"intro window n value {IntroWindow.n_value}")
        n_value = 1
        IntroWindow.n_value = n_value
        print(f"intro window n value {IntroWindow.n_value}")
        self.write_n(IntroWindow.user_id, n_value)
        self.nvalue_frame.destroy()
        self.destroy()
        print(f"Selected New N: {n_value}")
        start_main_app()

    def write_n(self, user_id, n_value):
        print(user_id,n_value)
        with open("users.csv", mode="r", newline="") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            fieldnames = reader.fieldnames
            for row in rows:
                if row["User ID"] == user_id:
                    row["N Value"] = n_value  

        with open("users.csv", mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader() 
            writer.writerows(rows)

        print(f"N Value for User ID {user_id} updated to {n_value}")

    def submit_register(self, name_entry, dob_entry):
        name = name_entry.get().strip()
        dob = dob_entry.get().strip()

        if not name.isalpha():
            ctk.CTkLabel(self.input_frame, text="Invalid Name! Only alphabets allowed.", fg_color="red").grid(row=4, column=0, columnspan=2, pady=5)
            return

        if not re.match(r"^\d{8}$", dob): 
            ctk.CTkLabel(self.input_frame, text="Invalid DoB! Format: DDMMYYYY", fg_color="red").grid(row=6, column=0, columnspan=2, pady=5)
            return

        #user_id = f"{name}_{dob}"
        user_id = f"{name}"

        user_exists = False
        with open(self.filepath, mode="r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row["User ID"] == user_id:
                    user_exists = True
                    break
        if user_exists:
            ctk.CTkLabel(self.input_frame, text="User already exists!", fg_color="red").grid(row=7, column=0, columnspan=2, pady=5)
            return
        
        with open(self.filepath, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([name, dob, user_id, 1])

        ctk.CTkLabel(self.input_frame, text="Registration successful!", fg_color="green").grid(row=7, column=0, columnspan=2, pady=5)

        self.input_frame.destroy()
        self.show_login_fields()


class MainApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.wm_title("N Back Task Application")
        self.geometry("800x600")
        self.minsize(600, 500)

        self.shape_history = [] 
        self.start_time = None  
        self.n_value = 1
        if IntroWindow.n_value is not None:
            self.n_value = int(IntroWindow.n_value)         
        self.correct_count = 0
        self.incorrect_responses = 0
        self.total_count = 0
        self.accuracy_treds_reference = []
        self.output_value = None
        self.button_pressed = None
        print(f'IntroWindow user_id:', IntroWindow.user_id)
        
        self.csv_file_path = os.path.abspath("AT_metrics.csv")
        self.csv_file = open(self.csv_file_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['User ID', 'accuracy', 'reaction_time', 'error_ratio', 'accuracy_trend','incorrect_responses','n', 'Button_pressed'])
        
        self.previous_shape = None
        self.shape_size = 150  
        self.rotation_angle = 0  

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.instructions = ctk.CTkLabel(
            self,
            text="Testing, Adaptive N Back Task",
            font=("Arial", 16)
        )
        self.instructions.grid(row=0, column=0, pady=10, padx=10, sticky="n")

        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            self.main_frame,
            bg="black",
            highlightthickness=0
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.controls_frame = ctk.CTkFrame(self.main_frame)
        self.controls_frame.grid(row=1, column=0, pady=20, sticky="ew")

        self.same_button = ctk.CTkButton(
            self.controls_frame,
            text="Same",
            command=self.display_same_shape,
            width=150
        )
        self.same_button.grid(row=0, column=0, padx=10, pady=10)

        self.not_same_button = ctk.CTkButton(
            self.controls_frame,
            text="Different",
            command=self.display_not_same_shape,
            width=150
        )
        self.not_same_button.grid(row=0, column=1, padx=10, pady=10)

        self.back_button = ctk.CTkButton(
            self.controls_frame, 
            text="Back", 
            command=self.go_back,
            width=150,
        )
        self.back_button.grid(row=0, column=4, padx=10, pady=10)

        self.size_slider_label = ctk.CTkLabel(
            self.controls_frame,
            text="Shape Size:",
            font=("Arial", 12)
        )
        self.size_slider_label.grid(row=0, column=2, padx=10)

        self.size_slider = ctk.CTkSlider(
            self.controls_frame,
            from_=50,
            to=300,
            command=self.update_size,
            width=150
        )
        self.size_slider.set(self.shape_size)
        self.size_slider.grid(row=0, column=3, padx=10)

        self.bind("<Configure>", self.on_resize)
        self.protocol("WM_DELETE_WINDOW", self.show_popup_graph)

        self.display_random_shape()
        check_Time  = time.time() - Time_out_time
        if check_Time  > Time_limit:
            print("Time exceeded")
            self.show_popup_time()
            self.show_popup_graph()

    def go_back(self):
        self.destroy_window()
        app = IntroWindow(self.n_value)
        app.mainloop()

    def on_resize(self, event):
        self.canvas_width = self.canvas.winfo_width()
        self.canvas_height = self.canvas.winfo_height()
        self.canvas.delete("all")
        if self.previous_shape:
            self.display_shape(self.previous_shape, animate=False)

    def update_size(self, value):
        self.shape_size = int(float(value))
        self.canvas.delete("all")
        if self.previous_shape:
            self.display_shape(self.previous_shape, animate=False)

    def generate_random_shape(self):
        shape_options = ["circle", "square", "triangle", "star", "pentagon", "rectangle", "arc"]
        
        probabilities = [0.1] * len(shape_options)
        if len(self.shape_history) >= self.n_value:
            recent_shapes = self.shape_history[-self.n_value:]
            for i, shape in enumerate(shape_options):
                if shape in recent_shapes:
                    probabilities[i] = 0.4 

        total_prob = sum(probabilities)
        probabilities = [p / total_prob for p in probabilities]

        shape_type = random.choices(shape_options, probabilities)[0]
        
        self.shape_history.append(shape_type)
        return shape_type

    def generate_random_gradient(self, shape_type):
        gradient_steps = 10
        base_color = random.choice([
            "#FF5733", "#33FF57", "#3357FF", "#F5FF33", "#33FFF5",
            "#FF33A8", "#A833FF", "#33FFD5", "#FF8C33", "#8C33FF"
        ])
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
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    def rgb_to_hex(self, rgb_color):
        return "#{:02x}{:02x}{:02x}".format(*rgb_color)

    def animate_shape(self, shape_type, gradient_colors, rotation=False):
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
        self.canvas.create_text(
                200, 25,  
                text=f"Current N Value : {self.n_value} ",
                fill="#1E90FF",
                font=("Arial", 14)
            )
        self.canvas.create_text(
                800, 25, 
                text=f"Response : {self.output_value} ",
                fill = "#1E90FF",
                font=("Arial", 14)
            )
        
        cx, cy = self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2
        x1, y1 = cx - size, cy - size
        x2, y2 = cx + size, cy + size

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

    def calculate_polygon_points(self, cx, cy, size, num_sides, angle):
        points = []
        for i in range(num_sides):
            theta = (2 * math.pi / num_sides) * i + math.radians(angle)
            x = cx + size * math.cos(theta)
            y = cy + size * math.sin(theta)
            points.extend([x, y])
        return points

    def calculate_star_points(self, cx, cy, outer_size, inner_size, num_points, angle):
        points = []
        for i in range(2 * num_points):
            r = outer_size if i % 2 == 0 else inner_size
            theta = (math.pi / num_points) * i + math.radians(angle)
            x = cx + r * math.cos(theta)
            y = cy + r * math.sin(theta)
            points.extend([x, y])
        return points

    def display_shape(self, shape_data, animate=True, rotate=False):
        shape_type, _ = shape_data
        gradient_colors = self.generate_random_gradient(shape_type)
        if animate:
            self.animate_shape(shape_type, gradient_colors, rotation=rotate)
        else:
            color = gradient_colors[-1]
            self.draw_shape(shape_type, self.shape_size, self.rotation_angle if rotate else 0, color)
        self.previous_shape = shape_data

    def display_random_shape(self):
        self.start_time = time.time() 
        shape_type = self.generate_random_shape()
        shape_data = (shape_type, None)
        self.display_shape(shape_data, animate=True, rotate=False)

    def calculate_accuracy_trend(self, accuracies):
        trials = np.arange(1, len(accuracies) + 1)  
        slope, _ = np.polyfit(trials, accuracies, 1) 
        return slope

    def check_response(self, is_correct, button_pressed):
        reaction_time = time.time() - self.start_time
        self.total_count += 1
        if is_correct:
            self.output_value = "Correct"
            self.correct_count += 1
        else:
            self.output_value = "Incorrect"
            self.incorrect_responses += 1

        accuracy = (self.correct_count / self.total_count) * 100
        self.accuracy_treds_reference.append(accuracy)
        accuracy_treds = self.calculate_accuracy_trend(self.accuracy_treds_reference)
        error_ratio = self.incorrect_responses / self.total_count
        print(f"N value: {self.n_value}, Accuracy: {accuracy:.2f}%, Reaction Time: {reaction_time:.4f}s, Error Ratio: {error_ratio:.2f}, accuracy_treds: {accuracy_treds : .2f},correct_responses: {self.correct_count}, incorrect_responses:{self.incorrect_responses}, total_responses : {self.total_count}, Result : {is_correct}, Button_pressed : {button_pressed}")

        self.csv_file = open(self.csv_file_path, "a", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([IntroWindow.user_id, accuracy, reaction_time, error_ratio,  accuracy_treds, self.incorrect_responses,self.n_value, button_pressed])
        self.model_path=os.path.abspath("decision_tree_model.pkl")
        self.model_predictor = ModelPredictor(self.model_path)

        if self.total_count > 24:  
            result = self.model_predictor.predict_from_csv(self.csv_file_path)
            print("Predicted class for the dataset:", result)
            self.n_value = self.n_value + int(result)
            if self.n_value ==  0 :
                self.n_value = 1
            if self.n_value > 5:
                self.n_value = 1
            self.show_popup()
            self.shape_history.clear()
            self.correct_count = 0
            self.total_count = 0
            self.incorrect_responses = 0
            self.copy_csv_contents(self.csv_file_path, 'graph.csv')
            self.clear_csv()

    def show_popup(self):
        self.popup = ctk.CTkToplevel(self)
        self.popup.geometry("300x150")
        self.popup.title("Notification")
        self.output_value = None

        if self.n_value > 5:
            self.n_value = 1
            label = ctk.CTkLabel(self.popup, text=f" Max Value Achived.Resetting N to 1. ")
            label.pack(pady=20)

        else:
            label = ctk.CTkLabel(self.popup, text=f"N value changes to : {self.n_value}")
            label.pack(pady=20)

        button = ctk.CTkButton(self.popup, text="Close", command=self.popup.destroy)
        button.pack(pady=10)

        self.popup.grab_set()  
        self.wait_window(self.popup)

    def copy_csv_contents(self, source_file, target_file):
        source_data = self._read_csv(source_file)
        source_data = source_data[1:] if self.has_header_row(target_file) else source_data
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
        try:
            with open(file_path, "r") as file:
                reader = csv.reader(file)
                first_row = next(reader, None)  
                return first_row is not None and any(cell.isalpha() for cell in first_row)
        except FileNotFoundError:
            return False

    def clear_csv(self):
        with open(self.csv_file_path, "r") as file:
            reader = csv.reader(file)
            header = next(reader, None)  

        with open(self.csv_file_path, "w", newline="") as file:
            writer = csv.writer(file)
            if header:
                writer.writerow(header)

    def check_match(self):
        if len(self.shape_history) <= self.n_value:
            return False
        return self.shape_history[-1] == self.shape_history[-(self.n_value + 1)]

    def display_same_shape(self):
        self.button_pressed = 'Same_shape'
        self.same_button.configure(state="disabled")
        self.rotation_angle = random.randint(0, 360) 
        self.check_response(self.check_match(), self.button_pressed)
        self.display_random_shape()
        check_Time = time.time() - Time_out_time
        if  check_Time > Time_limit:
            print("Time exceeded")
            self.show_popup_time()
            self.show_popup_graph()
        self.canvas.after(400, lambda: self.same_button.configure(state="normal"))

    def display_not_same_shape(self):
        self.button_pressed = 'Diffrent_shape'
        self.not_same_button.configure(state="disabled")
        self.rotation_angle = random.randint(0, 360)  
        self.check_response(not self.check_match(), self.button_pressed)
        self.display_random_shape()
        check_Time  = time.time() - Time_out_time
        if check_Time  > Time_limit:
            print("Time exceeded")
            self.show_popup_time()
            self.show_popup_graph()
        self.canvas.after(400, lambda: self.not_same_button.configure(state="normal"))

    def show_popup_time(self):
        self.popup = ctk.CTkToplevel(self)
        self.popup.geometry("300x150")
        self.popup.title("Time Out")

        label = ctk.CTkLabel(self.popup, text=f"Max Time Limit Exceed")
        label.pack(pady=20)

        button = ctk.CTkButton(self.popup, text="Close", command=self.popup.destroy)
        button.pack(pady=10)

        self.popup.grab_set()  
        self.wait_window(self.popup)

    def show_popup_graph(self):
        if hasattr(self, 'popup_shown') and self.popup_shown:
            return
        self.popup_shown = True
        self.popup = ctk.CTkToplevel(self)
        self.popup.geometry("300x150")
        self.popup.title("Graph Window")

        label = ctk.CTkLabel(self.popup, text="Do you want to see the Analyze your Progress?")
        label.pack(pady=20)

        button_frame = ctk.CTkFrame(self.popup)
        button_frame.pack(pady=10)

        ok_button = ctk.CTkButton(button_frame, text="OK", command=self.happy)
        ok_button.pack(side="left", padx=10)

        cancel_button = ctk.CTkButton(button_frame, text="Cancel", command=self.destroy_window)
        cancel_button.pack(side="right", padx=10)

        self.popup.grab_set()  
        self.wait_window(self.popup)

    def happy(self):
        self.popup_shown = True
        self.popup.destroy()
        #window = GraphWindow("graph.csv")
        self.destroy_window()
        main_app.quit()
        self.check_T = time.time() - Time_out_time 
        report_generator = CSVReportGenerator('graph.csv', self.check_T )
        report_generator.generate_reports()
        #window.mainloop()
        #window.quit()
        self.quit()
        self.destroy()
        exit()

    def destroy_window(self):
        self.popup_shown = True
        self.csv_file.close()
        with open('users.csv', mode="r", newline="") as file:
            reader = csv.DictReader(file)
            rows = list(reader) 
            fieldnames = reader.fieldnames 

        for row in rows:
            if row["User ID"] == IntroWindow.user_id:
                row["N Value"] = self.n_value
        with open('users.csv', mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader() 
            writer.writerows(rows) 
        super().destroy()
        self.quit()


class P(NBackApp):
    def __init__(self, main_app_instance, n_value):
        super().__init__()
        if hasattr(self, "instructions") and self.instructions is not None:
            self.instructions.destroy()
        self.canvas.delete("all") 
        self.n_value = n_value
        self.wm_title(f"Practice window for N value = {n_value}")
        self.geometry("800x600")
        self.minsize(600, 500)
        self.res = None

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.shape_history_practice = []

        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            self.main_frame,
            bg="black",
            highlightthickness=0,
            width=800,
            height=500
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.main_app = main_app_instance
        self.protocol("WM_DELETE_WINDOW", self.on_space_press)

        self.shape_count = 0
        self.feedback_label = None

        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.grid(row=2, column=0, pady=10)

        self.correct_button = ctk.CTkButton(self.button_frame, text="Matching", command=lambda: self.user_response(True))
        self.correct_button.grid(row=2, column=0, pady=5,padx=5)

        self.incorrect_button = ctk.CTkButton(self.button_frame, text="Not Matching", command=lambda: self.user_response(False))
        self.incorrect_button.grid(row=2, column=1, pady=5,padx=5)

        self.skip_button = ctk.CTkButton(self.button_frame, text="skip", command=self.on_close)
        self.skip_button.grid(row=2, column=2, pady=5,padx=5)

        self.display_shapes()
    
    def display_shapes(self):
        self.canvas.delete("all")
        self.shape_count += 1
        shape_type = self.main_app.generate_random_shape()
        self.shape_history_practice.append(shape_type)
        
        cx, cy = self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2
        size = 150
        color = random.choice([
        "#FF5733", "#33FF57", "#3357FF", "#F5FF33", "#33FFF5",
        "#FF33A8", "#A833FF", "#33FFD5", "#FF8C33", "#8C33FF"
        ])
        angle=0
        x1, y1 = cx - size, cy - size
        x2, y2 = cx + size, cy + size

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
            points = main_app.calculate_star_points(cx, cy, size, size / 2, 5, angle)
            self.canvas.create_polygon(points, fill=color, outline="")
        elif shape_type == "pentagon":
            points = main_app.calculate_polygon_points(cx, cy, size, 5, angle)
            self.canvas.create_polygon(points, fill=color, outline="")
        elif shape_type == "rectangle":
            self.canvas.create_rectangle(x1 + size * 0.2, y1, x2 - size * 0.2, y2, fill=color, outline="")

        print(f"Shape {self.shape_count} displayed.")
        
        self.canvas.create_text(
            500, 25, text=f"Displaying Shape: {self.shape_count}",
            fill='#1E90FF', font=("Arial", 14)
        )
        self.canvas.create_text(
            200, 25, text=f"Current N Value: {self.n_value}",
            fill="#1E90FF", font=("Arial", 14)
        )
       
    def user_response(self, is_correct):
        self.res = self.check_shape_match()
        print(f"User selected: {'Correct' if is_correct else 'Incorrect'}. Actual match: {self.res}")
        if self.feedback_label:
            self.canvas.delete(self.feedback_label)
        if is_correct and self.res:
            feedback_text = "Correct!"
            feedback_color = "green"
        elif is_correct and not self.res:
            feedback_text = "Incorrect!"
            feedback_color = "red"
        elif not is_correct and self.res:
            feedback_text = "Incorrect!"
            feedback_color = "red"
        else:
            feedback_text = "Correct!"
            feedback_color = "green"

        self.feedback_label = self.canvas.create_text(
            800, 25, 
            text=f"Your Response: {feedback_text}",
            fill=feedback_color,
            font=("Arial", 14)
        )
        
        self.canvas.update_idletasks() 
        
        if self.shape_count < 20:
            self.after(1000, self.display_shapes)
        elif self.shape_count >= 20 and self.shape_count < 40 and self.n_value < 3:
            self.n_value = 2
            self.after(1000, self.display_shapes)
        else:
            self.on_close()

    def check_shape_match(self):
        if len(self.shape_history_practice) <= self.n_value:
            return False
        return self.shape_history_practice[-1] == self.shape_history_practice[-(self.n_value + 1)]

    def on_space_press(self):
        self.quit()
        self.destroy()
        exit()

    def on_close(self):
        self.destroy()
        start_main_app()


if __name__ == "__main__":
    def start_practice():
        practice_window = P(main_app_instance=main_app, n_value=1)
        practice_window.mainloop()
        practice_window.quit()

    def start_main_app():
        main_app.mainloop()
        main_app.quit()

    main_app = MainApp()
    intro_window = IntroWindow(n_value=1)
    intro_window.mainloop()