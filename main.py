import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from ttkbootstrap import Style
from PIL import Image, ImageTk
from logistic_regression import LogisticRegression


class GUI(tk.Tk):

    def __init__(self):
        super().__init__(sync=True)
        self._gui_configs()

    def _gui_configs(self):
        self.title("Heart Disease Predictor")
        self.icon = ImageTk.PhotoImage(Image.open(f"./heart.ico"))
        self.tk.call("wm", "iconphoto", self._w, self.icon)
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f"1000x900+{screen_width//4}+100")
        self.style = Style(theme="darkly")
        self.font = ["Rockwell", 30]
        self.model = LogisticRegression(
            learning_rate=1, num_iters=360, verbose=True)
        self.font[1] = 16
        self.style.configure("TButton", font=self.font)
        self.style.configure("TButton", justify="center")

    def _header(self):
        self.font[1] = 30
        self._header_frame = ttk.Frame(self, style="TFrame")
        self._app_name = ttk.Label(self._header_frame, text="Heart Disease Predictor",
                                   font=self.font+["bold"], style="success.TLabel")
        self._app_name.pack(side="top", anchor="center")
        self._header_frame.pack(side="top", fill="x")

    def _body(self):
        self.font[1] = 20
        self._body_frame = ttk.Frame(self, style="TFrame")

        self._fit_button = ttk.Button(self._body_frame, text="Train model",
                                      style="warning.Outline.TButton", padding=(30, 10))
        self._fit_button.pack(side="top", anchor="center", pady=20)
        self._fit_button.bind(
            "<Button-1>",
            lambda e: (
                self.fit_data(),
                e.widget.configure(style="success.TButton",
                                   text="Model Trained ✔"),
                self.predict_btn.configure(state="normal"),
            ))

        self.sep = ttk.Separator(
            self._body_frame, orient="horizontal", style="secondary.TSeparator")
        self.sep.pack(side="top", fill="x", pady=(0, 20))

        self.font[1] = 16
        self._chol_frame = ttk.Frame(
            self._body_frame, style="TFrame", padding=(120, 0, 0, 0))
        self.chol = tk.StringVar()
        self.chol_lbl = ttk.Label(
            self._chol_frame, text="Cholesterol", font=self.font, style="info.TLabel")
        self.chol_lbl.grid(row=0, column=0, padx=(0, 30))
        self.font[1] = 14
        self._chol_entry = ttk.Entry(
            self._chol_frame, style="info.TEntry", textvariable=self.chol, font=self.font, width=10)
        self._chol_entry.grid(row=0, column=1, padx=180)
        self._chol_frame.pack(side="top", anchor="nw")

        self.font[1] = 16
        self._thalach_frame = ttk.Frame(
            self._body_frame, style="TFrame", padding=(120, 0, 0, 0))
        self.thalach = tk.StringVar()
        self.thalach_lbl = ttk.Label(
            self._thalach_frame, text="Maximum\nheart rate", font=self.font, style="info.TLabel")
        self.thalach_lbl.grid(row=0, column=0, padx=(0, 30))
        self.font[1] = 14
        self._thalach_entry = ttk.Entry(
            self._thalach_frame, style="info.TEntry", textvariable=self.thalach, font=self.font, width=10)
        self._thalach_entry.grid(row=0, column=1, padx=198)
        self._thalach_frame.pack(side="top", anchor="nw", pady=30)

        self.font[1] = 16
        self._cp_frame = ttk.Frame(
            self._body_frame, style="TFrame", padding=(120, 0, 0, 0))
        self.cp = tk.IntVar(value=-1)
        self.cp_lbl = ttk.Label(
            self._cp_frame, text="Chest Pain", style="info.TLabel", font=self.font)
        self.cp_lbl.grid(row=0, column=0, padx=(0, 60))
        self.font[1] = 14
        self.style.configure("Outline.Toolbutton",
                             font=self.font, justify="center")
        self.cp_op1 = ttk.Radiobutton(self._cp_frame, text="Typical\nangina", padding=(20, 15),
                                      style="info.Outline.ToolButton", value=0, variable=self.cp)
        self.cp_op1.grid(row=0, column=1)
        self.cp_op2 = ttk.Radiobutton(self._cp_frame, text="A-typical\nangina", padding=(18, 15),
                                      style="info.Outline.ToolButton", value=1, variable=self.cp)
        self.cp_op2.grid(row=0, column=2, padx=50)
        self.cp_op3 = ttk.Radiobutton(self._cp_frame, text="non-anginal", padding=(10, 25),
                                      style="info.Outline.ToolButton", value=2, variable=self.cp)
        self.cp_op3.grid(row=0, column=3)
        self.cp_op4 = ttk.Radiobutton(self._cp_frame, text="Asymptomatic", padding=(10, 25),
                                      style="info.Outline.ToolButton", value=3, variable=self.cp)
        self.cp_op4.grid(row=0, column=4, padx=50)
        self._cp_frame.pack(side="top", fill="x", anchor="nw")

        self.font[1] = 16
        self._exang_frame = ttk.Frame(
            self._body_frame, style="TFrame", padding=(120, 0, 0, 0))
        self.exang = tk.IntVar(value=-1)
        self.exang_lbl = ttk.Label(self._exang_frame, text="Chest Pain during exercise",
                                   style="info.TLabel", font=self.font)
        self.exang_lbl.grid(row=0, column=0, padx=(0, 15))
        self.exang_op1 = ttk.Radiobutton(self._exang_frame, text="Yes", width=9,
                                         style="info.Outline.ToolButton", value=1, variable=self.exang)
        self.exang_op1.grid(row=0, column=1, padx=50)
        self.exang_op2 = ttk.Radiobutton(self._exang_frame, text="No", width=9,
                                         style="info.Outline.ToolButton", value=0, variable=self.exang)
        self.exang_op2.grid(row=0, column=2)
        self._exang_frame.pack(side="top", fill="x", anchor="nw", pady=30)

        self._age_frame = ttk.Frame(
            self._body_frame, style="TFrame", padding=(120, 0, 0, 0))
        self.age = tk.StringVar()
        self.age_lbl = ttk.Label(
            self._age_frame, text="Age", style="info.TLabel", font=self.font)
        self.age_lbl.grid(row=0, column=0)
        self.font[1] = 14
        self.age_entry = ttk.Entry(
            self._age_frame, style="info.TEntry", font=self.font, textvariable=self.age, width=10)
        self.age_entry.grid(row=0, column=1, padx=285)
        self._age_frame.pack(side="top", fill="x", anchor="nw")

        self.font[1] = 16
        self._sex_frame = ttk.Frame(
            self._body_frame, style="TFrame", padding=(120, 0, 0, 0))
        self.sex = tk.IntVar(value=-1)
        self.sex_lbl = ttk.Label(
            self._sex_frame, text="Sex", style="info.TLabel", font=self.font)
        self.sex_lbl.grid(row=0, column=0, padx=(0, 240))

        self.sex_op1 = ttk.Radiobutton(self._sex_frame, text="Male", width=9,
                                       style="info.Outline.ToolButton", value=1, variable=self.sex)
        self.sex_op1.grid(row=0, column=1, padx=50)
        self.sex_op2 = ttk.Radiobutton(self._sex_frame, text="Female", width=9,
                                       style="info.Outline.ToolButton", value=0, variable=self.sex)
        self.sex_op2.grid(row=0, column=2)
        self._sex_frame.pack(side="top", fill="x", anchor="nw", pady=30)

        self.font[1] = 16
        self._trestbps_frame = ttk.Frame(
            self._body_frame, style="TFrame", padding=(120, 0, 0, 0))
        self.trestbps = tk.StringVar()
        self.trestbps_lbl = ttk.Label(
            self._trestbps_frame, text="Blood pressure", font=self.font, style="info.TLabel")
        self.trestbps_lbl.grid(row=0, column=0)
        self.font[1] = 14
        self._trestbps_entry = ttk.Entry(
            self._trestbps_frame, style="info.TEntry", textvariable=self.trestbps, font=self.font, width=10)
        self._trestbps_entry.grid(row=0, column=1, padx=177)
        self._trestbps_frame.pack(side="top", anchor="nw")

        self.font[1] = 16
        self._fbs_frame = ttk.Frame(
            self._body_frame, style="TFrame", padding=(120, 0, 0, 0))
        self.fbs = tk.IntVar(value=-1)
        self.fbs_lbl = ttk.Label(
            self._fbs_frame, text="Blood sugar\n(above 120mg/dl)", font=self.font, style="info.TLabel")
        self.fbs_lbl.grid(row=0, column=0, padx=(0, 102))
        self.fbs_op1 = ttk.Radiobutton(self._fbs_frame, text="Yes", width=9,
                                       style="info.Outline.ToolButton", value=1, variable=self.fbs)
        self.fbs_op1.grid(row=0, column=1, padx=50)
        self.fbs_op2 = ttk.Radiobutton(self._fbs_frame, text="No", width=9,
                                       style="info.Outline.ToolButton", value=0, variable=self.fbs)
        self.fbs_op2.grid(row=0, column=2)
        self._fbs_frame.pack(side="top", anchor="nw", pady=30)

        self.sep2 = ttk.Separator(
            self._body_frame, orient="horizontal", style="secondary.TSeparator")
        self.sep2.pack(side="top", fill="x", pady=(0, 10))

        self.predict_btn = ttk.Button(self._body_frame, text="Predict",
                                      style="success.Outline.TButton", padding=(50, 10),
                                      command=self.show, state="disabled")
        self.predict_btn.pack(side="top", anchor="center")

        self._body_frame.pack(side="top", fill="both", expand=True)

    def fit_data(self):
        data = pd.read_csv("./data/train.csv")
        X = data[["chol", "thalach", "cp", "exang",
                  "age", "sex", "trestbps", "fbs"]]
        Y = data.loc[:, "target"]
        X, self.mu, self.sigma = standardization(X)
        self.model.fit(X, Y)
        return 1

    def predict(self):
        chol = int(self.chol.get())
        thalach = int(self.thalach.get())
        cp = self.cp.get()
        exang = self.exang.get()
        age = int(self.age.get())
        sex = self.sex.get()
        trestbps = int(self.trestbps.get())
        fbs = self.fbs.get()

        data = pd.DataFrame(
            {"chol": [chol],
             "thalach": [thalach],
             "cp": [cp],
             "exang": [exang],
             "age": [age],
             "sex": [sex],
             "trestbps": [trestbps],
             "fbs": [fbs]
             })
        data = (data - self.mu) / self.sigma
        return self.model.predict(data)

    def show(self):
        self.preds = self.predict()[0]
        text = "Yes" if self.preds == 1 else "No"

        if hasattr(self, "_pred_lbl"):
            self._pred_lbl.configure(text=text, 
            style=("success" if self.preds == 0 else "danger") + ".Inverse.TLabel")
    
        self._pred_lbl = ttk.Label(self._body_frame, text=text, style=(
            "success" if self.preds == 0 else "danger") + ".inverse.TLabel", font=self.font, padding=(20, 10))
        self._pred_lbl.pack(side="left", anchor="center", padx=(300, 0))

        if hasattr(self, "acc_lbl"):    return
        self.acc_lbl = ttk.Label(self._body_frame,
                                 text=f"Model Accuracy: {self.model.accuracy*100}%",
                                 style="warning.TLabel", font=self.font, padding=(10, 10))
        self.acc_lbl.pack(side="right", anchor="center", padx=(0, 300))

    def run(self):
        self._header()
        self._body()
        self.mainloop()


def standardization(X: pd.DataFrame) -> tuple[pd.DataFrame | float]:
    """
    Normalized data using Z-score normalization
    Args:
        X: data
    Returns:
        norm_data: normalized data
        mu (μ): mean of each feature
        sigma (σ): standard deviation of each feature
    """
    # calculating mean of each feature
    mu = np.mean(X, axis=0)
    # calculating standard deviation of each feature
    sigma = np.std(X, axis=0)
    # normalizing data
    norm_data = (X - mu) / sigma
    return norm_data, mu, sigma


if __name__ == "__main__":
    mygui = GUI()
    mygui.run()
