import customtkinter as ctk
import joblib
import numpy as np
import pandas as pd
from tkinter import messagebox
import os
import csv
import json
from datetime import datetime

# -----------------------------
# Load AI model (with error handling)
# -----------------------------
if not os.path.exists("model.pkl") or not os.path.exists("scaler.pkl"):
    raise FileNotFoundError(
        "model.pkl or scaler.pkl not found. Run train_model.py first to generate them."
    )

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Benchmarks for "what's holding you back" analysis
# Uses the 75th percentile of strong applicants in the training data
# -----------------------------
FEATURE_ORDER = ["GRE", "TOEFL", "UniversityRating", "SOP", "LOR", "CGPA", "Research"]
FEATURE_DISPLAY_NAMES = {
    "GRE": "GRE Score",
    "TOEFL": "TOEFL Score",
    "UniversityRating": "University Rating",
    "SOP": "SOP Score",
    "LOR": "LOR Score",
    "CGPA": "CGPA",
    "Research": "Research Experience",
}

try:
    _df = pd.read_csv("dataset.csv")
    BENCHMARKS = {col: _df[col].quantile(0.75) for col in FEATURE_ORDER if col in _df.columns}
except Exception:
    BENCHMARKS = {}

try:
    FEATURE_IMPORTANCES = dict(zip(FEATURE_ORDER, model.feature_importances_))
except Exception:
    FEATURE_IMPORTANCES = {}


def analyze_profile(values, cgpa_10_scale, research_num):
    """Return a list of (field_display_name, gap_ratio, importance) sorted strongest to weakest.
    Only includes fields the user actually controls in the current form (excludes GRE/TOEFL/
    University Rating, which are now fixed defaults for the PhD form)."""
    if not BENCHMARKS or not FEATURE_IMPORTANCES:
        return []

    user_vals = {
        "SOP": values["SOP Score"],
        "LOR": values["LOR Score"],
        "CGPA": cgpa_10_scale,
        "Research": research_num,
    }

    results = []
    for key in user_vals:
        benchmark = BENCHMARKS.get(key)
        importance = FEATURE_IMPORTANCES.get(key, 0)
        if benchmark is None or benchmark == 0:
            continue
        gap_ratio = (benchmark - user_vals[key]) / benchmark  # negative means above benchmark (strength)
        results.append((FEATURE_DISPLAY_NAMES[key], gap_ratio, importance))

    # Sort by gap_ratio ascending: most negative (strongest) first
    results.sort(key=lambda x: x[1])
    return results


def generate_summary(percentage_value, profile_analysis):
    """Build a short personalized natural-language paragraph from the profile analysis."""
    if not profile_analysis:
        return "Enter your details and click Predict to get personalized feedback on your profile."

    strengths = [name for name, gap, imp in profile_analysis if gap < -0.02][:2]
    weaknesses = [name for name, gap, imp in profile_analysis if gap > 0.05]
    weaknesses_sorted = sorted(
        [item for item in profile_analysis if item[1] > 0.05],
        key=lambda x: x[1] * x[2],
        reverse=True
    )
    top_weakness = weaknesses_sorted[0][0] if weaknesses_sorted else None

    if percentage_value >= 85:
        opening = "This is a strong profile overall."
    elif percentage_value >= 70:
        opening = "This is a solid profile with good potential."
    elif percentage_value >= 50:
        opening = "This profile is in a reasonable but competitive range."
    else:
        opening = "This profile has room to grow before it's competitive."

    parts = [opening]

    if strengths:
        if len(strengths) == 1:
            parts.append(f"{strengths[0]} stands out as a clear strength.")
        else:
            parts.append(f"{strengths[0]} and {strengths[1]} stand out as clear strengths.")

    if top_weakness:
        parts.append(
            f"{top_weakness} is comparatively the weakest factor, and is also one of the more "
            f"influential ones for the prediction — improving it would likely move the needle the most."
        )
    elif not weaknesses:
        parts.append("All factors are fairly well balanced relative to one another.")

    return " ".join(parts)

# -----------------------------
# Prediction History Setup
# -----------------------------
HISTORY_DIR = "history"
HISTORY_FILE = os.path.join(HISTORY_DIR, "predictions_log.csv")
os.makedirs(HISTORY_DIR, exist_ok=True)

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "DegreeLevel", "Details", "PredictedChance"])


def log_prediction(degree_level, details, percentage_value):
    with open(HISTORY_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            degree_level,
            json.dumps(details),
            percentage_value
        ])

# -----------------------------
# App Settings
# -----------------------------
SETTINGS_FILE = "settings.json"


def load_settings():
    default = {"theme": "dark"}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                data = json.load(f)
                default.update(data)
        except Exception:
            pass
    return default


def save_settings(settings):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f)
    except Exception:
        pass


app_settings = load_settings()

ctk.set_appearance_mode(app_settings.get("theme", "dark"))
ctk.set_default_color_theme("blue")  # Themes: blue, green, dark-blue

# -----------------------------
# Create Main Window
# -----------------------------
app = ctk.CTk()

app.title("University Admission Prediction System")
app.geometry("1000x700")
app.minsize(900, 600)
app.resizable(True, True)

# -----------------------------
# Selection Screen (shown first)
# -----------------------------
selected_degree = {"value": None}

selection_frame = ctk.CTkFrame(app, fg_color="transparent")
selection_frame.pack(fill="both", expand=True)

selection_title = ctk.CTkLabel(
    selection_frame,
    text="🎓 University Admission Prediction System",
    font=("Arial", 30, "bold")
)
selection_title.pack(pady=(100, 10))

selection_subtitle = ctk.CTkLabel(
    selection_frame,
    text="What degree are you applying for?",
    font=("Arial", 18)
)
selection_subtitle.pack(pady=(0, 40))

degree_buttons_frame = ctk.CTkFrame(selection_frame, fg_color="transparent")
degree_buttons_frame.pack()


def select_degree(degree_name):
    selected_degree["value"] = degree_name
    degree_label.configure(text=f"Applying for: {degree_name}")
    selection_frame.pack_forget()
    content_frame.pack(fill="both", expand=True)
    if degree_name == "BS (Bachelor's)":
        build_bs_form()
    elif degree_name == "Master's":
        build_ms_form()
    elif degree_name == "PhD":
        build_phd_form()
    else:
        build_grad_form()
    reset_result_panel()


degree_options = [
    ("BS (Bachelor's)", "#3498db"),
    ("Master's", "#9b59b6"),
    ("PhD", "#e67e22"),
]

for degree_name, color in degree_options:
    btn = ctk.CTkButton(
        degree_buttons_frame,
        text=degree_name,
        width=200,
        height=60,
        font=("Arial", 16, "bold"),
        fg_color=color,
        hover_color=color,
        command=lambda d=degree_name: select_degree(d)
    )
    btn.pack(side="left", padx=15)


# -----------------------------
# Content Frame (shown after degree level is selected)
# -----------------------------
content_frame = ctk.CTkFrame(app, fg_color="transparent")

# -----------------------------
# Header
# -----------------------------
header_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
header_frame.pack(fill="x", padx=20, pady=(20, 0))

title = ctk.CTkLabel(
    header_frame,
    text="🎓 University Admission Prediction System",
    font=("Arial", 28, "bold")
)
title.pack(side="left", expand=True)


def toggle_theme():
    new_mode = "light" if app_settings.get("theme", "dark") == "dark" else "dark"
    app_settings["theme"] = new_mode
    ctk.set_appearance_mode(new_mode)
    save_settings(app_settings)
    theme_switch.configure(text="☀️ Light Mode" if new_mode == "light" else "🌙 Dark Mode")


theme_switch = ctk.CTkButton(
    header_frame,
    text="☀️ Light Mode" if app_settings.get("theme", "dark") == "light" else "🌙 Dark Mode",
    width=130,
    fg_color="gray30",
    hover_color="gray20",
    command=toggle_theme
)
theme_switch.pack(side="right", padx=10)

subtitle = ctk.CTkLabel(
    content_frame,
    text="AI-Based Admission Prediction",
    font=("Arial", 16)
)
subtitle.pack()

degree_label = ctk.CTkLabel(
    content_frame,
    text="",
    font=("Arial", 13),
    text_color="#5aa9e6"
)
degree_label.pack(pady=(2, 0))


def go_back_to_selection():
    content_frame.pack_forget()
    selection_frame.pack(fill="both", expand=True)


change_degree_btn = ctk.CTkButton(
    content_frame,
    text="Change",
    width=70,
    height=22,
    font=("Arial", 11),
    fg_color="transparent",
    text_color="#5aa9e6",
    hover_color="gray25",
    command=go_back_to_selection
)
change_degree_btn.pack(pady=(2, 0))

# -----------------------------
# Main Frame
# -----------------------------
main_frame = ctk.CTkFrame(content_frame)
main_frame.pack(fill="both", expand=True, padx=20, pady=20)

# -----------------------------
# Left Frame (Inputs)
# -----------------------------
left_frame = ctk.CTkScrollableFrame(main_frame)
left_frame.pack(side="left", fill="both", expand=True, padx=15, pady=15)

# -----------------------------
# Right Frame (Result)
# -----------------------------
right_frame = ctk.CTkFrame(main_frame, width=300)
right_frame.pack(side="right", fill="y", padx=15, pady=15)

# -----------------------------
# Input Fields (built dynamically based on selected degree)
# -----------------------------
entries = {}


def clear_left_frame():
    for widget in left_frame.winfo_children():
        widget.destroy()
    entries.clear()


def add_section_header(parent, text):
    header = ctk.CTkLabel(
        parent,
        text=text,
        font=("Arial", 14, "bold"),
        text_color="#5aa9e6"
    )
    header.pack(anchor="w", padx=20, pady=(18, 2))
    divider = ctk.CTkFrame(parent, height=2, fg_color="gray30")
    divider.pack(fill="x", padx=20, pady=(0, 6))


def add_labeled_entry(parent, field_name, placeholder, internal_key=None):
    key = internal_key if internal_key else field_name
    label = ctk.CTkLabel(parent, text=field_name, font=("Arial", 15))
    label.pack(anchor="w", padx=20, pady=(8, 2))
    entry = ctk.CTkEntry(parent, width=300, placeholder_text=placeholder)
    entry.pack(padx=20)
    entries[key] = entry
    return entry


# ---- PhD fields (AI model based) ----
# GRE, TOEFL, and University Rating are removed from the visible form per request,
# but the underlying model still needs all 7 features, so we feed it dataset medians
# for the fields the user is no longer asked about.
GRAD_FIELDS = ["SOP Score", "LOR Score", "CGPA"]

GRAD_PLACEHOLDERS = {
    "SOP Score": "e.g. 4.5 (1-5)",
    "LOR Score": "e.g. 4.0 (1-5)",
    "CGPA": "e.g. 3.5 (0-4.0)",
}

GRAD_FIELD_LABELS = {
    "SOP Score": "SOP Score",
    "LOR Score": "LOR Score",
    "CGPA": "CGPA",
}

GRAD_FIELD_GROUPS = [
    ("Profile Strength", ["SOP Score", "LOR Score", "CGPA"]),
]

GRAD_VALID_RANGES = {
    "SOP Score": (1, 5),
    "LOR Score": (1, 5),
    "CGPA": (0, 4.0),
}

# Default values for fields removed from the visible form (median of training data)
try:
    GRAD_HIDDEN_DEFAULTS = {
        "GRE Score": float(_df["GRE"].median()),
        "TOEFL Score": float(_df["TOEFL"].median()),
        "University Rating": float(_df["UniversityRating"].median()),
    }
except Exception:
    GRAD_HIDDEN_DEFAULTS = {"GRE Score": 317, "TOEFL Score": 107, "University Rating": 3}

research = None  # CTkComboBox, created in build_grad_form


def build_grad_form():
    global research
    clear_left_frame()

    for section_title, section_fields in GRAD_FIELD_GROUPS:
        add_section_header(left_frame, section_title)
        for field in section_fields:
            add_labeled_entry(left_frame, GRAD_FIELD_LABELS[field], GRAD_PLACEHOLDERS[field], internal_key=field)

    add_section_header(left_frame, "Research Experience")
    research = ctk.CTkComboBox(left_frame, values=["Yes", "No"], width=300)
    research.set("Yes")
    research.pack(padx=20)

    predict_btn = ctk.CTkButton(left_frame, text="Predict Admission", width=300, command=predict_grad)
    predict_btn.pack(pady=25)

    reset_btn = ctk.CTkButton(left_frame, text="Reset", width=300, fg_color="gray", command=reset_grad)
    reset_btn.pack()


# ---- BS fields (merit formula based, no AI model exists for this) ----
BS_FIELDS = ["Matric Percentage", "Intermediate Percentage", "Entry Test Percentage"]

BS_PLACEHOLDERS = {
    "Matric Percentage": "e.g. 85 (0-100)",
    "Intermediate Percentage": "e.g. 80 (0-100)",
    "Entry Test Percentage": "e.g. 75 (0-100)",
}

BS_VALID_RANGES = {
    "Matric Percentage": (0, 100),
    "Intermediate Percentage": (0, 100),
    "Entry Test Percentage": (0, 100),
}

# Standard Pakistani university merit aggregate weights (Matric 10%, Inter 40%, Entry Test 50%)
BS_WEIGHTS = {"Matric Percentage": 0.10, "Intermediate Percentage": 0.40, "Entry Test Percentage": 0.50}


def build_bs_form():
    clear_left_frame()

    add_section_header(left_frame, "Academic Record")
    for field in BS_FIELDS:
        add_labeled_entry(left_frame, field, BS_PLACEHOLDERS[field])

    note = ctk.CTkLabel(
        left_frame,
        text="Merit is calculated using the standard aggregate formula:\nMatric 10% + Intermediate 40% + Entry Test 50%",
        font=("Arial", 12),
        text_color="gray60",
        wraplength=300,
        justify="left"
    )
    note.pack(anchor="w", padx=20, pady=(15, 5))

    predict_btn = ctk.CTkButton(left_frame, text="Calculate Merit", width=300, command=predict_bs)
    predict_btn.pack(pady=25)

    reset_btn = ctk.CTkButton(left_frame, text="Reset", width=300, fg_color="gray", command=reset_bs)
    reset_btn.pack()


# ---- Master's fields (simplified merit formula) ----
MS_FIELDS = ["CGPA", "Entry Test Score"]

MS_PLACEHOLDERS = {
    "CGPA": "e.g. 3.5 (0-4.0)",
    "Entry Test Score": "e.g. 75 (0-100)",
}

MS_VALID_RANGES = {
    "CGPA": (0, 4.0),
    "Entry Test Score": (0, 100),
}

MS_DEGREE_OPTIONS = [
    "BS Computer Science",
    "BS Software Engineering",
    "BS Electrical Engineering",
    "BS Business Administration",
    "BS Mathematics",
    "BS Physics",
    "Other Relevant Degree",
]

ms_relevant_degree = None  # CTkComboBox, created in build_ms_form


def build_ms_form():
    global ms_relevant_degree
    clear_left_frame()

    add_section_header(left_frame, "Academic Record")

    add_labeled_entry(left_frame, "CGPA", MS_PLACEHOLDERS["CGPA"])
    add_labeled_entry(left_frame, "Entry Test Score", MS_PLACEHOLDERS["Entry Test Score"])

    degree_label_field = ctk.CTkLabel(left_frame, text="Relevant Degree", font=("Arial", 15))
    degree_label_field.pack(anchor="w", padx=20, pady=(8, 2))
    ms_relevant_degree = ctk.CTkComboBox(left_frame, values=MS_DEGREE_OPTIONS, width=300)
    ms_relevant_degree.set(MS_DEGREE_OPTIONS[0])
    ms_relevant_degree.pack(padx=20)

    note = ctk.CTkLabel(
        left_frame,
        text="Merit calculated as CGPA 50% + Entry Test 50%.",
        font=("Arial", 12),
        text_color="gray60",
        wraplength=300,
        justify="left"
    )
    note.pack(anchor="w", padx=20, pady=(15, 5))

    predict_btn = ctk.CTkButton(left_frame, text="Calculate Merit", width=300, command=predict_ms)
    predict_btn.pack(pady=25)

    reset_btn = ctk.CTkButton(left_frame, text="Reset", width=300, fg_color="gray", command=reset_ms)
    reset_btn.pack()


# ---- PhD fields (Pakistan-based merit formula) ----
# Based on HEC's typical PhD eligibility: MS/MPhil CGPA, GAT-Subject/HAT-Subject test, and research output
PHD_FIELDS = ["MS/MPhil CGPA", "GAT-Subject / HAT-Subject Score", "Research Publications"]

PHD_PLACEHOLDERS = {
    "MS/MPhil CGPA": "e.g. 3.4 (0-4.0)",
    "GAT-Subject / HAT-Subject Score": "e.g. 65 (0-100)",
    "Research Publications": "e.g. 1 (number of papers)",
}

PHD_VALID_RANGES = {
    "MS/MPhil CGPA": (0, 4.0),
    "GAT-Subject / HAT-Subject Score": (0, 100),
    "Research Publications": (0, 20),
}

PHD_DEGREE_OPTIONS = [
    "MS Computer Science",
    "MS Software Engineering",
    "MS Electrical Engineering",
    "MS Business Administration",
    "MS Mathematics",
    "MS Physics",
    "Other Relevant Degree",
]

phd_relevant_degree = None  # CTkComboBox, created in build_phd_form


def build_phd_form():
    global phd_relevant_degree
    clear_left_frame()

    add_section_header(left_frame, "Academic Record")

    for field in PHD_FIELDS:
        add_labeled_entry(left_frame, field, PHD_PLACEHOLDERS[field])

    degree_label_field = ctk.CTkLabel(left_frame, text="Relevant Degree", font=("Arial", 15))
    degree_label_field.pack(anchor="w", padx=20, pady=(8, 2))
    phd_relevant_degree = ctk.CTkComboBox(left_frame, values=PHD_DEGREE_OPTIONS, width=300)
    phd_relevant_degree.set(PHD_DEGREE_OPTIONS[0])
    phd_relevant_degree.pack(padx=20)

    note = ctk.CTkLabel(
        left_frame,
        text=(
            "Based on HEC PhD eligibility guidelines: minimum CGPA 3.0/4.0 (or 70%) in MS/MPhil, "
            "and a passing GAT-Subject/HAT-Subject score (typically 60%+).\n\n"
            "Merit calculated as CGPA 40% + Test Score 40% + Research Publications 20% "
            "(capped at 5 papers). Confirm exact thresholds with your target university and "
            "the official HEC/ETC portal, as requirements vary."
        ),
        font=("Arial", 12),
        text_color="gray60",
        wraplength=300,
        justify="left"
    )
    note.pack(anchor="w", padx=20, pady=(15, 5))

    predict_btn = ctk.CTkButton(left_frame, text="Calculate Merit", width=300, command=predict_phd)
    predict_btn.pack(pady=25)

    reset_btn = ctk.CTkButton(left_frame, text="Reset", width=300, fg_color="gray", command=reset_phd)
    reset_btn.pack()


def predict_grad():
    try:
        values = {}
        for field in GRAD_FIELDS:
            raw = entries[field].get().strip()
            if raw == "":
                messagebox.showerror("Missing Input", f"Please enter a value for {field}.")
                return
            try:
                val = float(raw)
            except ValueError:
                messagebox.showerror("Invalid Input", f"{field} must be a number.")
                return

            low, high = GRAD_VALID_RANGES[field]
            if not (low <= val <= high):
                messagebox.showerror(
                    "Out of Range",
                    f"{field} must be between {low} and {high}."
                )
                return
            values[field] = val

        research_value = research.get()
        research_num = 1 if research_value == "Yes" else 0

        # Convert CGPA from 4.0 scale (user input) to 10.0 scale (model was trained on this)
        cgpa_10_scale = values["CGPA"] * 2.5

        sample = np.array([[
            GRAD_HIDDEN_DEFAULTS["GRE Score"],
            GRAD_HIDDEN_DEFAULTS["TOEFL Score"],
            GRAD_HIDDEN_DEFAULTS["University Rating"],
            values["SOP Score"],
            values["LOR Score"],
            cgpa_10_scale,
            research_num
        ]])

        sample = scaler.transform(sample)

        prediction = model.predict(sample)[0]

        # Clamp prediction to a valid 0-1 probability range
        prediction = max(0.0, min(1.0, prediction))

        percentage_value = round(prediction * 100, 2)

        percentage.configure(text=f"{percentage_value}%")

        progress.set(prediction)

        log_prediction("Master's/PhD (AI Model)", {**values, "Research": research_num}, percentage_value)

        if percentage_value >= 85:
            tier_color = "#2ecc71"
            status.configure(text="Excellent Chance of Admission", text_color=tier_color)
            progress.configure(progress_color=tier_color)
        elif percentage_value >= 70:
            tier_color = "#f1c40f"
            status.configure(text="Good Chance of Admission", text_color=tier_color)
            progress.configure(progress_color=tier_color)
        elif percentage_value >= 50:
            tier_color = "#e67e22"
            status.configure(text="Average Chance of Admission", text_color=tier_color)
            progress.configure(progress_color=tier_color)
        else:
            tier_color = "#e74c3c"
            status.configure(text="Low Chance of Admission", text_color=tier_color)
            progress.configure(progress_color=tier_color)

        percentage.configure(text_color=tier_color)

        profile_analysis = analyze_profile(values, cgpa_10_scale, research_num)
        summary_text = generate_summary(percentage_value, profile_analysis)
        insight_label.configure(text=summary_text, text_color="gray85")

    except Exception as e:
        print("ERROR:", e)
        messagebox.showerror("Error", str(e))


def reset_grad():
    for field in GRAD_FIELDS:
        entries[field].delete(0, "end")
    research.set("Yes")
    reset_result_panel()


def predict_bs():
    try:
        values = {}
        for field in BS_FIELDS:
            raw = entries[field].get().strip()
            if raw == "":
                messagebox.showerror("Missing Input", f"Please enter a value for {field}.")
                return
            try:
                val = float(raw)
            except ValueError:
                messagebox.showerror("Invalid Input", f"{field} must be a number.")
                return

            low, high = BS_VALID_RANGES[field]
            if not (low <= val <= high):
                messagebox.showerror(
                    "Out of Range",
                    f"{field} must be between {low} and {high}."
                )
                return
            values[field] = val

        merit = sum(values[field] * BS_WEIGHTS[field] for field in BS_FIELDS)
        merit = max(0.0, min(100.0, merit))
        percentage_value = round(merit, 2)

        percentage.configure(text=f"{percentage_value}%")
        progress.set(merit / 100)

        log_prediction("BS (Bachelor's)", values, percentage_value)

        if percentage_value >= 85:
            tier_color = "#2ecc71"
            status.configure(text="Excellent Merit Position", text_color=tier_color)
        elif percentage_value >= 70:
            tier_color = "#f1c40f"
            status.configure(text="Good Merit Position", text_color=tier_color)
        elif percentage_value >= 50:
            tier_color = "#e67e22"
            status.configure(text="Average Merit Position", text_color=tier_color)
        else:
            tier_color = "#e74c3c"
            status.configure(text="Low Merit Position", text_color=tier_color)

        progress.configure(progress_color=tier_color)
        percentage.configure(text_color=tier_color)

        weakest = min(BS_FIELDS, key=lambda f: values[f])
        insight_label.configure(
            text=(
                f"Merit calculated as Matric 10% + Intermediate 40% + Entry Test 50%. "
                f"Your {weakest} is relatively the weakest input — since Entry Test and "
                f"Intermediate carry the most weight, improving those affects your merit the most."
            ),
            text_color="gray85"
        )

    except Exception as e:
        print("ERROR:", e)
        messagebox.showerror("Error", str(e))


def reset_bs():
    for field in BS_FIELDS:
        entries[field].delete(0, "end")
    reset_result_panel()


def predict_ms():
    try:
        values = {}
        for field in MS_FIELDS:
            raw = entries[field].get().strip()
            if raw == "":
                messagebox.showerror("Missing Input", f"Please enter a value for {field}.")
                return
            try:
                val = float(raw)
            except ValueError:
                messagebox.showerror("Invalid Input", f"{field} must be a number.")
                return

            low, high = MS_VALID_RANGES[field]
            if not (low <= val <= high):
                messagebox.showerror(
                    "Out of Range",
                    f"{field} must be between {low} and {high}."
                )
                return
            values[field] = val

        cgpa_as_percent = (values["CGPA"] / 4.0) * 100
        merit = (cgpa_as_percent * 0.5) + (values["Entry Test Score"] * 0.5)
        merit = max(0.0, min(100.0, merit))
        percentage_value = round(merit, 2)

        relevant_degree = ms_relevant_degree.get()

        percentage.configure(text=f"{percentage_value}%")
        progress.set(merit / 100)

        log_prediction(
            "Master's",
            {**values, "RelevantDegree": relevant_degree},
            percentage_value
        )

        if percentage_value >= 85:
            tier_color = "#2ecc71"
            status.configure(text="Excellent Merit Position", text_color=tier_color)
        elif percentage_value >= 70:
            tier_color = "#f1c40f"
            status.configure(text="Good Merit Position", text_color=tier_color)
        elif percentage_value >= 50:
            tier_color = "#e67e22"
            status.configure(text="Average Merit Position", text_color=tier_color)
        else:
            tier_color = "#e74c3c"
            status.configure(text="Low Merit Position", text_color=tier_color)

        progress.configure(progress_color=tier_color)
        percentage.configure(text_color=tier_color)

        insight_label.configure(
            text=(
                f"Merit calculated as CGPA 50% + Entry Test 50%, for {relevant_degree}. "
                f"Both factors are weighted equally, so improving whichever is currently "
                f"lower will raise your merit the most."
            ),
            text_color="gray85"
        )

    except Exception as e:
        print("ERROR:", e)
        messagebox.showerror("Error", str(e))


def reset_ms():
    for field in MS_FIELDS:
        entries[field].delete(0, "end")
    ms_relevant_degree.set(MS_DEGREE_OPTIONS[0])
    reset_result_panel()


def predict_phd():
    try:
        values = {}
        for field in PHD_FIELDS:
            raw = entries[field].get().strip()
            if raw == "":
                messagebox.showerror("Missing Input", f"Please enter a value for {field}.")
                return
            try:
                val = float(raw)
            except ValueError:
                messagebox.showerror("Invalid Input", f"{field} must be a number.")
                return

            low, high = PHD_VALID_RANGES[field]
            if not (low <= val <= high):
                messagebox.showerror(
                    "Out of Range",
                    f"{field} must be between {low} and {high}."
                )
                return
            values[field] = val

        cgpa_as_percent = (values["MS/MPhil CGPA"] / 4.0) * 100
        test_score = values["GAT-Subject / HAT-Subject Score"]
        publications_capped = min(values["Research Publications"], 5)
        publication_score = (publications_capped / 5) * 100

        merit = (cgpa_as_percent * 0.40) + (test_score * 0.40) + (publication_score * 0.20)
        merit = max(0.0, min(100.0, merit))
        percentage_value = round(merit, 2)

        relevant_degree = phd_relevant_degree.get()

        percentage.configure(text=f"{percentage_value}%")
        progress.set(merit / 100)

        log_prediction(
            "PhD",
            {**values, "RelevantDegree": relevant_degree},
            percentage_value
        )

        if percentage_value >= 85:
            tier_color = "#2ecc71"
            status.configure(text="Excellent Merit Position", text_color=tier_color)
        elif percentage_value >= 70:
            tier_color = "#f1c40f"
            status.configure(text="Good Merit Position", text_color=tier_color)
        elif percentage_value >= 50:
            tier_color = "#e67e22"
            status.configure(text="Average Merit Position", text_color=tier_color)
        else:
            tier_color = "#e74c3c"
            status.configure(text="Low Merit Position", text_color=tier_color)

        progress.configure(progress_color=tier_color)
        percentage.configure(text_color=tier_color)

        weakest_label = "CGPA" if cgpa_as_percent <= test_score and cgpa_as_percent <= publication_score else (
            "Test Score" if test_score <= publication_score else "Research Publications"
        )

        insight_label.configure(
            text=(
                f"Merit calculated as CGPA 40% + GAT-Subject/HAT-Subject 40% + Research Publications 20% "
                f"(capped at 5 papers), for {relevant_degree}. {weakest_label} is comparatively your "
                f"weakest factor — note that most HEC-recognized universities also require a minimum "
                f"CGPA of 3.0/4.0 (or 70%) and a passing test score (typically 60%+) regardless of overall merit."
            ),
            text_color="gray85"
        )

    except Exception as e:
        print("ERROR:", e)
        messagebox.showerror("Error", str(e))


def reset_phd():
    for field in PHD_FIELDS:
        entries[field].delete(0, "end")
    phd_relevant_degree.set(PHD_DEGREE_OPTIONS[0])
    reset_result_panel()


def reset_result_panel():
    percentage.configure(text="--", text_color="gray70")
    status.configure(
        text="Fill in your details and click Predict\nto see your chances",
        text_color="gray70"
    )
    progress.configure(progress_color="#5aa9e6")
    progress.set(0)
    insight_label.configure(
        text="Enter your details and click Predict to get personalized feedback on your profile.",
        text_color="gray70"
    )


# -----------------------------
# Result Section
# -----------------------------
result_title = ctk.CTkLabel(
    right_frame,
    text="Prediction Result",
    font=("Arial", 22, "bold")
)
result_title.pack(pady=20)

percentage = ctk.CTkLabel(
    right_frame,
    text="--",
    font=("Arial", 42, "bold"),
    text_color="gray70"
)
percentage.pack(pady=15)

status = ctk.CTkLabel(
    right_frame,
    text="Fill in your details and click Predict\nto see your chances",
    font=("Arial", 16),
    justify="center"
)
status.pack(padx=10)

progress = ctk.CTkProgressBar(
    right_frame,
    width=220,
    progress_color="#5aa9e6"
)

progress.set(0)

progress.pack(pady=25)

insight_label = ctk.CTkLabel(
    right_frame,
    text="Enter your details and click Predict to get personalized feedback on your profile.",
    font=("Arial", 13),
    text_color="gray70",
    wraplength=240,
    justify="left"
)
insight_label.pack(pady=(0, 15), padx=15, fill="x")

# -----------------------------
# Build the correct form now that result widgets exist
# -----------------------------
if selected_degree["value"] == "BS (Bachelor's)":
    build_bs_form()
else:
    build_grad_form()

# -----------------------------
# Run App
# -----------------------------
app.mainloop()