
# 📦 ConfigurableDLReviewClassifierPipeline

An end-to-end, **modular and configurable deep learning pipeline** for sentiment classification of movie reviews using RNN-based models — BiLSTM, LSTM, and GRU.

This project automates the full ML lifecycle: from data preprocessing, splitting, training, evaluation, saving outputs, and serving predictions manually via a Streamlit interface.

> ⚠️ **Note:** Streamlit integration is manual. You must manually select a model and update paths in `app.py`.

---

## 🎯 Key Features

* 🔧 **Fully configurable** via `config.yaml`
* 📁 **Modular folder structure** for clarity and reusability
* 🧠 Supports 3 RNN-based models: `Bi-LSTM`, `LSTM`, `GRU`
* ♻️ Easily extendable to other DL/NLP models
* 📊 Automatic saving of metrics, plots, models, tokenizer, thresholds
* 📝 **Logging** for all components to trace pipeline steps
* 🌐 **Streamlit** app for serving predictions (manual setup)

---

## 🛠️ Tech Stack

* **Language**: Python 3.10
* **Libraries**:

  * `TensorFlow 2.19.0`
  * `Scikit-learn 1.6.1`
  * `Matplotlib 3.10`
  * `Streamlit 1.45.1`
  * `PyYAML 6.0.2`
  * `Joblib 1.5.1`

---

## 📂 Project Structure

```
ConfigurableDLReviewClassifierPipeline/
│
├── dataset/                      # Raw and cleaned dataset
│   └── movie.csv, cleaned_dataset.csv
│
├── logs/                         # Logging
│   └── pipeline_logs.log
│
├── notebook/                     # Jupyter experiments and artifacts
│   └── experiments.ipynb, prediction.ipynb, tokenizer.pkl, maxlength.pkl, best_model.keras
│
├── outputs/
│   ├── metrics/                  # JSON evaluation metrics for all models
│   ├── plots/                    # Accuracy/loss + ROC plots
│   ├── preprocessing_artifacts/ # Tokenizer + max length
│   └── trained_models/          # Saved .keras models + thresholds
│
├── src/
│   ├── config/                  # Config loader
│   ├── data/                    # Data loading
│   ├── model_training/         # Training, evaluation, thresholding
│   │   └── model_definitions/  # GRU, LSTM, BiLSTM
│   ├── pipeline/               # Main full pipeline logic
│   ├── pre_processing/         # Data cleaning, splitting, tokenizing
│   └── utils/                  # Logging, helper functions
│
├── .gitignore
├── app.py                      # Streamlit app
├── config.yaml                 # Config file for the entire pipeline
├── main.py                     # Entry point for the pipeline
└── requirements.txt
```

---

## ⚙️ How It Works

The pipeline uses a single config file to control key behavior and parameters. Here's a high-level flow:

1. **Loads the dataset**
2. **Preprocesses the data** (cleaning, splitting, tokenizing)
3. **Trains the selected model** (BiLSTM/LSTM/GRU)
4. **Evaluates performance** using default threshold
5. **Finds optimal threshold** using ROC analysis
6. **Generates and saves** metrics, plots, models, tokenizer, and thresholds

---

## 🔁 Example Config (`config.yaml`)

```yaml
dataset:
  path: "dataset/movie.csv"
  cleaned_dataset_save_path: "dataset/cleaned_dataset.csv"
  text_column: "text"
  label_column: "label"

dataset_splitting:
  test_size: 0.2
  random_state: 42
  split_save_path: "dataset"

tokenization:
  num_of_words: 10000
  max_length: 200
  oov_token: "<OOV>"
  padding: "post"
  truncating: "post"
  save_tokenizer_path: "outputs/preprocessing_artifacts/tokenizer.joblib"
  save_max_length: "outputs/preprocessing_artifacts/max_length.json"
```

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/ConfigurableDLReviewClassifierPipeline.git
cd ConfigurableDLReviewClassifierPipeline
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Update Config

Update `config.yaml` paths and parameters to match your environment.
Then point `config_loader.py` to your `config.yaml` path.

### 4. Run the Pipeline

```bash
python main.py
```

### 5. View Outputs

* 📁 `outputs/` → Metrics, plots, models
* 📁 `logs/pipeline_logs.log` → Full logs for debugging

---

## 🌐 Streamlit App (Manual Setup)

1. Choose model & tokenizer from `outputs/trained_models` and `outputs/preprocessing_artifacts`
2. Update the corresponding file paths manually in `app.py`
3. Run:

```bash
streamlit run app.py
```

Then open browser at: `http://localhost:8501`

---

## 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## 📌 Notes

* Designed for demonstrating **clean code**, **config-driven architecture**, and **deep learning text classification**
* Streamlit integration is **manual** by design, for flexibility

---

## ✅ What I Learned

* Structuring modular deep learning pipelines for NLP
* Designing reusable, clean architecture for real-world projects
* Logging, output traceability, and reproducibility
* Trade-offs between manual vs. automated deployment

---

Let me know if you'd like a matching `LICENSE`, project badge suggestions, or `README` badges added!
