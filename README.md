# ML SMS Spam Detection System

A comprehensive machine learning project that classifies SMS messages as spam or legitimate (ham) using multiple Naive Bayes classifiers. This project includes data exploration, text preprocessing, model training, evaluation, and visualization.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Models](#models)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [File Descriptions](#file-descriptions)

## 🎯 Overview

This project implements a spam detection system that uses Natural Language Processing (NLP) and machine learning to classify SMS messages. The system compares three different Naive Bayes variants to determine which performs best for this classification task.

## 📊 Dataset

- **Source**: spam.csv
- **Size**: 5,572 SMS messages
- **Classes**: 
  - Ham (Legitimate): ~86.6%
  - Spam: ~13.4%
- **Format**: CSV with two columns - message label (ham/spam) and message text

## ✨ Features

- **Data Cleaning**: Handles missing values, duplicates, and unnecessary columns
- **Exploratory Data Analysis (EDA)**:
  - Class distribution visualization (pie charts)
  - Statistical analysis of message characteristics
  - Word cloud visualization for spam vs. ham messages
  - Correlation heatmaps and pair plots
  
- **Text Preprocessing**:
  - Lowercase conversion
  - Tokenization
  - Special character removal
  - Stop word removal
  - Punctuation removal
  - Stemming using Porter Stemmer
  
- **Feature Engineering**:
  - TF-IDF vectorization (Term Frequency-Inverse Document Frequency)
  - Feature scaling using MinMaxScaler
  - Message statistics (character count, word count, sentence count)

- **Model Evaluation**:
  - Accuracy scores
  - Confusion matrices
  - Precision, recall, and F1-scores
  - Model comparison and analysis

## 📁 Project Structure

```
ml-sms-spam-detection-system/
├── README.md                          # Project documentation
├── sms_spam_detection.ipynb          # Main Jupyter notebook with full analysis
├── spam.csv                          # Dataset (5,572 SMS messages)
├── test.py                           # Test script for model evaluation
├── LICENSE                           # Project license
└── [Additional project files as needed]
```

## 🚀 Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook (optional, for viewing the .ipynb file)

### Required Libraries

```bash
pip install numpy pandas scikit-learn nltk matplotlib seaborn wordcloud
```

### Download NLTK Data

Run the following in Python to download required NLTK datasets:
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

## 💻 Usage

### Using Jupyter Notebook

1. Open the notebook:
```bash
jupyter notebook sms_spam_detection.ipynb
```

2. Run all cells sequentially to see the complete analysis pipeline

### Running Tests

```bash
python test.py
```

## 🔧 Methodology

The project follows a structured machine learning workflow:

1. **Data Cleaning**
   - Load CSV file with proper encoding
   - Remove unnecessary columns
   - Rename columns for clarity
   - Encode labels (ham=0, spam=1)
   - Remove duplicate messages

2. **Exploratory Data Analysis**
   - Analyze class distribution (identify imbalance)
   - Extract message statistics (character, word, sentence counts)
   - Compare statistics between ham and spam messages
   - Create visualizations (histograms, heatmaps, pair plots)

3. **Text Preprocessing**
   - Convert to lowercase
   - Tokenize words and sentences
   - Remove special characters and punctuation
   - Remove stop words
   - Apply stemming for word normalization

4. **Feature Engineering**
   - Generate TF-IDF features from processed text
   - Scale numerical features using MinMaxScaler
   - Create feature vectors for model training

5. **Model Training & Evaluation**
   - Split data into training (80%) and testing (20%) sets
   - Train three Naive Bayes models
   - Evaluate each model's performance
   - Compare results

## 🤖 Models

### Naive Bayes Classifiers Used:

1. **Bernoulli Naive Bayes**
   - Works with binary/boolean features
   - Suitable for presence/absence of features

2. **Multinomial Naive Bayes**
   - Designed for discrete count data
   - Works well with TF-IDF features
   - Best for text classification tasks

3. **Gaussian Naive Bayes**
   - Assumes features follow Gaussian distribution
   - Good for continuous numerical features

## 📈 Results

The notebook includes:
- Model accuracy comparisons
- Confusion matrices for each model
- Precision, Recall, and F1-score metrics
- Visualizations of model performance
- Feature importance analysis

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **NLTK**: Natural Language Processing toolkit
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization
- **WordCloud**: Generate word cloud visualizations

## 📄 File Descriptions

| File | Description |
|------|-------------|
| `sms_spam_detection.ipynb` | Main Jupyter notebook containing the complete analysis, from data loading to model evaluation |
| `spam.csv` | Dataset containing 5,572 labeled SMS messages |
| `test.py` | Script for testing and validating the models |
| `README.md` | This file - project documentation |
| `LICENSE` | Project license information |

## 🎓 Learning Outcomes

This project demonstrates:
- End-to-end machine learning pipeline implementation
- Text preprocessing and NLP techniques
- Feature engineering for text data
- Model selection and comparison
- Data visualization and exploratory analysis
- Handling imbalanced datasets

## 📝 License

This project is licensed under the terms specified in the LICENSE file.

## 🤝 Contributing

Feel free to fork this project, make improvements, and submit pull requests!

## 📧 Contact

For questions or suggestions, please reach out through GitHub issues or contact the project maintainers.

---

**Note**: The dataset used in this project is the famous SMS Spam Collection dataset, commonly used for text classification tasks in machine learning education and research.