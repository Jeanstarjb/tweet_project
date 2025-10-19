# Twitter Sentiment Analysis for Tech Brands

## 1. Project Overview

This project presents a comprehensive, end-to-end machine learning pipeline for classifying the sentiment of tweets directed at major tech brands (**Apple** and **Google**). The primary business objective is to develop an automated system capable of identifying customer sentiment—with a critical focus on accurately detecting **negative feedback**—to enable proactive customer engagement and strategic brand management.

The project follows a strategic, iterative approach, beginning with classic machine learning models and culminating in the fine-tuning of a state-of-the-art, pre-trained Transformer model. The final result is a high-performing classifier deployed as an interactive web application using **Streamlit**.

---

## 2. Business Understanding

In the digital age, social media platforms like Twitter are a primary channel for public expression. The sheer volume of daily tweets about major tech brands is impossible to analyze manually. Without an automated solution, companies risk missing critical feedback, failing to spot emerging PR crises, and making strategic decisions based on incomplete evidence. This project aims to solve that problem by translating unstructured tweet data into structured, actionable sentiment data.

**Metrics of Success:**

- **Primary Metric:** Model's ability to accurately identify negative sentiment, measured by **Recall** and **F1-Score** for the *Negative* class.  
- **Secondary Metric:** Overall balanced performance across all classes, measured by the **Macro Average F1-Score**.

---

## 3. Data

The dataset consists of over **9,000 tweets**, each containing the tweet text and associated labels indicating the sentiment (*Positive*, *Negative*, or *Neutral*) and the brand (if any) the emotion is directed towards.

### Exploratory Data Analysis (EDA)

After an intensive cleaning phase, the EDA revealed several critical insights that guided the project:

- **Severe Class Imbalance:** The dataset was heavily skewed towards *Neutral* sentiment, with *Negative emotion* being the smallest and most critical minority class.  
- **Distinct Brand Personas:** *Apple* generated significantly more emotional engagement (both positive and negative) compared to *Google*, whose discussions were more informational and neutral.  
- **Statistical Significance:** Hypothesis testing (Chi-Square, T-test) confirmed that the observed differences in sentiment distribution between brands and tweet length between sentiments were statistically significant, validating their potential as predictive features.

---

## 4. Methodology

The project was structured as an iterative journey, starting with simple techniques and progressively incorporating more advanced solutions.

### 4.1. Data Preprocessing

A vivid text cleaning pipeline was essential for preparing the noisy tweet data. This was a two-stage process:

**Standardization:**  
A function was created to handle basic cleaning tasks like converting text to lowercase, removing "rt" artifacts, and stripping out all URLs, user mentions (@username), hashtags (#topic), and punctuation.

**Advanced NLP Processing:**  
Using the **NLTK** library, a second function performed:
- **Tokenization:** Splitting text into individual words.  
- **Custom Stopword Removal:** This was a key strategic decision. The standard NLTK stopword list was customized to preserve crucial negation and intensity words (e.g., *"not"*, *"never"*, *"very"*), which are vital for maintaining sentiment context.  
- **Lemmatization:** Reducing words to their dictionary root form (e.g., *"running"* becomes *"run"*) to group related concepts.

---

### 4.2. Modeling Journey

**Iteration 1: Classic ML Baselines**  
The initial approach was to train a suite of classic machine learning models (**Logistic Regression**, **SVC**, **Random Forest**, **XGBoost**) on the 3-class problem. While these models achieved modest accuracy (~60%), they failed at the core business problem, yielding extremely poor recall for the *Negative* class due to the severe class imbalance.

**Iteration 2: Binary Classification Success**  
To prove the viability of the approach, the problem was simplified to a **binary classifier** (*Positive vs. Negative*). This more balanced task allowed for effective model tuning. A **GridSearchCV-tuned Logistic Regression** model emerged as a champion, achieving an excellent **Macro F1-Score of 0.72**. This confirmed the feature engineering was effective and provided a strong baseline.

**Iteration 3: The Deep Learning Breakthrough (Transformers)**  
After hitting a performance plateau with traditional methods on the multiclass problem, the project moved to state-of-the-art Transformer models.

- **DistilBERT:** A general-purpose Transformer model provided a significant breakthrough, achieving a **Macro F1-Score of 0.57** and a **Negative Recall of 53%**.  
- **Twitter-RoBERTa (The Champion):** The final model, `cardiffnlp/twitter-roberta-base-sentiment-latest`, is a specialist pre-trained on billions of tweets. This model emerged as the undisputed champion.

---

## 5. Final Model and Evaluation

🏆 **Final Champion Model:** Fine-Tuned **Twitter-RoBERTa**

This model, fine-tuned on a **Google Colab GPU**, provided the best and most balanced performance on the challenging 3-class problem.

| Metric | Precision | Recall | F1-Score |
|:--------|:----------:|:-------:|:--------:|
| **Negative emotion** | 0.44 | 0.60 | 0.51 |
| **Neutral** | 0.71 | 0.73 | 0.72 |
| **Positive emotion** | 0.67 | 0.60 | 0.64 |

**Final Macro Avg F1-Score:** 0.62  
**Final Accuracy:** 67%

This model successfully solved the core business problem by identifying **60% of all negative tweets**, a task where all previous classic models had failed.

### Model Explainability with LIME

To ensure the model was not a "black box," **LIME** was used to explain its predictions. The analysis confirmed that the model was making intelligent, human-like decisions by correctly identifying sentiment-bearing words (like *"headache"*) as the primary drivers for its predictions.

---

## 6. Deployment

The final, trained **RoBERTa** model was deployed as an interactive **web application using Streamlit**. The app provides a user-friendly interface where anyone can enter a tweet and receive an instant sentiment analysis, complete with confidence scores for each class.

The model itself is hosted on the **Hugging Face Hub**, and the application is deployed on **Hugging Face Spaces**, demonstrating a complete, professional, end-to-end **MLOps pipeline**.

**Live App URL:** [https://huggingface.co/spaces/Shifterr/sentiment-analyzer](https://huggingface.co/spaces/Shifterr/sentiment-analyzer)

---

## 7. Conclusion and Recommendations

This project successfully demonstrates the journey from a complex, imbalanced dataset to a high-performing, interpretable, and deployed sentiment classification system. The iterative modeling process proved that for nuanced NLP tasks, fine-tuning a pre-trained, domain-specific Transformer model is a vastly superior strategy to classic machine learning or building deep learning models from scratch.

**Recommendations:**

- **Integrate the Deployed App:** The live Streamlit app should be integrated into the workflow of a customer support or social media team to begin flagging negative feedback in real-time.  
- **Establish a Human-in-the-Loop System:** The model's predictions should be reviewed by a human team. This not only allows for immediate customer engagement but also creates a valuable new dataset of corrected labels that can be used to re-train and further improve the model.  
- **Future Work (V2.0):** The next iteration should focus on data augmentation to create more examples of tricky, ambiguous sentences to improve the model's precision on the *Negative* class.
