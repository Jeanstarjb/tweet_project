# tweet_project
### Modeling Journey

**Iteration 1: Classic ML Baselines**  

-The initial approach was to train a suite of classic machine learning models (**Logistic Regression**, **SVC**, **Random Forest**, **XGBoost**) on the 3-class problem. While these models achieved modest accuracy (~60%), they failed at the core business problem, yielding extremely poor recall for the *Negative* class due to the severe class imbalance.

**Iteration 2: Binary Classification Success** 

-To prove the viability of the approach, the problem was simplified to a **binary classifier** (*Positive vs. Negative*). This more balanced task allowed for effective model tuning. A **GridSearchCV-tuned Logistic Regression** model emerged as a champion, achieving an excellent **Macro F1-Score of 0.72**. This confirmed the feature engineering was effective and provided a strong baseline.

**Iteration 3: The Deep Learning Breakthrough (Transformers)** 

After hitting a performance plateau with traditional methods on the multiclass problem, the project moved to state-of-the-art Transformer models:

1. **DistilBERT:** A general-purpose Transformer model provided a significant breakthrough, achieving a **Macro F1-Score of 0.57** and a **Negative Recall of 53%**.  
2. **Twitter-RoBERTa (The Champion):** The final model, `cardiffnlp/twitter-roberta-base-sentiment-latest`, is a specialist model pre-trained on billions of tweets. This model emerged as the undisputed champion.

---

### Final Model and Evaluation

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
