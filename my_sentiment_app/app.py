# import streamlit as st
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import pandas as pd

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="SENTINEL AI",
#     page_icon="🤖",
#     layout="wide"
# )

# # --- Caching the Model (for speed) ---
# @st.cache_resource
# def load_model():
#     print("🚀 Loading the fine-tuned RoBERTa model...")
#     # !! IMPORTANT: Change 'checkpoint-296' to your actual folder name !!
#     MODEL_PATH = "checkpoint-885"
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
#     model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
#     print("✅ Model loaded successfully!")
#     return tokenizer, model

# # Load the model and tokenizer
# tokenizer, model = load_model()
# labels = ['Negative emotion', 'Neutral', 'Positive emotion']

# # --- Prediction Function (Upgraded to return probabilities) ---
# def predict_sentiment(text):
#     """
#     Takes a raw tweet string and returns the predicted sentiment and its probabilities.
#     """
#     if not text.strip():
#         return None, None

#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#     with torch.no_grad():
#         logits = model(**inputs).logits
    
#     # Convert logits to probabilities
#     probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
#     predicted_class_id = probabilities.argmax().item()
    
#     return labels[predicted_class_id], probabilities.numpy()


# # --- Building the User Interface ---

# # Custom CSS for a cooler look
# st.markdown("""
# <style>
#     .stApp {
#         background-color: #0E1117;
#     }
#     .stTextArea textarea {
#         background-color: #1A1A2E;
#         color: #E0E0E0;
#         border-radius: 10px;
#     }
#     .stButton button {
#         background-color: #1F618D;
#         color: white;
#         border-radius: 10px;
#         border: none;
#     }
# </style>
# """, unsafe_allow_html=True)


# st.title("🤖 SENTINEL AI: Twitter Sentiment Analyzer")
# st.write(
#     "Powered by a fine-tuned `Twitter-RoBERTa` model, this tool analyzes the sentiment of tweets in real-time."
# )

# # --- User Input Section ---
# user_input = st.text_area("Enter a tweet to analyze:", "The new iPhone is amazing, I love it so much!", height=100)

# if st.button("Analyze Sentiment"):
#     if not user_input.strip():
#         st.warning("Please enter some text to analyze.")
#     else:
#         with st.spinner("🧠 Sentinel is thinking..."):
#             prediction, probs = predict_sentiment(user_input)
        
#         st.write("---")
#         st.header("Analysis Result")

#         # Use columns for a cleaner layout
#         col1, col2 = st.columns([1, 2])

#         # Column 1: Display the main prediction and emoji
#         with col1:
#             st.subheader("Predicted Sentiment")
#             if prediction == 'Positive emotion':
#                 st.markdown(f"<h2 style='text-align: center; color: #2ECC71;'>{prediction} 👍</h2>", unsafe_allow_html=True)
#             elif prediction == 'Negative emotion':
#                 st.markdown(f"<h2 style='text-align: center; color: #E74C3C;'>{prediction} 👎</h2>", unsafe_allow_html=True)
#             else: # Neutral
#                 st.markdown(f"<h2 style='text-align: center; color: #3498DB;'>{prediction} 😐</h2>", unsafe_allow_html=True)
        
#         # Column 2: Display the confidence score (probabilities)
#         with col2:
#             st.subheader("Confidence Score")
#             prob_df = pd.DataFrame(probs, index=labels, columns=['Probability'])
#             prob_df = prob_df.sort_values(by='Probability', ascending=False)
            
#             # Display probabilities with progress bars
#             for label, row in prob_df.iterrows():
#                 st.write(f"**{label}:**")
#                 st.progress(float(row['Probability']))
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="SENTINEL AI",
    page_icon="🤖",
    layout="wide"
)

# --- Caching the Model ---
@st.cache_resource
def load_model():
    print("🚀 Loading the fine-tuned RoBERTa model...")
    MODEL_PATH = "Shifterr/sentiment-model-roberta" # !! Make sure this matches your folder name !!
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    print("✅ Model loaded successfully!")
    return tokenizer, model

# Load the model and tokenizer
tokenizer, model = load_model()
labels = ['Negative emotion', 'Neutral', 'Positive emotion']

# --- Prediction Function (Upgraded to return probabilities) ---
def predict_sentiment(text):
    if not text.strip():
        return None, None
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
    predicted_class_id = probabilities.argmax().item()
    return labels[predicted_class_id], probabilities.numpy()

# --- Building the User Interface ---

# Custom CSS for a cool look
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .stTextArea textarea { background-color: #1A1A2E; color: #E0E0E0; border-radius: 10px; }
    .stButton button { background-color: #1F618D; color: white; border-radius: 10px; border: none; }
</style>
""", unsafe_allow_html=True)

# --- THIS IS THE NEW, UPGRADED TITLE SECTION ---
st.markdown("<h1 style='text-align: center; color: #FFFFFF;'>🤖 SENTINEL AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #808B96;'>A Twitter Sentiment Analyzer powered by RoBERTa</h4>", unsafe_allow_html=True)
st.write("---") # Adds a horizontal line for separation

# --- User Input Section ---
user_input = st.text_area("Enter a tweet to analyze:", "The new iPhone is amazing, I love it so much!", height=100)

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("🧠 Sentinel is thinking..."):
            prediction, probs = predict_sentiment(user_input)
        
        st.header("Analysis Result")

        # Use columns for a cleaner layout
        col1, col2 = st.columns([1, 2])

        # Column 1: Display the main prediction and emoji
        with col1:
            st.subheader("Predicted Sentiment")
            if prediction == 'Positive emotion':
                st.markdown(f"<h2 style='text-align: center; color: #2ECC71;'>{prediction} 👍</h2>", unsafe_allow_html=True)
            elif prediction == 'Negative emotion':
                st.markdown(f"<h2 style='text-align: center; color: #E74C3C;'>{prediction} 👎</h2>", unsafe_allow_html=True)
            else: # Neutral
                st.markdown(f"<h2 style='text-align: center; color: #3498DB;'>{prediction} 😐</h2>", unsafe_allow_html=True)
        
        # Column 2: Display the confidence score (probabilities)
        with col2:
            st.subheader("Confidence Score")
            prob_df = pd.DataFrame(probs, index=labels, columns=['Probability'])
            prob_df = prob_df.sort_values(by='Probability', ascending=False)
            
            for label, row in prob_df.iterrows():
                st.write(f"**{label}:**")
                st.progress(float(row['Probability']))
