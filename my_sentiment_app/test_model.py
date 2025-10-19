from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Use the exact same path as your Streamlit app
MODEL_PATH = "Shifterr/sentiment_model_roberta"

try:
    print(f"🚀 Attempting to download model from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    print("\n✅ SUCCESS! The model was downloaded and loaded correctly on your local machine.")
    print("This means the problem is with the Streamlit Cloud environment, not your model.")
except Exception as e:
    print("\n❌ FAILURE! The model could not be downloaded locally.")
    print("This means the problem is with the model on the Hugging Face Hub (it's either private or the path is wrong).")
    print(f"Error details: {e}")