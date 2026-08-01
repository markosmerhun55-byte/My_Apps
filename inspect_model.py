import joblib
import pandas as pd

# 1. Load the binary model pipeline from disk
pipeline = joblib.load("best_sms_spam_model.joblib")

# 2. Extract vectorizer and classifier components
vectorizer = pipeline.named_steps['tfidf']
classifier = pipeline.named_steps['classifier']

# 3. Retrieve feature names (vocabulary)
feature_names = vectorizer.get_feature_names_out()
print(f"Total learned words in vocabulary: {len(feature_names)}")

# 4. Extract weights and flatten array to 1D to prevent length mismatch errors
weights = classifier.coef_.ravel()

# 5. Build structured DataFrame
word_importance = pd.DataFrame({
    'Word': feature_names,
    'Weight': weights
}).sort_values(by='Weight', ascending=False)

# 6. Display results
print("\n--- Top 10 Words that Trigger SPAM Classification ---")
print(word_importance.head(10).to_string(index=False))

print("\n--- Top 10 Words that Trigger HAM Classification ---")
print(word_importance.tail(10).to_string(index=False))