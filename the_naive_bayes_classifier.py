from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample data
texts = ["Free prize now!", "Meeting at noon", "Win a free ticket"]
labels = ["spam", "ham", "spam"]  # 0=ham, 1=spam

# Step 1: Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)  # Document-term matrix
print("Vocabulary:", vectorizer.get_feature_names_out())
# Output: ['free', 'meeting', 'now', 'prize', 'ticket', 'win']

# Step 2: Train classifier
model = MultinomialNB()
model.fit(X, labels)

# Predict new text
test_text = ["Free meeting"]
X_test = vectorizer.transform(test_text)
print("Prediction:", model.predict(X_test))  # Output: ['spam']