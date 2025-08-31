import sys
import re

try:
    import spacy
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Error: Required libraries not found.")
    print("Please install them using pip:")
    print("pip install spacy scikit-learn")
    print("\nFor SpaCy, you also need to download a model. After installing, run:")
    print("python -m spacy download en_core_web_sm")
    sys.exit(1)

# Load the small English model from SpaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Error: SpaCy model 'en_core_web_sm' not found.")
    print("Please download it by running:")
    print("python -m spacy download en_core_web_sm")
    sys.exit(1)

# --- FAQ Knowledge Base ---
# A dictionary of questions and their corresponding answers.
faq_data = {
    "What is your refund policy?": "Our refund policy allows for a full refund within 30 days of purchase, provided the item is in its original condition.",
    "How can I contact customer support?": "You can reach our customer support team via email at support@example.com or by calling us at (555) 123-4567.",
    "What are your business hours?": "Our business hours are Monday to Friday, 9:00 AM to 5:00 PM EST.",
    "Do you offer free shipping?": "Yes, we offer free standard shipping on all orders over $50.",
    "How do I track my order?": "Once your order has shipped, you will receive a tracking number via email. You can use this number to track your package on our website.",
    "What payment methods do you accept?": "We accept all major credit cards, PayPal, and Apple Pay."
}


# --- Preprocessing Function ---
def preprocess_text(text):
    """
    Cleans and preprocesses text using SpaCy.
    Tokenizes, lemmatizes, removes stop words and punctuation.
    """
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text, flags=re.I | re.A)
    # Process the text with spacy
    doc = nlp(text.lower().strip())
    # Return a list of lemmatized, non-stopword, non-punctuation tokens
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)


# Preprocess all FAQ questions and create a list of preprocessed texts
preprocessed_questions = [preprocess_text(q) for q in faq_data.keys()]

# --- Vectorization and Similarity Matching ---
# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
# Fit the vectorizer on the preprocessed FAQ questions and transform them
tfidf_matrix = vectorizer.fit_transform(preprocessed_questions)


def get_best_match(user_query, threshold=0.3):
    """
    Finds the best matching FAQ question for the user's query.
    Returns the answer and similarity score, or None if no good match is found.
    """
    # Preprocess the user's query
    processed_query = preprocess_text(user_query)
    if not processed_query:
        return "Sorry, I didn't understand that. Please try rephrasing your question.", 0.0

    # Transform the user query into a TF-IDF vector
    query_vector = vectorizer.transform([processed_query])

    # Calculate the cosine similarity between the user query vector and all FAQ question vectors
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Find the index of the best match
    best_match_index = cosine_similarities.argmax()
    best_match_score = cosine_similarities[best_match_index]

    # Check if the best match is above the similarity threshold
    if best_match_score >= threshold:
        # Get the original question corresponding to the best match index
        best_match_question = list(faq_data.keys())[best_match_index]
        return faq_data[best_match_question], best_match_score
    else:
        return "I'm sorry, I couldn't find a relevant answer. Could you please rephrase your question?", best_match_score


# --- Main Chatbot Loop ---
print("FAQ Chatbot: Hello! How can I help you today? Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("FAQ Chatbot: Goodbye!")
        break

    # Get the best answer and similarity score for the user's query
    answer, score = get_best_match(user_input)

    # Display the result to the user
    print(f"FAQ Chatbot: {answer}")
    print(f"(Similarity Score: {score:.2f})")
