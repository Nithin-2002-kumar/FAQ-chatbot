### FAQ Chatbot
This is a simple, command-line chatbot designed to answer Frequently Asked Questions (FAQs) using Natural Language Processing (NLP). The chatbot uses a predefined knowledge base of questions and answers and matches user queries to the most relevant FAQ using ``TF-IDF`` vectorization and cosine similarity.

## Features
* Knowledge Base: A pre-configured dictionary of questions and answers.

* Text Preprocessing: Utilizes the SpaCy library to clean and process text by tokenizing, lemmatizing, and removing stop words.

* Query Matching: Employs TF-IDF (Term Frequency-Inverse Document Frequency) and Cosine Similarity to find the best match between a user's question and the FAQs in the knowledge base.

* Confidence Score: Displays a similarity score to show how confident the chatbot is in its answer.

* User-Friendly Interface: Provides a simple command-line interface for interaction.

## Prerequisites
To run this chatbot, you need to have Python installed along with the following libraries:

``spacy``

``scikit-learn``

# Setup
Install the required libraries by running the following command in your terminal:

```pip install spacy scikit-learn```

Download a SpaCy language model. This model is essential for the text preprocessing step. For this project, you need the en_core_web_sm model. Run the following command:

```python -m spacy download en_core_web_sm```

# Usage
Save the code in a Python file (e.g., faq_chatbot.py).

Run the script from your terminal:

```python faq_chatbot.py```

Start interacting with the chatbot. You can type your questions related to the topics in the knowledge base. Type exit to quit the chatbot.

# Customization
You can easily customize the chatbot by editing the faq_data dictionary within the script. Add or modify the questions and their corresponding answers to suit your specific needs. You can also adjust the threshold in the get_best_match function to control the minimum similarity score required for a valid answer.
