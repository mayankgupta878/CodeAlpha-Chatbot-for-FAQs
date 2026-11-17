import streamlit as st
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK tokenizer data quietly
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

st.set_page_config(page_title="FAQ Chatbot", page_icon="🤖")
st.title("🤖 AI Customer Support FAQ Chatbot")
st.write("Ask any question about our tech store, orders, returns, or warranties.")

# Custom FAQ Dataset
faqs = [
    {"q": "What are your shipping options and delivery times?", 
     "a": "We offer Standard Shipping (3-5 business days) and Express Shipping (1-2 business days)."},
    {"q": "How can I track my order?", 
     "a": "Once shipped, you will receive a tracking link via email and SMS to monitor your package in real time."},
    {"q": "What is your return and refund policy?", 
     "a": "You can return any unused item within 30 days of delivery for a full refund or exchange."},
    {"q": "Do your electronics come with a warranty?", 
     "a": "Yes, all electronic devices include a standard 1-year manufacturer warranty covering hardware defects."},
    {"q": "What payment methods do you accept?", 
     "a": "We accept UPI, Credit/Debit Cards, Net Banking, and Cash on Delivery (COD) for eligible pin codes."},
    {"q": "How do I cancel or modify my order?", 
     "a": "Orders can be canceled or modified within 2 hours of placement from the 'My Orders' section."},
    {"q": "How can I contact customer support?", 
     "a": "You can reach our 24/7 support team at support@techstore.example.com or call 1800-123-4567."}
]

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    filtered = [w for w in tokens if w not in stop_words]
    return " ".join(filtered)

# Preprocess FAQ questions
processed_questions = [preprocess(item["q"]) for item in faqs]
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(processed_questions)

def get_response(user_query):
    clean_query = preprocess(user_query)
    if not clean_query.strip():
        return "Could you please rephrase your question with a bit more detail?"
    
    query_vec = vectorizer.transform([clean_query])
    similarities = cosine_similarity(query_vec, faq_vectors).flatten()
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]
    
    if best_score < 0.2:
        return "I'm sorry, I don't have an exact answer for that. Please contact support@techstore.example.com for further help."
    return faqs[best_idx]["a"]

# Chat history state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you today?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
        
    reply = get_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)