import streamlit as st

st.set_page_config(
    page_title="TaxGPT",
    page_icon="",
    layout="centered",
    initial_sidebar_state="collapsed"
)
# Custom CSS 
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #000000;
        margin: 0;
        padding: 0;
    }
    .header {
        background-color: #000000;   
        padding: 20px;
        color: white;
        font-family: Söhne, sans-serif;
        text-align: center;
    }
    .header h1 {
        margin: 0;
        font-size: 4em;
        font-weight: 400;
    }
    .nav {
        background-color: #000000;
        display: flex;
        justify-content: center;
        padding: 10px 0;
    }
    .nav a {
        color: #d1d5db;
        text-decoration: none;
        margin: 0 15px;
        font-weight: 500;
        font-size: 1em;
        transition: color 0.2s;
    }
    .nav a:hover {
        color: #10b981;
    }
    .hero {
        text-align: center;
        padding: 60px 20px;
        background-color: #363738;
        border-radius: 10px;
        margin: 30px auto;
        width: 80%;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .hero h1 {
        font-size: 2.5em;
        color: #ffffff;
    }
    .hero p {
        font-size: 1.2em;
        color: #ffffff;
        margin-bottom: 20px;
    }
    .btn-container {
        margin-top: 20px;
    }
    .btn-container a {
        text-decoration: none;
        padding: 10px 20px;
        margin: 5px;
        background-color: #10b981;
        color: white;
        border-radius: 5px;
        font-size: 1em;
        transition: background-color 0.2s;
    }
    .btn-container a:hover {
        background-color: #059669;
    }
    .footer {
        text-align: center;
        padding: 20px;
        margin-top: 50px;
        font-size: 0.9em;
        color: #6b7280;
    }
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
        padding: 20px;
    }
    [data-testid="stSidebar"] h1 {
        color: #10b981;
        font-size: 1.5em;
    }
    </style>
""", unsafe_allow_html=True)

# Navigation bar
st.markdown("""
<div class="nav">
    <a href="#about">About</a>
    <a href="#features">Features</a>
    <a href="#how-it-works">How It Works</a>
    <a href="#get-started">Get Started</a>
</div>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <h1>Introducing TaxGPT</h1>
</div>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero">
    <h1>Your AI-Powered Tax Assistant</h1>
    <p>Effortlessly navigate your tax complexities with TaxGPT.</p>
    <div class="btn-container">
        <a href="/chat">Get Started</a>
    </div>
</div>
""", unsafe_allow_html=True)

# About Section
st.markdown("""
<div id="about" class="section">
    <h2>About</h2>
    <p>Our platform is designed to provide precise and reliable answers to all your tax-related questions. 
    Powered by Retrieval-Augmented Generation (RAG), it delivers real-time, personalized assistance.</p>
</div>
""", unsafe_allow_html=True)

# Features Section
st.markdown("""
<div id="features" class="section">
    <h2>Features</h2>
    <ul>
        <li>Secure and fast responses</li>
        <li>Powered by the latest AI technologies</li>
        <li>Seamless integration with your queries</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# How It Works Section
st.markdown("""
<div id="how-it-works" class="section">
    <h2>How It Works</h2>
    <p>Our tool uses advanced document embeddings and AI models to retrieve, process, and answer your queries 
    in real-time. Whether you have PDFs, web searches, or custom documents, we’ve got you covered.</p>
</div>
""", unsafe_allow_html=True)



# Footer
st.markdown("""
<div class="footer">
    <p>© 2025 Your Personal Tax Assistant. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)


