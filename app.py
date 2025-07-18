import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With Groq"

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

### Generate Response 
def generate_response(question, groq_api_key, llm_model, temperature, max_tokens):     
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model=llm_model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question})
    return answer

# Title of the app
st.title("Enhanced Q&A Chatbot With Groq")

# Sidebar for settings
st.sidebar.title("Settings")
groq_api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Supported Groq models
model = st.sidebar.selectbox(
    "Select Groq model",
    ["llama3-8b-8192", "llama3-70b-8192", "gemma-7b-it"]
)

# Adjust temperature and token limits
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=1000, value=300)

# Main interface
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

# Handle input
if user_input and groq_api_key:
    response = generate_response(user_input, groq_api_key, model, temperature, max_tokens)
    st.write(response)
elif user_input:
    st.warning("Please enter the Groq API Key in the sidebar.")
else:
    st.write("Please provide your input.")
