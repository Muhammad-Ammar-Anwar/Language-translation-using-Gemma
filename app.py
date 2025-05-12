import streamlit as st
import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize the model
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Create the prompt template
generic_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", generic_template), ("user", "{text}")]
)

# Output parser
parser = StrOutputParser()

# Combine into a chain
chain = prompt_template | model | parser

# Streamlit UI
st.title("Language Translator with Gemma")

text_input = st.text_area("Enter text to translate", height=150)
target_language = st.text_input("Enter target language (e.g., French, Spanish)")

if st.button("Translate"):
    if not text_input or not target_language:
        st.warning("Please enter both the text and the target language.")
    else:
        with st.spinner("Translating..."):
            try:
                output = chain.invoke({"text": text_input, "language": target_language})
                st.success("Translation:")
                st.write(output)
            except Exception as e:
                st.error(f"An error occurred: {e}")
