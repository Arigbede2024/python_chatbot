# Streamlit app
import streamlit as st
from script import *
def main():
    st.title(" Python Chatbot")

    user_input = st.text_input("Ask me a question:")
    if user_input:
        response = chatbot(user_input)
        st.write("Chatbot:", response)

if __name__ == "__main__":
    main()