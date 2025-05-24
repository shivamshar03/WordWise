import streamlit as st
# import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()

st.set_page_config(page_title="WordWise", page_icon=":robot:")
st.header("Hey, Ask me something & I will give out similar things")

# converting text into vectores (numeric)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load CSV data
loader = CSVLoader(file_path='myData.csv', csv_args={
    'delimiter': ',',
    'quotechar': '"',
    'fieldnames': ['Words']
})
data = loader.load()

# Create FAISS vector database
db = FAISS.from_documents(data, embeddings)

user_input = st.text_input("You: ", key="input")
submit = st.button('Find similar Things')

if submit and user_input:
    docs = db.similarity_search(user_input)

    st.subheader("Top Matches:")
    for i, doc in enumerate(docs[:3]):
        st.text(f"{i+1}. {doc.page_content}")

    # Explanation

    # llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.7)
    # response = llm.invoke(user_input)
    # st.subheader("Explanation:")
    # st.write(response.content)


