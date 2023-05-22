import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def main():
    st.set_page_config(page_title="📚 Chat with your PDF")
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    st.title('📚 Chat w/ PDF')
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    #extract pdf text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        # split text into sentences
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=2000,
            chunk_overlap=200,
                )

        chunks = splitter.split_text(text)
        
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="davinci")
        
        knowledge_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
        
        # user question
        user_question = st.text_input("Ask a question")
        if user_question:
            # load question answering chain
            docs = knowledge_store.similarity_search(user_question)
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)
                st.write(response)
            
if __name__ == '__main__':
    main()