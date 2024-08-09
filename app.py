import json
import os
import sys
import boto3
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader

## Titan Embeddings Model to generate Embedding

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

## Data Ingestion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding And Vector Store

from langchain.vectorstores import FAISS

## LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Create Bedrock Clients 
bedrock=boto3.client(service_name="bedrock-runtime",region_name="us-east-1")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)

## Data ingestion
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):

    # - in our testing Character split works better with this PDF data set
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,
                                                 chunk_overlap=1000)
    
    chunks=text_splitter.split_text(text)
    return chunks



## Vector Embedding and vector store

def get_vector_store(text_chunks):
    vectorstore_faiss=FAISS.from_texts(
        text_chunks,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

def get_claude_llm():
    ##create the Anthropic Model
    llm=Bedrock(model_id="anthropic.claude-v2",client=bedrock,model_kwargs={'max_tokens_to_sample':512})
    
    return llm

def get_llama3_llm():
    ##create the Anthropic Model
    llm=Bedrock(model_id="meta.llama3-70b-instruct-v1:0",client=bedrock,
                model_kwargs={'max_gen_len':512})
    
    return llm

prompt_template = """

Human: Use the following pieces of context to provide a concise answer to the question at the end but use atleast 
summarize with 250 words with detailed explanation. If you don't know the answer, just say that you don't know, 
don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']


def main():
    st.set_page_config("Query Documents")
    
    st.header("Query on contents of documents")

    user_question = st.text_input("Enter Query", label_visibility='hidden')

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload files", label_visibility='hidden', accept_multiple_files=True)

        # Select LLM type
        llm_type = st.radio("Which LLM would you like to choose?", ["Claude", "Llama"])

        # Initialize session state
        if "llm" not in st.session_state:
            st.session_state.llm = None
        if "faiss_index" not in st.session_state:
            st.session_state.faiss_index = None

        # Process button
        if st.button("Upload and Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                
                # Initialize LLM and FAISS index based on selection
                if llm_type == "Claude":
                    with st.spinner("Generating Answer with Claude..."):
                        st.session_state.faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                        st.session_state.llm = Bedrock(model_id="anthropic.claude-v2", client=bedrock, model_kwargs={'max_tokens_to_sample':512})
                        st.success("Done")
                        
                elif llm_type == "Llama":
                    with st.spinner("Generating Answer with Llama..."):
                        st.session_state.faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                        st.session_state.llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len':512})
                        st.success("Done")

        # Generate Answer button
    if st.button("Generate Answer"):
        if st.session_state.llm and st.session_state.faiss_index:  # Ensure llm and faiss_index are initialized
            response = get_response_llm(st.session_state.llm, st.session_state.faiss_index, user_question)
            st.write(response)
        else:
            st.error("Please upload and process files, and select an LLM type first.")


if __name__ == "__main__":
    main()














