import json
import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
import numpy as np
import streamlit as st
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector Embedding And Vector Store
from langchain.vectorstores import FAISS

## LLm Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Bedrock Clients
bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)

## Data ingestion
def data_ingestion():
    markdown_path = "C:/Users/jbaquerb/Documents/Juan/RAG_on_AWS/src/data/amazon-sagemaker-toolkits.md"
    loader = UnstructuredMarkdownLoader(markdown_path)
    data = loader.load()

    # - in our testing Character split works better with this PDF data set
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,
                                                 chunk_overlap=1000)
    
    docs=text_splitter.split_documents(data)
    return docs

## Vector Embedding and vector store
def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

def get_titan_llm():
    ##create the Titan Model
    llm=Bedrock(model_id="amazon.titan-text-lite-v1",client=bedrock,
                model_kwargs={'maxTokenCount':1024})
    
    return llm


prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end. Summarize with 
350 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
Don't answer questions that are not related with AWS Sagemaker documentation.
Don't use 'Human' label in your answer.
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
    st.set_page_config("Q&A AWS Sagemaker")
    
    st.header("AWS SageMaker documentation Q&A app")

    user_question = st.text_input("Ask everything related to AWS SageMaker service")

    with st.sidebar:

        # Display a SageMaker image from a URL
        st.image('https://d1.awsstatic.com/product-marketing/IronMan/AWS-service-icon_sagemaker.5ccec16f16a04ed56cb1d7f02dcdada8de261923.png', caption='Documentation up-to-date', use_column_width=True)

        st.title("Update documentation:")
        
        if st.button("Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")
        

    if st.button("Submit question - Titan"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings,allow_dangerous_deserialization=True)
            llm=get_titan_llm() 
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

if __name__ == "__main__":
    main()














