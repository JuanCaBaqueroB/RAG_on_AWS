import pandas as pd
import boto3
import markdown
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import S3DirectoryLoader

# Load md files from S3
def read_md_files_from_s3():
    s3 = boto3.client('s3')

    loader = S3DirectoryLoader("sagemakerdocuments")
    data_s3 = loader.load()
    
    return data_s3

# Get a list with the documents retrieved
def get_unique_docs(answer):
    used_docs=[]
    for i in range(0,len(answer['source_documents'])):
        used_docs.append(answer['source_documents'][i].metadata['source'])

    unique_values = set(used_docs)
    #print(unique_values)
    
    return unique_values




"""
# Specify the bucket name
bucket_name = 'sagemakerdocuments'


try:
    # List objects in the bucket
    response = s3.list_objects_v2(Bucket=bucket_name)

    # Print object keys
    docs_list=[]
    if 'Contents' in response:
        print("Objects in bucket:")
        for obj in response['Contents']:
            print(obj['Key'])
            docs_list.append(obj['Key'])
    else:
        print("Bucket is empty")
except Exception as e:
    print("An error occurred:", e)

    
    
faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings,allow_dangerous_deserialization=True)
llm=get_titan_llm() 
user_question="What is AWS SageMaker?"

def get_response_llm(llm,vectorstore_faiss,query):
    global answer
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
    answer['source_documents'][0].metadata['source']
    #print(answer['source'])
    #return answer['result']

get_response_llm(llm,faiss_index,user_question)

"""