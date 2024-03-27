import pandas as pd
import boto3
import markdown
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Create an S3 client
s3 = boto3.client('s3')

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

docs = data_ingestion()
get_vector_store(docs)













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
    #print(answer['source'])
    #return answer['result']

get_response_llm(llm,faiss_index,user_question)










