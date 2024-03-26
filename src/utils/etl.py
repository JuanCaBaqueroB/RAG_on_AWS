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












s3_uri = "https://s3.console.aws.amazon.com/s3/buckets/sagemakerdocuments?region=us-east-1&bucketType=general&tab=properties"

#Load dataframe to DataLake
def load_to_data_lake(df,name):
    object_name = f's3://data-lake-jcbb/{name}.csv'#_'{time.strftime("%d_%m_%Y_%H_%M")}.csv'
    print(f"-------------------->Guardando {name}.csv en s3",object_name)
    df.to_csv(object_name,index=False)

load_to_data_lake(df2,"Bogota")