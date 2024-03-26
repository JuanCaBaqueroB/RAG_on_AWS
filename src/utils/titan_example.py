import boto3
import json

prompt_data="""
Act as a Shakespeare and write a poem on Genertaive AI
"""

bedrock_runtime=boto3.client(service_name="bedrock-runtime")

kwargs={
 "modelId": "amazon.titan-text-lite-v1",
 "contentType": "application/json",
 "accept": "application/json",
 "body": "{\"inputText\":\"this is where you place your input text\",\"textGenerationConfig\":{\"maxTokenCount\":4096,\"stopSequences\":[],\"temperature\":0,\"topP\":1}}"
}

kwargs={
  "modelId": "amazon.titan-text-lite-v1",
  "contentType": "application/json",
  "accept": "application/json",
  "body": "{\"inputText\":\"User: Generate synthetic data for daily product sales in various categories - include row number, product name, category, date of sale and price. Produce output in JSON format. Count records and ensure there are no more than 5.\\n\\nBot:\",\"textGenerationConfig\":{\"maxTokenCount\":1024,\"stopSequences\":[\"User:\"],\"temperature\":0,\"topP\":1}}"
}

print(kwargs)

response = bedrock_runtime.invoke_model(**kwargs)
response