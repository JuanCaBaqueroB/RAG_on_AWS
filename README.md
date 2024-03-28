# Intro
This repo present a solution designed to address the documentation navigation challenges faced by Company X's developers. It's a Proof of Concept (POC) initiative to significantly reduce time spent by developers searching through extensive documentation and asking to experienced partners. Our solution leverages AWS services and Large Language Models (LLMs), specifically tailored to Company X's needs. By employing a RAG (Retrieval-Augmented Generation) architecture, the application provides quick and accurate responses to queries given by user based on publicly available AWS documentation. Through the integration of Titan Text G1 Embeddings and Titan Express models, the system enhances comprehension and accessibility of the documentation. 

The Q&A system also retrieves the documents used by the app to create the response, pointing the users to the source and other documents that may be relevant to what they are currently working or looking for. This version focus is on AWS SageMaker documentation, but the solution scope is replicable, extensible and adaptable to other documents or SageMaker updates.

# RAG on AWS
This repo includes an implementation of a RAG architecture to create a Q&amp;A app deployed in AWS able to answer questions related with Amazon SageMaker service. The app uses different models like Titan Text G1 Emebeddings and Titan Express, respectively. The app answer is designed to answer questions related with the documentation of the AWS SageMaker service.

## RAG Architecture
To solve the business needs we decided to implement a RAG pattern architecture. The PoC implemented have the following architecture:

![alt text](https://github.com/JuanCaBaqueroB/RAG_on_AWS/blob/main/src/RAG_architecture.png)

## Architecture in production environment:
The architecture used to deploy the web app in a production environment on AWS is presented below:

![alt text](https://github.com/JuanCaBaqueroB/RAG_on_AWS/blob/main/src/RAG_on_AWS.png)

It is important to note that the web app developed in Streamlit, as shown in the RAG architecture, will be embedded in a Docker image stored in AWS ECR and will run on AWS ECS. There are other architectures in which the web app could operate. For example, the app could run on a simple compute instance (EC2). However, implementing certain requirements such as privacy or security concerns on this architecture could be more challenging compared to using a Docker container managed by AWS.
