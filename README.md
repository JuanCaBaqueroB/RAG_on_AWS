# Intro
This repo present a solution designed to address the documentation navigation challenges faced by Company X's developers. It's a Proof of Concept (POC) initiative to significantly reduce time spentby developers searching through extensive documentation. Our solution leverages AWS services and Large Language Models (LLMs), specifically tailored to Company X's needs. By employing a RAG (Retrieval-Augmented Generation) architecture, the application provides quick and accurate responses to queries given by user based on publicly available AWS documentation. Through the integration of Titan Text G1 Embeddings and Titan Express models, our system enhances comprehension and accessibility of the documentation. While initial focus is on AWS SageMaker documentation, the solution framework is extensible and adaptable to accommodate diverse internal documentation needs.

# RAG on AWS
This repo includes an implementation of a RAG architecture to create a Q&amp;A app deployed in AWS able to answer questions related with Amazon SageMaker service. The app uses different models like Titan Text G1 Emebeddings and Titan Express, respectively. The app answer is designed to answer questions related with the documentation of the AWS SageMaker service.

## RAG Architecture
To solve the business needs we decided to implement a RAG pattern architecture. The PoC implemented have the following architecture:

![alt text](https://github.com/JuanCaBaqueroB/RAG_on_AWS/blob/main/src/RAG_architecture.png)

## Architecture in production environment:
The architecture used to deploy the web app in a production environment on AWS is presented below:

![alt text](https://github.com/JuanCaBaqueroB/RAG_on_AWS/blob/main/src/RAG_on_AWS.png)

There are other architectures that enable the web app working on. For example, the app could be run over a simple compute instance (EC2). However, implement some requirements like privacy or security concerns on this architecture could be harder than use a Docker container managed by AWS. 
