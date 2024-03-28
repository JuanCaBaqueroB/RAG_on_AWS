# RAG_on_AWS
This repo includes an implementation of a RAG architecture to create a Q&amp;A app deployed in AWS able to answer questions related with Amazon SageMaker service. The app uses different models like Titan Text G1 Emebeddings and Titan Express, respectively. The app answer is designed to answer questions related with the documentation of the AWS SageMaker service.

## RAG Architecture


## Architecture in production environment:
The architecture used to deploy the web app in a production environment on AWS is presented below:

![alt text](https://github.com/JuanCaBaqueroB/RAG_on_AWS/blob/main/src/RAG_on_AWS.png)

There are other architectures that enable the web app working on. For example, the app could be run over a simple compute instance (EC2). However, to implement some requirements like privacy or security concerns could be harder than use a Docker container managed by AWS. 
