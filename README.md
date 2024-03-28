# RAG_on_AWS
This repo includes an implementation of a RAG architecture to create a Q&amp;A app deployed in AWS able to answer questions related with Amazon SageMaker service. The app uses different models like Titan Text G1 Emebeddings and Titan Express, respectively. The app answer is designed to answer questions related with the documentation of the AWS SageMaker service.

## RAG Architecture


## Architecture in production environment:
The architecture used to deploy the web app in a production environment on AWS is presented below:

![alt text](https://github.com/JuanCaBaqueroB/RAG_on_AWS/blob/main/src/RAG_on_AWS.png)

There are other architectures that enable the web app working on, for example, a compute instance in EC2. However, this architecture don't cover some requirements like privacy or security issues. 
