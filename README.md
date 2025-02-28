# Project Description:

* The purpose of this project is to build and deploy a customer service coffee-shop chatbot application. The chatbot will take orders, provide detailed information about menu and items block irrelevant conversation and recommend products based on a custom recommendation engine.  

# Skills you'll acquire/built in the project:

1. Prompt Engineering - crafting precise instructions to guide the chatbot on how to generate accurate and relevant responses for a given query.

2. Retrieval Augmented Generation (RAG) - a system that allows the chatbot to generate response based on personalized data from your coffee-shop database and context provided to the system.

3. Agent-Based Systems - The customer service coffee-shop chatbot will consist of different Agents performing specific functions. E.g. Order Agents (taking orders), Details Agent (providing information), Recommendation Agents (Recommending items), Guard Agents (filtering out irrelevant conversations).
The Recommendation Agent can be connected to the recommendation engine or database to allow for contribution of outputs into the conversation. Agent-Based systems are added into Production environments across industries, making them valuable to your toolkit.

4. Market Basket Analysis Recommendation Engine - an engine that analyses items customer purchased together, allowing suggestions of complimentary products to users. This engine will be trained from scratch and integrated with the Agents-Based systems, enabling the chatbot to provide tailored suggestions into a conversational manner for the user.

5. Deployment using RunPod - deploying the LLM, embedding model, custom API and Agent-based system on RunPod (allows deployment of GPU and CPU endpoints simple, cost effective and scalable). This is a serverless infrastructure, you only focus on the codes. This is the Backend.

6. React Native - connects to the RunPods endpoints and database for building an end-to-end mobile applications. On the home screen of mobile application, we retrieved displayed items from the database where Users can filter each products based on choice; also, there will be a card screen for the total price of selected products. We connect the chatbot, deployed on RunPod to this mobile application. This is the Frontend.

7. By the end of this project, we'll have a chatbot application that takes orders, provide intelligent recommendations and provide seamless user experience while learning Prompt engineering, Advance LLM and Full-stack development. 

# Steps and Procedures for this project.

* www.runpod.io --> signup on RUNPOD if you have none, go to Billing and pay some couple of dollars to use the cloud serverless.

* Llama3 model, which is an open source, will be used for this project - llama3(meta-llama/Llama-3.1-8B-Instruct) from HuggingFace. I'm using (meta-llama/Llama-3.2-1B-Instruct). Instruct means it's finetuned to be a question/answer model using Alpaca dataset. Also, ensure you read the terms, fill your details and agree to access the model. To view the model approval, go to settings on HuggingFace, click on Gated Repositories. Wait until you see a notification of access to the model. 

* On HuggingFace, go to profile, click Access token, click write and create a new token(llama-token) for this project. Copy the token and keep safe. Token generated below:
llama-token : "insert HF_TOKEN"

* Go to RUNPOD, click Serverless, start Serverless VLLM, copy and paste both the HuggingFace token and the (meta-llama/Meta-Llama-3.2-1B-Instruct) you were granted access to. click Next, click Next again and select 48GB GPU, Max workers == 2, choose any Endpoint Name (emma_coffeeshop_project in this case), click deploy.

* After deploying the endpoint, the llm (llama-3.2-1) can be accessed via API and commands from RUNPOD. Also, test the Request in the deployment. ask any question(prompt) - what's the capital of Italy? - and wait for response by clicking RUN. This is a lot cheaper than OPENAI and can be access locally with codes.

* mkdir coffee_shop --> create a directory for the project.

* conda create -n coffeeshop_env python=3.11 --> create a venv on the directory

* conda activate coffeeshop_env --> activate the created venv to exclude its dependencies from laptop dependencies.

* code . --> open the vs-code IDE on the virtual environment.

* touch requirements.txt --> create file and include needed libraries for this project. check it out

* pip install -r requirements.txt --> install the packages

* pip install "jmespath<2.0.0,>=0.7.1" "urllib3!=2.2.0,<3,>=1.25.4"  --> to fix dependencies' error

* create a .env file --> this contains all tokens/keys needed. from RUNPOD settings, create API key, copy and post key here. Also, from RUNPOD SERVERLESS, get the BASEURL of created Endpoint -for accessing RUNPOD (https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/openai/v1)- and post here. check them out on codes

* mkdir python_code --> create folder in the working directory. Inside this folder, create a jupter notebook (prompt_engr_tutorial.ipynb). Select the Python kernel for this notebook. check codes

# Note: with the below code in the ipynb notebook, you can access an LLM API via a Serverless (RUNPOD endpoint) on the cloud, without having a GPU on local machine. View RUNPOD billing to see amount left/deducted

# combining all into a single function
def get_chatbot_response(client, model_name, messages, temperature=0):
    input_message = []
    for message in messages:
        input_message.append({'role': message['role'], 'content': message['content']})

    response = client.chat.completions.create(
        model=model_name,
        messages=input_message,
        temperature=temperature,
        top_p=0.8,
        max_tokens = 2000
    ).choices[0].message.content

    return response

messages = [
    {'role':'user', 'content':'What is the capital of Nigeria?'}
]

response = get_chatbot_response(client, model_name, messages=messages)
response

* Prompt Engineering, Input Structure and Chain of Thoughts  --> check the prompt_engr_tutorial.ipynb notebook

* RUNPOD will be use to create an embedding model endpoint. On RUNPOD click start on Infinity RUNPOD embedding, click next, click next, select 24GB GPU PRO, Max Workers = 1. Note: u can select and GPU size and max worker. Click Deploy. Click Serverless to see created endpoints.

* Test the embedding model. Click request, use input to request instead of prompt and click RUN. Wait for response. You should get list of vectors representations

* copy the URL of the embedding model and paste in .env, click save. Check notebook for codes on embeddings


# Step 1 - Recommendation Engine: data is from Kaggle, which will be used to train a recommendation engine, to suggest a supplement when an order is made. Paste the dataset in the python_code directory.

* create a new notebook in the python_code folder -- recommendation_engine_training.ipynb and pip install mlxtend. check the codes in this notebook.

* Apriori Recommendation Engine --> check codes in recommendation_engine_training.ipynb notebook

* I downloaded the following information from the tutor's GitHub link: 
1. products.jsonl - generated with LLM to describe each product and its category, 
2. Merry's_way_about_us.txt - name and information of the Coffee shop, 
3. menu_items_text.txt - price of individual product.

* Note: my own data does not have products: Ginger Biscotti and Chocolate chip Biscotti, which makes my data 16 products, 2 less than 18 products in the tutorial video.


# Step 2: Create a Firebase project for React Native AI Application

* open https://firebase.google.com/, sign in with google if not signed in, go to console and create a firebase project (coffeeshop-app or any name). Accept terms, click continue(twice) and create project.

* on firebase UI, click drop-down arrow of build and select both Realtime Database (NoSQL database to store products.jsonl) and Storage (bucket to store and access images).

* Generate Firebase Key and APPS: click project overview, click project settings, click service account, select python and generate new private key for downloads. Thereafter, click General, click React icon (</>), register app with any name (react_native_app), click register app and head to the python_code directory on VSCODE.

* update requirements.txt file with firebase-admin==6.0.1 (to access firbeabse) and google-cloud-storage==2.18.2. run pip install -r requirements.txt

* pip install colorama==0.4.6 docutils==0.16 PyYAML==6.0 s3transfer==0.10.0 rsa==4.7.2 --> resolve dependency issues.

* update the .env file with the downloaded firebase json file credentials. First, Open with VSCODE to view it, copy into .env and edit the json file --> check .env for reference

* create a firebase_upload.ipynb notebook for React Native Application --> check codes.


# Step 3: Pinecone: Vector Embedding Database

* open Pinecone and login --> I already have an account. Create an API Key for this project. (Pinecone_api_key : 'insert Pinecone_key')

* Put pinecone api-key and index in .env file. --> check the .env file for reference

* pip install pinecone --> update the requirements.txt file and check version (pip freeze | grep pinecone)

* create a pinecone notebook (build_vector_database.ipynb) in the python_code folder. --> check codes


# Step 4: Agent-Based Systems

* Agents are created for this project to distribute huge tasks to smaller tasks for different agents, with their respective prompts to achieve higher accuracy. Agents here are:

1. Guard Agents - to filter out query by the user. If user query is related to the engine, it goes to Classifier agent. If not, the user gets an unrelated response.

2. Classifier Agents sends the query to either the Order Agent (to make an order), Recommendation Agent (to make recommendations alongside the order), or Details Agent (connected to the Pinecone database, to get details on the User's query). 

* create a folder 'agents'(for different agents), development_code.py (to test developing agents), in the api directory

* add runpod==1.7.1 to requirements.txt --> just for deployment

* create a utils.py helper function in the api/agents folder. copy chatbot and embedding functions from prompt_engr_tutorial.ipynb notebook. Codes below:

def get_chatbot_response(client,model_name,messages,temperature=0):
    input_messages = []
    for message in messages:
        input_messages.append({"role": message["role"], "content": message["content"]})

    response = client.chat.completions.create(
        model=model_name,
        messages=input_messages,
        temperature=temperature,
        top_p=0.8,
        max_tokens=2000,
    ).choices[0].message.content
    
    return response

def get_embedding(embedding_client,model_name,text_input):
    output = embedding_client.embeddings.create(input = text_input,model=model_name)
    
    embedings = []
    for embedding_object in output.data:
        embedings.append(embedding_object.embedding)

    return embedings
 
* create agent_protocol.py (telling all agents to generate the output as Dict from the List input) files in the api/agent folder --> check codes

* create guard_agent.py in the api/agents folder --> check codes

* create an __init__.py (expose modules inside the agents folder outside to initialize any response).  

* from .guard_agent import GuardAgent  --> write this code in __Init__.py

* create a development.py 

# write this code below in development.py:

from agents import (GuardAgent)

def main():
    guard_agent = GuardAgent()

    messages = []
    while True:
        # Display the chat history
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("\n\nPrint Messages ...............")
        for message in messages:
            print(f"{message['role']}: {message['content']}")

        # Get user input from terminal
        prompt = input("User: ")
        messages.append({"role": "user", "content": prompt})

        # Get GuardAgent's response
        guard_agent_response = guard_agent.get_response(messages)
        print("Guard_Agent_Output:", guard_agent_response)
        messages.append(guard_agent_response)


if __name__ == "__main__":
    main()

* python3 python_code/api/development_code.py --> run this and ask any user question to the Guard Agent (e.g. I want to order a Latte please). I got this error: raise JSONDecodeError("Extra data", s, end)
json.decoder.JSONDecodeError: Extra data: line 7 column 1 (char 162) -- error was fixed

* # os.system('cls' if os.name == 'nt' else 'clear') --> do this in development_code.py

* python3 python_code/api/development_code.py --> run this and ask any user question to the Guard Agent (e.g. What's 1+3). I got this response, which shows it's working: 
Print Messages ...............
User: What's 1+3
Guard_Agent_Output: {'role': 'assistant', 'content': "Sorry, I can't help with that. Can I help you with your order?", 'memory': {'agent': 'guard_agent', 'guard_decision': 'allowed'}}

Print Messages ...............
user: What's 1+3
assistant: Sorry, I can't help with that. Can I help you with your order?

* try another User input: I want to order one Latte please and I got an error.

# update with this code below in development.py:

from agents import (GuardAgent)

def main():
    guard_agent = GuardAgent()

    messages = []
    while True:
        # Display the chat history
        os.system('cls' if os.name == 'nt' else 'clear')   --> always # this if need be
        
        print("\n\nPrint Messages ...............")
        for message in messages:
            print(f"{message['role'].capitalize()}: {message['content']}")

        # Get user input
        prompt = input("User: ")
        messages.append({"role": "user", "content": prompt})

        # Get GuardAgent's response
        guard_agent_response = guard_agent.get_response(messages)
	
	if guard_agent_response["memory"]["guard_decision"] == "not allowed":
        	messages.append(guard_agent_response)
                continue

if __name__ == "__main__":
    main()

* os.system('cls' if os.name == 'nt' else 'clear') --> unhash in development_code.py to clean up memory

* python3 python_code/api/development_code.py --> run this and ask any user question to the Guard Agent (e.g. I want to order a Latte please). No response which shows it's working and should continue to classification agent.

* create classification_agent.py in the api/agents folder --> check codes

# update the __init__.py with the code below:
from .guard_agent import GuardAgent
from .classification_agent import ClassificationAgent

# update development_code.py with the below code:
from agents import (GuardAgent,
                    ClassificationAgent)

import os 

def main():
    guard_agent = GuardAgent()
    classification_agent = ClassificationAgent()

  messages = []
    while True:
        # Display the chat history
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("\n\nPrint Messages ...............")
        for message in messages:
            print(f"{message['role'].capitalize()}: {message['content']}")

        # Get user input
        prompt = input("User: ")
        messages.append({"role": "user", "content": prompt})

        # Get GuardAgent's response
        guard_agent_response = guard_agent.get_response(messages)
        if guard_agent_response["memory"]["guard_decision"] == "not allowed":
            messages.append(guard_agent_response)
            continue
        
        # Get ClassificationAgent's response
        classification_agent_response = classification_agent.get_response(messages)
        chosen_agent=classification_agent_response["memory"]["classification_decision"]
        print("Chosen Agent: ", chosen_agent)


if __name__ == "__main__":
    main()

* # os.system('cls' if os.name == 'nt' else 'clear') --> found in developement_code.py

* python3 python_code/api/development_code.py --> run this and ask (e.g. recommend me anything please (recommendation_agent),  I want to order one Latte please (order_taking_agent), What's the price of a Latte(decision_agent) ) for response. Note: This worked just fine according to each agent.

* create details_agent.py in the api/agents folder --> check codes. It includes Pinecone vector database to get details of each item.

# update the __init__.py with the code below:
from .guard_agent import GuardAgent
from .classification_agent import ClassificationAgent
from .details_agent import DetailsAgent
from .agent_protocol import AgentProtocol

# update development_code.py with the below code:
from agents import (GuardAgent,
                    ClassificationAgent,
		    DetailsAgent,
		    AgentProtocol
		    )

from typing import Dict
import os

def main():
    guard_agent = GuardAgent()
    classification_agent = ClassificationAgent()
        
    agent_dict: Dict[str, AgentProtocol] = {
    "details_agent": DetailsAgent()
    }

  messages = []
    while True:
        # Display the chat history
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("\n\nPrint Messages ...............")
        for message in messages:
            print(f"{message['role'].capitalize()}: {message['content']}")

        # Get user input
        prompt = input("User: ")
        messages.append({"role": "user", "content": prompt})

        # Get GuardAgent's response
        guard_agent_response = guard_agent.get_response(messages)
        if guard_agent_response["memory"]["guard_decision"] == "not allowed":
            messages.append(guard_agent_response)
            continue
        
        # Get ClassificationAgent's response
        classification_agent_response = classification_agent.get_response(messages)
        chosen_agent=classification_agent_response["memory"]["classification_decision"]
        print("Chosen Agent: ", chosen_agent)

	# Get the chosen agent's response
        agent = agent_dict[chosen_agent]
        response = agent.get_response(messages)
        
        messages.append(response)

if __name__ == "__main__":
    main()

* python3 python_code/api/development_code.py --> run this and ask (e.g. What's the price of a Latte (decision_agent) for response from the Pinecone database. This User question gave a response from assistant.

* create recommendation_agent.py in the api/agents folder -->  check codes below to initialize and get_popular_recommendations function:

import json
import pandas as pd
import os
from .utils import get_chatbot_response
from openai import OpenAI
from copy import deepcopy
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())


class RecommendationAgent():
    def __init__(self,apriori_recommendation_path,popular_recommendation_path):
        self.client = OpenAI(
            api_key=os.getenv("RUNPOD_TOKEN"),
            base_url=os.getenv("RUNPOD_CHATBOT_URL"),
        )
        self.model_name = os.getenv("MODEL_NAME")

        with open(apriori_recommendation_path, 'r') as file:
            self.apriori_recommendations = json.load(file)

        self.popular_recommendations = pd.read_csv(popular_recommendation_path)
        self.products = self.popular_recommendations['product'].tolist()
        self.product_categories = self.popular_recommendations['product_category'].tolist()

     def get_popular_recommendation(self,product_categories=None,top_k=5):
        recommendations_df = self.popular_recommendations
        
        if type(product_categories) == str:
            product_categories = [product_categories]

        if product_categories is not None:
            recommendations_df =   self.popular_recommendations[self.popular_recommendations['product_category'].isin(product_categories)]
        recommendations_df = recommendations_df.sort_values(by='number_of_transactions',ascending=False)
        
        if recommendations_df.shape[0] == 0:
            return []

        recommendations = recommendations_df['product'].tolist()[:top_k]
        return recommendations 

* # all the codes in the development_code.py except this new code below to print out most popular recommendations:
from agents import (GuardAgent,
                    ClassificationAgent,
                    DetailsAgent,
                    RecommendationAgent,
                    AgentProtocol
                    )

from typing import Dict
import os
import pathlib
import sys
folder_path = pathlib.Path(__file__).parent.resolve()      # to resolve path issues

def main():

    recommendation_agent = RecommendationAgent(
        os.path.join(folder_path, "recommendation_objects/apriori_recommendations.json"),
        os.path.join(folder_path, "recommendation_objects/popular_recommendation.csv")
    )

    print(recommendation_agent.get_popular_recommendation(product_categories="Coffee")) 
    # Note: Coffee can be changed with any product_category from popular_recommendation.csv

if __name__ == "__main__":
    main()

* update the __init__.py with the code below:
from .guard_agent import GuardAgent
from .classification_agent import ClassificationAgent
from .details_agent import DetailsAgent
from .recommendation_agent import RecommendationAgent
from .agent_protocol import AgentProtocol

* python3 python_code/api/development_code.py --> run this to get the popular recommendations

* go back to recommendation_agent.py and add the get_apriori_recommendation function --> check updated codes

* update the development_code.py with the below code:
from agents import (GuardAgent,
                    ClassificationAgent,
                    DetailsAgent,
                    RecommendationAgent,
                    AgentProtocol
                    )

from typing import Dict
import os
import pathlib
import sys
folder_path = pathlib.Path(__file__).parent.resolve()      # to resolve path issues

def main():

    recommendation_agent = RecommendationAgent(
        os.path.join(folder_path, "recommendation_objects/apriori_recommendations.json"),
        os.path.join(folder_path, "recommendation_objects/popular_recommendation.csv")
    )

    print(recommendation_agent.get_apriori_recommendation(["Latte"]))
    # Note: Latte can be changed with any product from popular_recommendation.csv

if __name__ == "__main__":
    main()

* go back to recommendation_agent.py and add other NLP functions --> check updated codes

* readjust in development_code.py like these codes below:

from agents import (GuardAgent,
                    ClassificationAgent,
                    DetailsAgent,
#                    OrderTakingAgent,
                    RecommendationAgent,
                    AgentProtocol
                    )

from typing import Dict
import os
import pathlib
import sys
folder_path = pathlib.Path(__file__).parent.resolve()      # to resolve path issues

def main():

#    recommendation_agent = RecommendationAgent(
#        os.path.join(folder_path, "recommendation_objects/apriori_recommendations.json"),
#        os.path.join(folder_path, "recommendation_objects/popular_recommendation.csv")
#    )

#    print(recommendation_agent.get_popular_recommendation(product_categories="Bakery")) 
#    print(recommendation_agent.get_apriori_recommendation(["Latte"])) 

    guard_agent = GuardAgent()
    classification_agent = ClassificationAgent()


    agent_dict: Dict[str, AgentProtocol] = {
        "details_agent": DetailsAgent(),
#        "order_taking_agent": OrderTakingAgent(recommendation_agent),
        "recommendation_agent": RecommendationAgent(
                                    os.path.join(folder_path, "recommendation_objects/apriori_recommendations.json"),
                                    os.path.join(folder_path, "recommendation_objects/popular_recommendation.csv")
                                )
    }
    
    messages = []
    while True:
        # Display the chat history
    #    os.system('cls' if os.name == 'nt' else 'clear')
        
        print("\n\nPrint Messages ...............")
        for message in messages:
            print(f"{message['role']}: {message['content']}")

        # Get user input from terminal
        prompt = input("User: ")
        messages.append({"role": "user", "content": prompt})

        # Get GuardAgent's response from agent_protocol.py
        guard_agent_response = guard_agent.get_response(messages)
        if guard_agent_response["memory"]["guard_decision"] == "not allowed":   # it should print sorry...., but if allowed, continue to classification agent
            messages.append(guard_agent_response)
            continue
        
        # Get ClassificationAgent's response
        classification_agent_response = classification_agent.get_response(messages)
        chosen_agent=classification_agent_response["memory"]["classification_decision"]
        print("Chosen Agent: ", chosen_agent)

        # Get the chosen agent's response
        agent = agent_dict[chosen_agent]
        response = agent.get_response(messages)
        
        messages.append(response)

        
    
if __name__ == "__main__":
    main()


* clear the output and run python3 python_code/api/development_code.py --> ask the following: what do you recommend me to get? i got the answer below:
User: what do you recommend me to get?
Guard_Agent Output: {
  "chain of thought": "The user is asking a question about menu items, which is a valid category.",
  "decision": "recommended",
  "message": "I'd be happy to help you with some recommendations. What type of drink or pastry are you in the mood for?"
}
Classification_Agent Output: {
    "chain of thought": "The user is asking for a recommendation, so the recommendation agent is the most relevant.",
    "decision": "recommendation_agent",
    "message": ""
}
Chosen Agent:  recommendation_agent


Print Messages ...............
user: what do you recommend me to get?
assistant: • Cappuccino
• Latte
• Dark chocolate
• Chocolate Croissant
• Espresso shot


* update the utils.py script with the codes below. This can ne considered as another agent if need be
# function to fix any json output errors
def double_check_json_output(client, model_name, json_string):
    prompt = f"""You are an AI assistant. Your job is to validate and correct JSON strings. 
    Follow these rules strictly:
    1. If the JSON string is correct, return it without any modifications.
    2. If the JSON string is invalid, correct it and return only the valid JSON.
    3. Ensure the JSON keys: "chain of thought", "decision", "message", "recommendation_type", "order", "step number" are enclosed in double quotes and the structure is a proper JSON object.
    4. Do not add or remove extra information—just ensure the JSON is valid and includes all required fields.
    5. The first character of your response must be an open curly brace '{{', and the last character must be a closing curly brace '}}'.

    Here is the JSON string to validate and correct:

    ```{json_string}```
    """
    messages = [{"role": "user", "content": prompt}]
    response = get_chatbot_response(client, model_name, messages)
    response = response.replace("`", "").strip()
    json_string = response
    return json_string

* update these parts of guard_agent.py script with the double_check_json_output function:
	from .utils import get_chatbot_response, double_check_json_output

        chatbot_output = get_chatbot_response(self.client,self.model_name,input_messages)
        chatbot_output = double_check_json_output(self.client,self.model_name,chatbot_output)
        output = self.postprocess(chatbot_output)
        return output

* update these parts of classification_agent.py script with the double_check_json_output function:
	from .utils import get_chatbot_response, double_check_json_output

        chatbot_output = get_chatbot_response(self.client,self.model_name,input_messages)
        chatbot_output = double_check_json_output(self.client,self.model_name,chatbot_output)
        output = self.postprocess(chatbot_output)
        return output

* update these parts of recommendation_agent.py script with the double_check_json_output function:
	from .utils import get_chatbot_response, double_check_json_output

        chatbot_output = get_chatbot_response(self.client,self.model_name,input_messages)
        chatbot_output = double_check_json_output(self.client,self.model_name,chatbot_output)
        output = self.postprocess_classfication(chatbot_output)
        return output

* delete this part of the development_code.py below and save(ctrl s):

#    recommendation_agent = RecommendationAgent(
#        os.path.join(folder_path, "recommendation_objects/apriori_recommendations.json"),
#        os.path.join(folder_path, "recommendation_objects/popular_recommendation.csv")
#    )

#    print(recommendation_agent.get_popular_recommendation(product_categories="Bakery")) 
#    print(recommendation_agent.get_apriori_recommendation(["Latte"])) 

* clear the output and run python3 python_code/api/development_code.py. Ask the following: what do you recommend me to get OR can you recommend me anything to get? (NB: i got a response). Next, I asked: tell me something about the recommended items (Note: I got a response).

* create order_agent.py in the api/agents folder -->  check codes. Note: order_agents will be calling the recommendation_agent to recommend items from the apriori

* update the __init__.py with the code below:
from .guard_agent import GuardAgent
from .classification_agent import ClassificationAgent
from .details_agent import DetailsAgent
from .order_taking_agent import OrderTakingAgent
from .recommendation_agent import RecommendationAgent
from .agent_protocol import AgentProtocol

* readjust in development_code.py like these codes below:
from agents import (GuardAgent,
                    ClassificationAgent,
                    DetailsAgent,
                    OrderTakingAgent,
                    RecommendationAgent,
                    AgentProtocol
                    )

from typing import Dict
import os
import pathlib
#import sys
folder_path = pathlib.Path(__file__).parent.resolve()      # to resolve path issues

def main():
    guard_agent = GuardAgent()
    classification_agent = ClassificationAgent()
    recommendation_agent = RecommendationAgent(
                            os.path.join(folder_path, "recommendation_objects/apriori_recommendations.json"),
                            os.path.join(folder_path, "recommendation_objects/popular_recommendation.csv")
                            )
                            
                                    
    agent_dict: Dict[str, AgentProtocol] = {
        "details_agent": DetailsAgent(),
        "recommendation_agent": recommendation_agent,
        "order_taking_agent": OrderTakingAgent(recommendation_agent)
    }
    
    messages = []
    while True:
        # Display the chat history
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("\n\nPrint Messages ...............")
        for message in messages:
            print(f"{message['role']}: {message['content']}")

        # Get user input from terminal
        prompt = input("User: ")
        messages.append({"role": "user", "content": prompt})

        # Get GuardAgent's response from agent_protocol.py
        guard_agent_response = guard_agent.get_response(messages)
        if guard_agent_response["memory"]["guard_decision"] == "not allowed":   # it should print sorry...., but if allowed, continue to classification agent
            messages.append(guard_agent_response)
            continue
        
        # Get ClassificationAgent's response
        classification_agent_response = classification_agent.get_response(messages)
        chosen_agent=classification_agent_response["memory"]["classification_decision"]
        #print("Chosen Agent: ", chosen_agent)

        # Get the chosen agent's response
        agent = agent_dict[chosen_agent]
        response = agent.get_response(messages)
        
        messages.append(response)

        
if __name__ == "__main__":
    main()

* run python3 python_code/api/development_code.py. Ask the following: (1) I want to order a Latte please (2) can you recommend me anything to order please. I got response for both questions.

# Testing individual agents

* run python3 python_code/api/development_code.py. Ask the following: 

# Various question and responses from the Agents and bot:

Print Messages ...............
user: what's 1+3?  ( unnecessary to ask)
assistant: Sorry, I can't help with that. Can I help you with your order?
user: what do you recommend me to get?
assistant: Here are the recommendations:

• Cappuccino
• Latte
• Dark chocolate
• Chocolate Croissant
• Espresso shot
user: How much is the Latte?
assistant: Our Latte is priced at $4.75.
user: I want to order a Latte please
assistant: Sorry, I can't help with that. Can I help you with your order?
user: I want to order one Latte please 
assistant: I can help you with that. Based on your order, I recommend the following items:

1. Sugar Free Vanilla syrup
2. Carmel syrup
3. Croissant
4. Chocolate Croissant
5. Dark chocolate

Would you like to add any other items to your order?
user: No, all good for today. Thanks!
assistant: I'm glad you're enjoying your coffee at Merry's Way. I'll go ahead and put that order in for you. I'll also let our delivery team know that you'd like to pick up your order today.

To confirm, I'll just need to confirm your order. You'd like to order a Latte, correct? And would you like to add any of the items I listed earlier, such as sugar-free vanilla syrup, caramel syrup, or a croissant?
user: a croissant
assistant: I'll put in the order for a classic croissant. I'll go ahead and put that in for you. I'll also let our delivery team know that you'd like to pick up your order today.

To confirm, I'll just need to confirm your order. You'd like to order a classic croissant, correct? And would you like to add any of the items I listed earlier, such as sugar-free vanilla syrup, caramel syrup, or a croissant?
User:

# Note: this is the end of the development code.

# We now test/deploy RUNPOD-Serverless so that access will be via an API instead of the terminals.

* create agent_controller.py in python_code/api/ --> check codes(copied from development_code.py). This controls all agents deployed on RUNPOD.

* create main.py in python_code/api/ --> check codes. This is to access agents via an API on RUNPOD.

* create test_input.json in python_code/api/ --> check codes. To test/query any input on RUNPOD.

* cd python_code/api and run python3 main.py. This accesses RUNPOD-SERVERLESS via an API and calls the test_input.json script. Note: ensure to cd first before running to avoid errors of test_input.json not loading. (I got an HTTP output response from the test_input.json file).

* create a Dockerfile in python_code/api/ --> for building image, containerization and deployment.

* docker build -t coffeebot:v1 . --> build the docker image. Ensure docker desktop is up and running

* Run the docker container with the environment variables of installed packages below:
$ docker run -it \
> -e PINECONE_API_KEY="pcsk_7K7sPt_Myerwx9GofxEfQED1vvFMJgcJdt7Txt9ZxpTkKBKzX7JsFGbEeyjdk6Qrpj8Vif" \
> -e PINECONE_INDEX_NAME="coffeeshop" \
> -e RUNPOD_TOKEN='rpa_JDSCV1XFLEDGG0E56PCTAKV2RHR841T4KTRQQZUVkpjjkm' \
> -e RUNPOD_CHATBOT_URL='https://api.runpod.ai/v2/vllm-ydggzj1v0bj46m/openai/v1' \
> -e RUNPOD_EMBEDDING_URL='https://api.runpod.ai/v2/2ble084temonhm/openai/v1' \
> -e MODEL_NAME='meta-llama/Llama-3.2-1B-Instruct' \
> coffeebot:v1

* I got an HTTP output response  from the test_input.json file below:
discover_namespace_packages.py:12   2025-01-26 02:02:19,985 Discovering subpackages in _NamespacePath(['/usr/local/lib/python3.12/site-packages/pinecone_plugins'])
discover_plugins.py :9    2025-01-26 02:02:19,989 Looking for plugins in pinecone_plugins.inference
installation.py     :10   2025-01-26 02:02:20,137 Installing plugin inference into Pinecone
--- Starting Serverless Worker |  Version 1.7.1 ---
INFO   | Using test_input.json as job input.
DEBUG  | Retrieved local job: {'input': {'messages': [{'role': 'user', 'content': ' I want to order one Latte please'}]}, 'id': 'local_test'}
INFO   | local_test | Started.
_client.py          :1025 2025-01-26 02:03:08,553 HTTP Request: POST https://api.runpod.ai/v2/vllm-ydggzj1v0bj46m/openai/v1/chat/completions "HTTP/1.1 200 OK"
_client.py          :1025 2025-01-26 02:03:10,382 HTTP Request: POST https://api.runpod.ai/v2/vllm-ydggzj1v0bj46m/openai/v1/chat/completions "HTTP/1.1 200 OK"
_client.py          :1025 2025-01-26 02:03:16,096 HTTP Request: POST https://api.runpod.ai/v2/vllm-ydggzj1v0bj46m/openai/v1/chat/completions "HTTP/1.1 200 OK"
_client.py          :1025 2025-01-26 02:03:21,209 HTTP Request: POST https://api.runpod.ai/v2/vllm-ydggzj1v0bj46m/openai/v1/chat/completions "HTTP/1.1 200 OK"
_client.py          :1025 2025-01-26 02:03:27,120 HTTP Request: POST https://api.runpod.ai/v2/vllm-ydggzj1v0bj46m/openai/v1/chat/completions "HTTP/1.1 200 OK"
_client.py          :1025 2025-01-26 02:03:32,763 HTTP Request: POST https://api.runpod.ai/v2/vllm-ydggzj1v0bj46m/openai/v1/chat/completions "HTTP/1.1 200 OK"
_client.py          :1025 2025-01-26 02:03:37,619 HTTP Request: POST https://api.runpod.ai/v2/vllm-ydggzj1v0bj46m/openai/v1/chat/completions "HTTP/1.1 200 OK"
DEBUG  | local_test | Handler output: {'role': 'assistant', 'content': "Based on your order, I recommend the following items:\n\n1. Sugar Free Vanilla syrup\n2. Carmel syrup\n3. Croissant\n4. Chocolate Croissant\n5. Dark chocolate\n\nI've included all the items you've requested in the order. Would you like to add any other items to your drink or would you like me to prepare your order?", 'memory': {'agent': 'order_taking_agent', 'step number': 'Determine which task you are on based on the conversation.', 'order': [{'item': 'Latte', 'quantity': '1', 'price': '4.75'}], 'asked_recommendation_before': True}}
DEBUG  | local_test | run_job return: {'output': {'role': 'assistant', 'content': "Based on your order, I recommend the following items:\n\n1. Sugar Free Vanilla syrup\n2. Carmel syrup\n3. Croissant\n4. Chocolate Croissant\n5. Dark chocolate\n\nI've included all the items you've requested in the order. Would you like to add any other items to your drink or would you like me to prepare your order?", 'memory': {'agent': 'order_taking_agent', 'step number': 'Determine which task you are on based on the conversation.', 'order': [{'item': 'Latte', 'quantity': '1', 'price': '4.75'}], 'asked_recommendation_before': True}}}
INFO   | Job local_test completed successfully.
INFO   | Job result: {'output': {'role': 'assistant', 'content': "Based on your order, I recommend the following items:\n\n1. Sugar Free Vanilla syrup\n2. Carmel syrup\n3. Croissant\n4. Chocolate Croissant\n5. Dark chocolate\n\nI've included all the items you've requested in the order. Would you like to add any other items to your drink or would you like me to prepare your order?", 'memory': {'agent': 'order_taking_agent', 'step number': 'Determine which task you are on based on the conversation.', 'order': [{'item': 'Latte', 'quantity': '1', 'price': '4.75'}], 'asked_recommendation_before': True}}}
INFO   | Local testing complete, exiting.

* clear output, # the test_input.json in the Dockerfile and rebuild the image again with version 2 (docker build -t coffeebot:v2 .)

* Ensure you are signed in to docker hub, click the Docker icon on vscode, check for built image (coffeebot:v2 .), right click on image and push image to Docker hub. Refresh Docker hub for pushed image.

* On RUNPOD, click Severless, click New Endpoint (custom endpoint), select Docker, create a name (coffeshop_bot in this case), fill in docker image from dockerhub, select CPU, max_workers=1, add environment variable (from .env file) for Pinecone, Runpod, and Model_name, click create endpoint. Head back to Serverless and see created coffeeshop_bot endpoint on Runpod. It should be Ready.

* click on the new endpoint, click request, copy and paste the code test_input.json ({"input": {"messages": [{"role":"user","content": " I want to order one Latte please"}]}}) in the request. click Run. Note: I got a response from the Status side(right side and it showed completed below). This image can be accessed via an API endpoint.

* End of backend code.

