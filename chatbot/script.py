# --- Steps for this to properly run ---
# When launching. Press Cmd+Shift+P. Type Python: Select Interpreter. Select gsb570 env
# To run: run "streamlit run /Users/benlaufer/Desktop/GSB-570/Code/proof_of_concept/complete_agent/script.py" in console

# --- Libaries ---
#System & Utility Libraries
import os
import sys
import time
import json
import traceback
from pathlib import Path
from typing import Optional

#Data Handling & Storage
import numpy as np
import pandas as pd
import sqlite3
import pyarrow

#Environment & Configuration
from dotenv import load_dotenv
from botocore.config import Config

#AWS (Bedrock & Boto3)
import boto3
import botocore

#Salesforce
from simple_salesforce import Salesforce

#LangChain Core & Chat Models
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_models import BedrockChat

#LangChain Memory
from langchain.memory import ConversationBufferMemory

#LangChain Embeddings & Splitters
from langchain_community.embeddings import BedrockEmbeddings, HuggingFaceEmbeddings
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

#LangChain Tools & APIs
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsInput
from langchain.tools import Tool

#Streamlit for UI
import streamlit as st

# --- Environment ---
load_dotenv() 
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
sf_username = os.getenv("SF_USERNAME")
sf_password = os.getenv("SF_PASSWORD")
sf_security_token =  os.getenv("SF_SECURITY_TOKEN")
tavily_api_key = os.getenv('TAVILY_API_KEY')
print(aws_access_key, aws_secret_key, sf_username, sf_password, sf_security_token, tavily_api_key)


# --- AWS Client ---
aws_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2"
)


# --- Sales Force Client ---
def create_salesforce_client():
    sf = Salesforce(
        username=os.getenv("SF_USERNAME"),
        password=os.getenv("SF_PASSWORD"),
        security_token=os.getenv("SF_SECURITY_TOKEN")
    )
    return sf


# --- Read in SCHEMA of Salesforce data ---
base_path = Path(__file__).resolve().parent
schema_path = base_path.parent / 'data' / 'salesforce_schema.json'
# read & parse
with schema_path.open('r', encoding='utf-8') as f:
    salesforce_schema = json.load(f)

# --- Create Boto3 - Bedrock client ---
def get_bedrock_client(
    runtime: Optional[bool] = True,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None
):
    """
    Creates and returns a Boto3 client to interact with either 
    Amazon Bedrock's runtime (for inference) or control plane (for model management)
    """
    #seperate 2 types of services
    #bedrock-rtuntime - real-life inference (generating responses)
    if runtime:
        service_name = 'bedrock-runtime'
    else:
        #bedrock - control operations (lists available models)
        service_name = 'bedrock'

    #create clients for bedrock service using Boto3 SDK
    bedrock_runtime = boto3.client(
        #service used
        service_name=service_name,
        #data center
        region_name="us-west-2",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token
    )
    print("boto3 Bedrock client successfully created!")
    print(bedrock_runtime._endpoint)
    return bedrock_runtime


# --- Create Bedrock Client ---
bedrock_runtime = get_bedrock_client()


# --- Invoke Bedrock Model ---
def invoke_model(body, model_id, accept, content_type):
    """
    Sends a prompt or request to an Amazon Bedrock model and gets back a generated response (invokes)
    """

    try:
        #AWS API call
        response = bedrock_runtime.invoke_model(
            #converts to JSON
            body=json.dumps(body),
            #specified model that we inputted 
            modelId=model_id, 
            #input and output type
            accept=accept, 
            contentType=content_type
        )

        return response

    #error handling
    except Exception as e:
        print(f"Couldn't invoke {model_id}")
        raise e
    

# --- Generate SQL code ---
def get_soql_from_prompt(prompt, salesforce_schema):
    """
    Sends a user prompt and Salesforce schema to a Bedrock model to generate and return a SOQL query
    """

    #detailed prompt to guide LLM
    formatted_prompt = f"""
    \n\nHuman: You are a SOQL expert. Given the following database schema and user request, generate the best SQL query to answer the question.

    SCHEMA:
    {salesforce_schema}

    USER REQUEST:
    {prompt}

    IMPORTANT RULES:
    - Do NOT use aliases (e.g., "TotalRevenue") in the ORDER BY clause.
    - If you use an aggregate function (like SUM, COUNT), repeat the full expression in ORDER BY.
    - Respond with ONLY the SOQL query and nothing else. No explanations or formatting.
    - Only 1 table at a time should be refrence when doing queries. Do not refrence or attempt to join more than 1 table.

    \n\nAssistant:
    """

    #formats prompt inside list of messages 
    messages = [{"role": "user", "content": formatted_prompt}]

    #create body of API request
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 250,
        "messages": messages
    }
    #set model and formats
    modelId = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    accept = "application/json"
    contentType = "application/json"

    #invoke model
    response = invoke_model(body, modelId, accept, contentType)
    response_body = json.loads(response.get("body").read())

    sql_query = response_body["content"][0]["text"].strip()
    return sql_query


# --- Query Salesforce Data ---
def salesforce_query(prompt, salesforce_schema: str) -> str:
    """
    Execute a SOQL query against Salesforce and return all records as JSON.
    """
    #gets query from prompt
    query = get_soql_from_prompt(prompt, salesforce_schema)
    #connect to sf client
    sf = create_salesforce_client()
    #query data
    result = sf.query_all(query)
    #extract records out of dictionary
    records = result.get("records", [])

    # Remove Salesforce metadata from every record
    simplified = [
        {k: v for k, v in rec.items() if k != "attributes"}
        for rec in records
    ]

    return json.dumps(simplified, indent=2)


# --- Create Salesforce tool ---
salesforce_tool = Tool(
    name="salesforce_query_tool",
    #takes users prompt and passes it to salesforce_query to get results
    func=lambda user_prompt: salesforce_query(user_prompt, salesforce_schema),
    description="Use this tool to run queries against the Salesforce database. Provide natural language prompts like 'Get the contacts for the accounts with the highest paid industry'."
)


# --- RAG ---

#read in pickle data
base_path = Path(__file__).resolve().parent
pickle_path = base_path.parent / 'data' / 'PRIZM_Embedded.pkl'
dft = pd.read_pickle(pickle_path)


# --- Limit String Size ---
def limit_string_size(x, max_chars=2048):
    # Check if the input is a string
    if isinstance(x, str):
        return x[:max_chars]
    else:
        return x


# --- Embed Docs with cohere ---
def embed_documents_with_cohere(texts):
    """
    Takes one or more texts
    Sends them to Cohere's embedding model via AWS
    Then returns their semantic embeddings as NumPy arrays
    """
    #if input is string, will wrap into list
    if isinstance(texts, str):
        texts = [texts]  # promote single string to list
        single = True
    else:
        single = False

    #parameters of embedding model
    input_type = "clustering"
    truncate = "NONE"
    model_id = "cohere.embed-english-v3"
    #shortens long text so they fit in models max token limit
    json_params = {
        'texts': [limit_string_size(t) for t in texts],
        'truncate': truncate,
        'input_type': input_type
    }
    #calls AWS bedrock, sends payload and gets embeddings
    result = aws_client.invoke_model(
        body=json.dumps(json_params),
        modelId=model_id
    )
    #loads into python dict
    response = json.loads(result['body'].read().decode())
    #converts list of floats to Numpy array
    embeddings = [np.array(vec) for vec in response['embeddings']]
    return embeddings[0] if single else embeddings


# --- Cosine Similarity ---
def cosine_similarity(vec1, vec2):
    """
    calculates and returns cosine similarity between two vectors
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity


# --- Searches for top prizm segments ---
def search_prizm_segments(user_prompt: str) -> str:
    """
    Embeds a user prompt, compares it to stored PRIZM segment embeddings
    Then returns the top three most similar segments with descriptions and similarity scores
    """
    #creates embedding of users prompt
    query_vector = embed_documents_with_cohere(user_prompt)

    results = []
    #loop through each row pickled df
    for index, row in dft.iterrows():
        #calculates cosine similarity for row and appends score
        article_embedding = row["embedding"]
        score = cosine_similarity(article_embedding, query_vector)
        results.append((index, score))

    #sorts from most similiar to least
    results.sort(key=lambda x: x[1], reverse=True)

    top_matches = []
    #loops through results. Picks top 3 matches
    for i, (idx, score) in enumerate(results[:3]):
        #retrieves segment name and desc, then formats
        segment = dft.iloc[idx]["prizm_segment"]
        description = dft.iloc[idx]["text"]
        top_matches.append(
            f"Match #{i+1}:\n"
            f"Segment: {segment}\n"
            f"Description: {description}\n"
            f"Relevance Score: {round(score, 3)}\n"
        )

    return "\n".join(top_matches) or "No matches found."


# --- Creates RAG tool ---
rag_tool = Tool(
    name="prizm_segment_matcher",
    func=search_prizm_segments,
    description=(
        "Use this tool to find relevant PRIZM segments from a user prompt. "
        "Input should describe lifestyle, preferences, or demographic traits. "
        "The output includes matching segment names, descriptions, and relevance scores."
    )
)


# --- Tavily Web Search Tool ---
tavily_search_tool = Tool(
    name="tavily_web_search",
    func=TavilySearchResults(max_results=1, TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")),
    description=(
        "Use this tool to perform a web search when the user prompt requires current, factual, or location-specific information. "
        "Best for answering questions about recent events, trends, or external data that is not embedded."
    )
)

# --- Main Agent Body ---
#all tools available for the agent when answering prompts
tools = [
    tavily_search_tool,
    salesforce_tool,
    rag_tool
]

# define LLM (sonnet for fast responses)
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

# LLM parameters
model_kwargs =  { 
    "max_tokens": 2048,
    #temperature set to 0 to be as factual as possible
    "temperature": 0.0,
    "top_k": 250,
    "top_p": 0.9,
    "stop_sequences": ["\n\nHuman"],
}

llm = BedrockChat(
    client=aws_client,
    model_id=model_id,
    model_kwargs=model_kwargs,
)


# --- Memory ---
#store memory so that agent can remember past prompts in the given conversation
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
       memory_key="chat_history",
        return_messages=True
    )
memory = st.session_state.memory


# Get the prompt to use with our agent 
# prompt instructs agent how to behave...
# i.e. use tools, format answers, think step-by-step (structured agent)
prompt = hub.pull("hwchase17/structured-chat-agent")

# 3 Construct the structured agent with giving it the LLM to reason, the tools, and prompt
# https://python.langchain.com/v0.1/docs/modules/agents/agent_types/
agent = create_structured_chat_agent(llm, tools, prompt)

# --- Run the Agent ---
# manager of the agent
# receives input, manages thinking steps, decides when and how to call tools, handles memory, deals with errors
# this is run in a loop
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    memory=memory, 
    verbose=True, 
    handle_parsing_errors=True
)


# --- Streamlit UI ---
#basic page settings
st.set_page_config(page_title="PRIZM Agent Chat", layout="centered")
st.title("Chat Agent - Ben Laufer")

#store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

#chat input box for user
user_input = st.chat_input("Ask the agent something...")

#when user sends message, it gets sent to agent_executer
if user_input:
    response = agent_executor.invoke({"input": user_input})
    #saves input and output into chat history
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Agent", response["output"]))

# Display the whole chat history
for speaker, msg in st.session_state.chat_history:
    with st.chat_message(speaker):
        st.markdown(msg)


