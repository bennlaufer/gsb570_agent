import json
from langchain.tools import Tool
from config.env import create_salesforce_client
from services.llm import invoke_model

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


def salesforce_query(prompt, schema):
    
    query = get_soql_from_prompt(prompt, schema)

    # Guardrail check
    if not query.lower().startswith("select"):
        return "The generated SOQL was invalid or not a real query."
    
    sf = create_salesforce_client()
    result = sf.query_all(query)
    records = result.get("records", [])
    simplified = [{k: v for k, v in rec.items() if k != "attributes"} for rec in records]
    return json.dumps(simplified, indent=2)

def create_salesforce_tool(schema):
    return Tool(
        name="salesforce_query_tool",
        func=lambda user_prompt: salesforce_query(user_prompt, schema),
        description="Use this tool to query Salesforce database using a natural language prompt."
    )
