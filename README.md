# Agentic Marketing Assistant

An intuitive, AI-driven marketing assistant that bridges complex company data and clear, actionable insights. Powered by Claude Sonnet 3.7, this tool integrates real-time segmentation from Claritas PRIZM and live sales figures from Salesforce, enabling both technical and non-technical users to extract strategic marketing recommendations through simple natural-language queries.

---

## Key Features

- **Natural-Language Interaction**: Query your data in plain English and receive detailed marketing insights.
- **Real-Time Data Integration**: Connects to Claritas PRIZM segmentation via embedded embeddings and Salesforce via API.
- **Automated Query Generation**: Uses Claude Sonnet 3.7 to translate user prompts into SOQL for Salesforce and semantic searches for PRIZM segments.
- **Scalable Architecture**: Virtual data storage and API-based tool design ensure effortless scaling and consistent performance.
- **Conversation Logging**: Downloadable chat logs in CSV format for record-keeping and analysis.

---

## Architecture Overview

1. **Agent Setup** (`agent_setup.py`)  
   - Initializes a structured chat agent with LangChain, pulling schema definitions for Salesforce tools and PRIZM embeddings.  
2. **Tools**  
   - **Salesforce Tool**: Generates SOQL queries, executes them, and returns JSON results.  
   - **PRIZM RAG Tool**: Performs semantic segment matching using Cohere embeddings on Claritas data.  
   - **Web Search Tool**: Provides live web data via Tavily API.  
3. **User Interface** (`ui.py`)  
   - Streamlit-based chat UI where users input questions and receive responses.  
4. **Entry Point** (`main.py`)  
   - Launches the UI.

---

## Prerequisites

- Python 3.8+  
- MySQL database (for custom data integration)  
- Streamlit  
- AWS credentials with Bedrock access  
- Salesforce account and API permissions  
- Tavily API key

---

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-org/agentic-marketing-assistant.git
   cd agentic-marketing-assistant
