"""
========================================================
LangChain Agent Tutorial (Fully Commented Reference)
========================================================

Goal:
- Create a LangChain agent
- Use an LLM (OpenAI)
- Use a web search tool (Tavily)
- Enforce structured output using Pydantic
- Make the code easy to revisit and extend later

This file is intentionally verbose and educational.
"""

# ------------------------------------------------------
# 1. Standard & Third-Party Imports
# ------------------------------------------------------

# Used for typing structured lists in Pydantic models
from typing import List

# Pydantic is used to strictly define the agent's output schema
from pydantic import BaseModel, Field

# Used to load API keys from a .env file
from dotenv import load_dotenv


# ------------------------------------------------------
# 2. Load Environment Variables
# ------------------------------------------------------

# This loads variables such as:
# - OPENAI_API_KEY
# - TAVILY_API_KEY
#
# These should be defined in a `.env` file at the project root.
load_dotenv()


# ------------------------------------------------------
# 3. LangChain Core Imports
# ------------------------------------------------------

# Factory function to create a reasoning + tool-using agent
from langchain.agents import create_agent

# Base class for defining custom tools (not used yet, but imported for extensibility)
from langchain.tools import tool

# Standard message format expected by LangChain agents
from langchain_core.messages import HumanMessage

# OpenAI chat-based LLM wrapper
from langchain_openai import ChatOpenAI

# Tavily web search tool (used for real-time internet search)
from langchain_tavily import TavilySearch


# ------------------------------------------------------
# 4. Define Output Schemas (Critical for Reliability)
# ------------------------------------------------------

class Source(BaseModel):
    """
    Represents a single source used by the agent.

    Why this exists:
    - Prevents hallucinated citations
    - Makes sources machine-readable
    - Useful for downstream UI or logging
    """
    url: str = Field(
        description="The URL of the source"
    )


class AgentResponse(BaseModel):
    """
    Defines the structured output format of the agent.

    The agent MUST return:
    - answer: Natural language response
    - sources: List of URLs backing the answer
    """
    answer: str = Field(
        description="The agent's answer to the query"
    )

    sources: List[Source] = Field(
        default_factory=list,
        description="List of sources used to generate the answer"
    )


# ------------------------------------------------------
# 5. Initialize the Language Model
# ------------------------------------------------------

# ChatOpenAI is a wrapper around OpenAI's chat completion API
# gpt-3.5-turbo is chosen for cost efficiency and fast iteration
llm = ChatOpenAI(
    model="gpt-3.5-turbo"
)

# ------------------------------------------------------
# 6. Define Available Tools
# ------------------------------------------------------

# TavilySearch allows the agent to:
# - Search the web
# - Retrieve fresh, real-world information
# - Provide source URLs
tools = [
    TavilySearch()
]


# ------------------------------------------------------
# 7. Create the Agent
# ------------------------------------------------------

# create_agent() combines:
# - The LLM (reasoning engine)
# - Tools (actions it can take)
# - A strict output schema (AgentResponse)
#
# This ensures:
# - Tool usage is automatic
# - Output is always structured
agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse
)


# ------------------------------------------------------
# 8. Main Execution Function
# ------------------------------------------------------

def main():
    """
    Entry point of the script.

    Sends a human query to the agent and prints the result.
    """

    # HumanMessage is required because agents operate on message objects,
    # not raw strings.
    query = HumanMessage(
        content=(
            "Search for 3 job postings for an AI engineer "
            "using LangChain in the Bay Area on LinkedIn "
            "and list their details."
        )
    )

    # agent.invoke() triggers:
    # - Reasoning
    # - Tool usage (TavilySearch)
    # - Structured response generation
    result = agent.invoke(
        {
            "messages": query
        }
    )

    # The result is already validated against AgentResponse
    print(result)


# ------------------------------------------------------
# 9. Script Entry Guard
# ------------------------------------------------------

# Ensures this file runs only when executed directly
# (not when imported as a module)
if __name__ == "__main__":
    main()
