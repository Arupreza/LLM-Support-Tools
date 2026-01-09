from dotenv import load_dotenv
load_dotenv()

from typing import List
from pydantic import BaseModel, Field

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


# -----------------------------
# Output schema (links ONLY)
# -----------------------------

class JobLinks(BaseModel):
    links: List[str] = Field(description="List of job posting URLs")


# -----------------------------
# Prompt (strict, no verbosity)
# -----------------------------

PROMPT = """
You are given access to a web search tool.

Task:
- Find job postings for an AI engineer using LangChain in the Bay Area.
- Extract ONLY the job posting URLs.

Rules:
- Output ONLY URLs.
- No explanation.
- Follow the format instructions strictly.

Question: {input}
"""

prompt = PromptTemplate(
    template=PROMPT,
    input_variables=["input"],
    partial_variables={
        "format_instructions": JobLinks.model_json_schema()
    },
)


# -----------------------------
# Model + tool
# -----------------------------

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
structured_llm = llm.with_structured_output(JobLinks)

search_tool = TavilySearch()


def run_search(query: str):
    return search_tool.invoke({"query": query})


# -----------------------------
# Runnable pipeline
# -----------------------------

chain = (
    RunnablePassthrough.assign(
        search_results=lambda x: run_search(x["input"])
    )
    | prompt
    | structured_llm
)


# -----------------------------
# Run
# -----------------------------

def main():
    result: JobLinks = chain.invoke(
        {
            "input": (
                "AI engineer LangChain jobs Bay Area LinkedIn"
            )
        }
    )
    print(result.links)


if __name__ == "__main__":
    main()
