from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

def main():

    info = """Cell Broadcast System(CBS) is a broadcast mechanism that delivers the same message
    simultaneously to all user equipment (UE) within a target cell area, enabling rapid dissemination
    of emergency information regardless of network congestion. Recently, Advances in software-defined
    radio (SDR) technologies and the widespread availability of open-source LTE stacks have enabled
    low-cost environments that can emulate LTE base stations, introducing new security threats.
    In particular, the System Information Block Type 12 (SIB12) channel used for public alert delivery
    operates as a one-way broadcast and applies neither encryption nor authentication mechanisms, making
    it vulnerable to forged emergency alert attacks via rogue base stations."""

    summary_template = """
        Give me a small summary of the following text:

        Text:
        {info}

        Return:
        1) Short summary
        2) Two interesting points
        """

    summary_prompt = PromptTemplate(
        input_variables=["info"],
        template=summary_template
    )

    llm = ChatOllama(
        temperature=0.7,
        model="llama3.1:8b"
    )

    chain = summary_prompt | llm
    response = chain.invoke({"info": info})

    print(response.content)

if __name__ == "__main__":
    main()