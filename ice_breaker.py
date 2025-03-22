import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from third_parties.linkedin import scrape_linkedin_profile

load_dotenv()

if __name__ == "__main__":

    summary_template = """
    given the information {information} about a person from i want you to create.
    1. a short summary.
    2. two interesting facts about the person.
    """

    summary_prompt_template = PromptTemplate(
        template=summary_template, input_variables="information"
    )

    # llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    llm = ChatOllama(temperature=0, model="llama3")
    chain = summary_prompt_template | llm 
    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url="https://www.linkedin.com/in/elonmusk/", mock=True
    )
    res = chain.invoke(input={"information": linkedin_data})

    print(res)
