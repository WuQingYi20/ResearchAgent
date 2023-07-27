from typing import Any
from dotenv import load_dotenv
import os
import requests
import json
from bs4 import BeautifulSoup
import base64
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools.base import BaseTool
from langchain.schema import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from pydantic import BaseModel, Field
from typing import Type
import streamlit as st

load_dotenv()
OPENAI_key = os.getenv('OPENAI_API_KEY')
SERP_API_KEY = os.getenv('SERP_API_KEY')
BROWSERLESS_API_KEY = os.getenv('BROWSERLESS_API_KEY')


def search():
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": "apple inc",
        "gl": "se"
    })
    headers = {
        'X-API-KEY': SERP_API_KEY,
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)
    return response.text


def scrape_website(researchObject: str, url: str):
    try:
        auth_header = 'Basic ' + \
            base64.b64encode(BROWSERLESS_API_KEY.encode()).decode()
        print("Scraping website...")
        headers = {
            'Cache-Control': 'no-cache',
            'Content-Type': 'application/json',
            'Authorization': auth_header,
        }
        data = {
            "url": url
        }
        data_json = json.dumps(data)
        response = requests.post(
            'https://chrome.browserless.io/content',
            headers=headers,
            data=json.dumps(data),
        )
        response.raise_for_status()  # raise exception for 4XX or 5XX responses

        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("Content: ", text)

        if len(text) > 10000:
            text = summary(researchObject, text)
        return text

    except requests.exceptions.HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except requests.exceptions.RequestException as req_err:
        print(f'Request error occurred: {req_err}')
    except Exception as e:
        print(f'An error occurred: {e}')


def summary(RAObject, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=[" ", ".", ";", ","],
        chunk_size=10000,
        chunk_overlap=500,
        length_function=len,
    )
    docs = text_splitter([content])
    map_summary = """
    write a summary of the following text for {RAObject}:
    "{text}"
    Summary:
    """
    map_prompt_template = PromptTemplate(
        template=map_summary, input_variables=["text", "RAObject"]
    )
    summary_chain = load_summarize_chain(
        llm,
        map_prompt_template,
        chian_type="map_reduce",
        verbose=True
    )

    output = summary_chain.run(input_document=docs, RAObject=RAObject)
    print("use summary funcs")
    return output


class ScrapeWebsiteInput(BaseModel):
    researchObject: str = Field(
        description="The objective & task that user give to the agent")
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "Scrape a website from a website url passing both url and research goal,summarize content if too long."
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, args: Any):
        researchObject = args.researchObject
        url = args.url
        return scrape_website(researchObject, url)

    def _arun(self, args: Any):
        url = args.url
        raise NotImplementedError(
            "can't find implementation for no research objective version")


class SearchTool(BaseTool):
    name = "Search"
    description = "useful for when you need to answer questions about current events, data. You should ask targeted questions"

    def _run(self, args: Any):
        return search()

    def _arun(self, args: Any):
        raise NotImplementedError("Not implemented")


tools = [
    SearchTool(),
    ScrapeWebsiteTool()
]

system_message = SystemMessage(
    content="""
You are a world class researcher, who can do detailed research on any topic and produce facts based on results;
You do not make things up, you will try as hard as possible to gather facts & data to back up research
Please make sure you complete the objective above with the following rules:
1/ You should do enough research to gather as much as information as possible about the objective
2/ If there are url of relevant links & articles, you will scrape it to gather more information
3/ After scraping & search, you should think "is there any news things I should search & scrape based on the data I collected to increase research quality?" if answer is yes, continue; But dont do this more than 3 iterations
4/ You should not make things up, you should only write facts & data that you have gathered
5/ In the final output, you should include all reference data & links to back up your research
6/ In the final output, you should include all reference data & links to back up your research
"""
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message
}

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory
)

# simple web UI


def main():
    st.set_page_config(page_title="Research Agent", page_icon=":books:")
    st.header("Research Agent")

    query = st.text_input("Enter your research objective")
    if query:
        st.write("doing research for", query)

        result = agent({"input": query})

        st.info(result['output'])


if __name__ == '__main__':
    main()
