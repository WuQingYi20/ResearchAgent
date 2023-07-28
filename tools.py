from typing import Type, Any
from pydantic import BaseModel, Field
from langchain.tools.base import BaseTool

from search import search
from scrape import scrape_website

class ScrapeWebsiteInput(BaseModel):
    researchObject: str = Field(
        description="The objective & task that user give to the agent")
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "Scrape a website from a website url passing both url and research goal, summarize content if too long."
    args_schema: Type[BaseModel] = ScrapeWebsiteInput
    scraped_urls = []  # Store all scraped URLs here

    def _run(self, **kwargs: Any):
        researchObject = kwargs.get('researchObject')
        url = kwargs.get('url')
        result = scrape_website(researchObject, url)
        # Add the URL to the list of scraped URLs
        self.scraped_urls.append(url)
        return result

    def get_scraped_urls(self):
        # Method to get all scraped URLs
        return self.scraped_urls

    def _arun(self, args: Any):
        url = args.url
        raise NotImplementedError(
            "Can't find implementation for no research objective version")


class SearchTool(BaseTool):
    name = "Search"
    description = "useful for when you need to answer questions about current events, data. You should ask targeted questions"

    def _run(self, researchObject: str):
        return search(researchObject)

    def _arun(self, args: Any):
        raise NotImplementedError("Not implemented")

