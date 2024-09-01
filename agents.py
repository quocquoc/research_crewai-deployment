from crewai import Agent
import os
from crewai_tools import SerperDevTool, ScrapeWebsiteTool 
from langchain_google_genai import ChatGoogleGenerativeAI
from textwrap import dedent



class ResearchCrewAgents:

    def __init__(self):
        # Initialize tools if needed
        self.serper = SerperDevTool()
        self.web_scrape=ScrapeWebsiteTool()


       ## If you call the Gemini models, you can open the following code
        llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                           verbose=True,
                           temperature=0.5,
                           google_api_key=os.getenv("GOOGLE_API_KEY"))
        
        # CHANGE YOUR MODEL HERE
        self.selected_llm = llm


    def news_researcher(self):
        return Agent(
        role='News Senior Researcher',
        goal='Find the latest English news articles on technology based on the given {topic}',
        backstory=dedent(
        """\
          You are an experienced technology news researcher with a 
          keen eye for the latest trends and innovations.
          """),
        verbose=True,
        allow_delegation=False,
        llm=self.selected_llm,
        max_iter=3,
        tools=[self.serper],
        )
    
    
    def content_writer(self):
        return Agent(
        role='Technology Content Writer',
        goal='Write engaging blog posts based on the provided news articles',
        backstory=dedent(
        """\
          You are a skilled technology writer with a talent for explaining complex 
          concepts in an easy-to-understand manner.
          """),
        verbose=True,
        allow_delegation=False,
        llm=self.selected_llm,
        max_iter=3,
        tools=[self.web_scrape],
        )
    
    
    def content_quality(self):
        return Agent(
        role='Technology Content Quality',
        goal='Ensure the accuracy, grammar, and overall quality of the written blog posts',
        backstory=dedent(
        """\
          You are a meticulous editor with a deep understanding of technology 
          and a passion for clear, engaging writing.
          """),
        verbose=True,
        allow_delegation=True,
        llm=self.selected_llm,
        max_iter=3,
        )
    
    def senior_translator(self):
        return Agent(
        role='Technology Senior Translate',
        goal='Translate the final blog posts from English to Vietnamese, maintaining technical accuracy and reader engagement',
        backstory=dedent(
        """\
          You are a bilingual technology expert with years of experience in translating complex technical content.
          """),
        verbose=True,
        allow_delegation=False,
        llm=self.selected_llm,
        max_iter=3,
        )
