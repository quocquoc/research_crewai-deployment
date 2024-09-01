from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from dotenv import load_dotenv
import os
from textwrap import dedent
from current_time import get_time_string
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


# --- Input ---

# print("## Welcome to the youtube summarize Crew")
# print('-------------------------------')
# topic = input("What is the technology topic you want me summarize?\n")
print("Welcome to the Research Technology News Crew")
print("We will help you research the 2 latest news articles in English about a specific topic, write blog posts based on the articles, and translate the final posts into Vietnamese.")
print("---------------------------------------")
topic = input("Please enter the technology topic of your research: ")

# --- LLM ---

## If you call the openAI models, you can open the following code
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# os.environ["OPENAI_MODEL_NAME"]="gpt-4o-mini"

## If you call the Gemini models, you can open the following code
llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                           verbose=True,
                           temperature=0.5,
                           google_api_key=os.getenv("GOOGLE_API_KEY"))


# --- Tools ---

os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY')
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()


# --- Agents ---

news_researcher=Agent(
    role='News Senior Researcher',
    goal='Find the latest English news articles on technology based on the given {topic}',
    backstory=dedent(
        """\
          You are an experienced technology news researcher with a 
          keen eye for the latest trends and innovations.
          """),
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    # llm=llm
)

content_writer=Agent(
    role='Technology Content Writer',
    goal='Write engaging blog posts based on the provided news articles',
    backstory=dedent(
        """\
          You are a skilled technology writer with a talent for explaining complex 
          concepts in an easy-to-understand manner.
          """),
    verbose=True,
    allow_delegation=False,
    tools=[scrape_tool],
    # llm=llm
)

content_quality=Agent(
    role='Technology Content Quality',
    goal='Ensure the accuracy, grammar, and overall quality of the written blog posts',
    backstory=dedent(
        """\
          You are a meticulous editor with a deep understanding of technology 
          and a passion for clear, engaging writing.
          """),
    verbose=True,
    allow_delegation=True,
    # llm=llm
)

senior_translator=Agent(
    role='Technology Senior Translate',
    goal='Translate the final blog posts from English to Vietnamese, maintaining technical accuracy and reader engagement',
    backstory=dedent(
        """\
          You are a bilingual technology expert with years of experience in translating complex technical content.
          """),
    verbose=True,
    allow_delegation=False,
    # llm=llm
)

# --- Tasks ---

research_task = Task(
  description=dedent(
        """\
          Research the latest news articles in English about {topic} from internet in 2024, focusing on the US market. 
          Return {num_articles} links to the most relevant and recent articles.
          For each article, provide the title, link, a brief snippet, and the publication date.
          """),
  expected_output=dedent(
        """\
          A list of {num_articles} recent and relevant English news articles about {topic} from the US market, including:
            1. Article title
            2. Article link
            3. Brief snippet or summary
            4. Publication date
          """),
  tools=[search_tool],
  agent=news_researcher,
)

writing_task = Task(
  description=dedent(
        """\
          Read the content of each provided link and write a blog post summarizing 
          the key points and insights. Include the original link in your post.
          """),
  expected_output=dedent(
        """\
          A well-written blog post for each provided link, summarizing the key points 
          and insights from the article. Each post need include the original source link.
          """),
  tools=[scrape_tool],
  agent=content_writer,
)

quality_check_task = Task(
  description=dedent(
        """\
          Review the blog posts for accuracy, grammar, and overall quality. 
          Provide feedback and suggestions for improvement. If revisions are needed, send the feedback to the Content Writer.
          """),
  expected_output=dedent(
        """\
          A detailed quality report for each blog post, including feedback on accuracy, grammar, overall quality, and check include the original source link. 
          If revisions are needed, specific suggestions for improvement should be provided.
          """),
  agent=content_quality,
)

translation_task = Task(
  description=dedent(
        """\
          Translate the final version of the blog post from English to Vietnamese. Ensure the translation is technically 
          accurate, engaging, and ready for publication on a technology blog.
          """),
  expected_output=dedent(
        """\
          A high-quality Vietnamese translation of each blog post, maintaining technical accuracy and reader engagement. 
          The translated posts should be ready for immediate publication on a Vietnamese technology blog.
          """),
  agent=senior_translator,
  context=[writing_task]
)

# --- Crew ---

# Forming the tech-focused crew with some enhanced configurations
crew = Crew(
  agents=[news_researcher, content_writer, content_quality, senior_translator],
  tasks=[research_task, writing_task, quality_check_task, translation_task],
  process=Process.sequential,  # Optional: Sequential task execution is default
  memory=True,
  cache=True,
  max_rpm=100,
  verbose=True
)

## start the task execution process with enhanced feedback
result=crew.kickoff(inputs={'topic':topic, 'num_articles':2})
print("\n\n##############################")
print("## Here are the results of your research project:")
print("##############################\n")
print(result)
