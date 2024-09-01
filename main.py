import os
from decouple import config
from crewai import Crew, Process
from textwrap import dedent
from agents import ResearchCrewAgents
from tasks import ResearchCrewTasks



class ResearchCrew:
    def __init__(self, topic):
        self.topic = topic
        self.agents = ResearchCrewAgents()
        self.tasks = ResearchCrewTasks()

    def run(self):
        # Initialize agents
        news_researcher = self.agents.news_researcher()
        content_writer = self.agents.content_writer()
        content_quality = self.agents.content_quality()
        senior_translator = self.agents.senior_translator()

        # Initialize tasks with respective agents
        research_task = self.tasks.research_task(news_researcher, self.topic, 2)
        writing_task = self.tasks.writing_task(content_writer)
        quality_check_task = self.tasks.quality_check_task(content_quality)
        translation_task = self.tasks.senior_translator(content_quality, writing_task)

        # Form the crew with defined agents and tasks
        crew = Crew(
            agents=[news_researcher, content_writer, content_quality, senior_translator],
            tasks=[research_task, writing_task, quality_check_task, translation_task],
            process=Process.sequential
        )

        # Execute the crew to carry out the research project
        return crew.kickoff()

if __name__ == "__main__":
    print("Welcome to the Research News Crew")
    print("We will help you research the 2 latest news articles in English about a specific topic, write blog posts based on the articles, and translate the final posts into Vietnamese.")
    print("---------------------------------------")
    topic = input("Please enter the topic of your research: ")

    research_crew = ResearchCrew(topic)
    result = research_crew.run()

    print("\n\n##############################")
    print("## Here are the results of your research project:")
    print("##############################\n")
    print(result)
