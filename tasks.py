from crewai import Task
from textwrap import dedent



class ResearchCrewTasks:

    def research_task(self, agent, topic, num_articles):
      return Task(
          agent=agent,
          topic=topic,
          num_articles=num_articles,
          description=dedent(
            """\
              Research the latest news articles in English about topic {topic} from internet in 2024, focusing on the US market. 
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

      )
    
    
    def writing_task(self, agent):
      return Task(
          agent=agent,
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

      )
    

    def quality_check_task(self, agent):
      return Task(
          agent=agent,
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
      )
    
    def translation_task(self, agent, context):
      return Task(
          agent=agent,
          context=context,
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
      )




