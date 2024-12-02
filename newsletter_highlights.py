from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI
from operator import itemgetter
from dotenv import load_dotenv
import os
from task_assets import REDDIT_TASK_ASSETS, YOUTUBE_TASK_ASSETS, TOP_LEVEL_PROMPT, GENERAL_TASK_ASSETS
from collection import RedditDataCollector, YoutubeDataCollector
from prelabeling import RedditFormatter, YoutubeFormatter
import time
from datetime import datetime, timezone

class NewsletterHighlightsChain:
    def __init__(self, openai_key):
        self.openai_key = openai_key
        
        # Initialize the formatter chains
        self.reddit_formatting_chain = RunnableLambda(
            lambda x: RedditFormatter().format_multiple_posts(x['posts'])
        )
        
        self.youtube_formatting_chain = RunnableLambda(
            lambda x: YoutubeFormatter().format_multiple_videos(x['videos'])
        )
        
        # Initialize the highlight extraction chains
        self.reddit_highlights_chain = (
            RunnableLambda(lambda x: {
                'posts_and_comments': x['formatted_string']
            })
            | self.create_chain_from_task_asset(REDDIT_TASK_ASSETS['newsletter_highlights'])
        )
        
        self.youtube_highlights_chain = (
            RunnableLambda(lambda x: {
                'videos_and_comments': x['formatted_string']
            })
            | self.create_chain_from_task_asset(YOUTUBE_TASK_ASSETS['newsletter_highlights'])
        )
        
        # Initialize the script generation chain
        self.script_generation_chain = self.create_chain_from_task_asset(
            GENERAL_TASK_ASSETS['script_generation']
        )
        
        # Build the full chain
        self.full_chain = (
            RunnableParallel({
                'reddit_formatted': (
                    RunnableLambda(lambda x: {'posts': x['reddit_posts']})
                    | self.reddit_formatting_chain
                    | RunnableLambda(lambda x: {'formatted_string': x})
                    | self.reddit_highlights_chain
                ),
                'youtube_formatted': (
                    RunnableLambda(lambda x: {'videos': x['youtube_videos']})
                    | self.youtube_formatting_chain
                    | RunnableLambda(lambda x: {'formatted_string': x})
                    | self.youtube_highlights_chain
                )
            })
            | RunnableLambda(lambda x: {
                'reddit_highlights': x['reddit_formatted']['highlights'],
                'youtube_highlights': x['youtube_formatted']['highlights']
            })
            | self.script_generation_chain
        )

    def create_chain_from_task_asset(self, task_asset):
        """Creates a chain from a task asset configuration."""
        prompt_template = ChatPromptTemplate.from_messages([
            ('system', TOP_LEVEL_PROMPT),
            ('system', task_asset['task']),
            ('user', task_asset['prompt'])
        ])
        
        llm = ChatOpenAI(   
            model=task_asset.get('model', 'gpt-4o'),
            temperature=task_asset.get('temperature', 0.0),
            max_tokens=task_asset.get('max_tokens', 4096),
            openai_api_key=self.openai_key
        )

        if 'json_schema' in task_asset:
            llm = llm.with_structured_output(task_asset['json_schema'])
       
        return prompt_template | llm

    def invoke(self, reddit_posts, youtube_videos, **kwargs):
        """
        Generate a newsletter script from Reddit posts and YouTube videos.
        
        Parameters:
        - reddit_posts (list): List of Reddit post dictionaries
        - youtube_videos (list): List of YouTube video dictionaries
        - **kwargs: Additional arguments to pass to the chain
        
        Returns:
        - dict: Contains 'script' and 'estimated_duration'
        """
        return self.full_chain.invoke({
            'reddit_posts': reddit_posts,
            'youtube_videos': youtube_videos
        }, **kwargs)

if __name__ == '__main__':
    def test_newsletter_highlights():
        # Load environment variables
        load_dotenv()
        
        # Initialize collectors
        reddit_collector = RedditDataCollector({
            'client_id': os.getenv('REDDIT_CLIENT_ID'),
            'client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
            'user_agent': os.getenv('REDDIT_USER_AGENT')
        })
        
        youtube_collector = YoutubeDataCollector(os.getenv('YOUTUBE_API_KEY'))
        
        try:
            # Collect more Reddit posts with comments and filter by age
            reddit_posts = reddit_collector.get_posts_search(
                search_term="artificial intelligence",
                limit=20,  # Increased from 5
                comments_limit=3
            )
            
            # Filter Reddit posts by age (4 days)
            current_time = time.time()
            max_age = 4 * 24 * 60 * 60  # 4 days in seconds
            reddit_posts = [
                post for post in reddit_posts 
                if (current_time - post['timestamp']) <= max_age
            ][:10]  # Keep top 10 after filtering
            
            # Collect more YouTube videos with comments
            youtube_videos = youtube_collector.search_videos(
                query="artificial intelligence news",
                max_results=20  # Increased from 5
            )
            
            # Filter YouTube videos by age (4 days)
            youtube_videos = [
                video for video in youtube_videos
                if (datetime.now(timezone.utc) - datetime.fromisoformat(video['publish_time'].replace('Z', '+00:00'))).days <= 4
            ][:10]  # Keep top 10 after filtering
            
            # Add comments to each YouTube video
            for video in youtube_videos:
                video['comments'] = youtube_collector.get_comments(
                    video_id=video['video_id'],
                    max_results=3
                )
            
            # Initialize the newsletter chain
            chain = NewsletterHighlightsChain(openai_key=os.getenv('OPENAI_API_KEY'))
            
            # Generate the newsletter
            result = chain.invoke(reddit_posts, youtube_videos)
            
            # Print results
            print("\nGenerated Script:")
            print(result['script'])
            print("\nEstimated Duration:", result['estimated_duration'], "seconds")
            
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            import traceback
            traceback.print_exc()

    # Run the test
    test_newsletter_highlights()