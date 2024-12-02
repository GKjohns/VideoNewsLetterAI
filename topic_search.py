# Standard langchain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter

from operator import itemgetter

# System and path setup
import sys
from pathlib import Path

# Add the correct parent directory to sys.path
sys.path.append(str(Path(__file__).parent))

# Import from backend
from collection import RedditDataCollector, YoutubeDataCollector
from prelabeling import RedditFormatter, YoutubeFormatter
from task_assets import REDDIT_TASK_ASSETS, YOUTUBE_TASK_ASSETS, TOP_LEVEL_PROMPT, GENERAL_TASK_ASSETS

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# Other necessary imports
import os
import time
import json

# Initialize a rate limiter
RATE_LIMITER = InMemoryRateLimiter(
    requests_per_second=1,  # Super slow! We can only make a request once every 10 seconds
    check_every_n_seconds=0.5,  # Wake up every 100 ms to check whether allowed to make a request
    max_bucket_size=5  # Controls the maximum burst size
)

class RedditTopicSearchChain:
    '''
    A class to perform topic search and analysis using Reddit data and OpenAI's language model.

    This class initializes with credentials for Reddit and OpenAI, and sets up a series of processing chains
    to search for Reddit posts, format them, summarize them, and enhance the final report with embedded URLs.

    Attributes:
    - reddit_credentials (dict): Stores Reddit API credentials.
    - openai_key (str): OpenAI API key.
    - reddit_search_chain (RunnableLambda): Chain to search Reddit posts based on user concerns.
    - reddit_formatting_chain (RunnableLambda): Chain to format Reddit posts.
    - select_and_format_posts_chain (RunnableLambda): Chain to select and format specific Reddit posts.
    - multi_post_summary_chain (RunnableLambda): Chain to summarize multiple Reddit posts.
    - embed_post_urls_chain (RunnableLambda): Chain to embed post URLs in the report.
    - initial_parallel_chain (RunnableParallel): Initial parallel chain to process search parameters and posts.
    - formatting_parallel_chain (RunnableParallel): Chain to format posts and manage search parameters.
    - summary_parallel_chain (RunnableParallel): Chain to generate a summary report.
    - selection_parallel_chain (RunnableParallel): Chain to select posts used in the report.
    - enhancement_parallel_chain (RunnableParallel): Chain to enhance the report with embedded URLs.
    - full_chain (RunnableParallel): The complete processing chain combining all steps.
    '''

    def __init__(self, openai_key, reddit_user_agent, reddit_client_id, reddit_client_secret):
        '''
        Initializes the RedditTopicSearchChain with OpenAI and Reddit credentials.

        Parameters:
        - openai_key (str): OpenAI API key.
        - reddit_user_agent (str): Reddit user agent.
        - reddit_client_id (str): Reddit client ID.
        - reddit_client_secret (str): Reddit client secret.
        '''
        # Initialize Reddit credentials
        self.reddit_credentials = {
            'user_agent': reddit_user_agent,
            'client_id': reddit_client_id,
            'client_secret': reddit_client_secret
        }

        self.openai_key = openai_key

        # Initialize Chains
        self.reddit_search_chain = (
            RunnableLambda(lambda params: {
                'search_term': params['search_term'],
                'user_concerns': params['user_concerns'],
                'limit': params.get('total_posts', 10),
                'comments_limit': params.get('max_comments', 3)
            })
            | RunnableLambda(lambda search_params: RedditDataCollector(
                session_params=self.reddit_credentials
            ).get_posts_search_with_concerns(**search_params))
        )

        self.reddit_formatting_chain = RunnableLambda(lambda x: RedditFormatter().format_multiple_posts(x['posts']))

        self.select_and_format_posts_chain = (
            RunnableLambda(lambda params: {
                'posts': [post for post in params['all_posts'] if post['post_id'] in params['posts_used']],
            })
            | RunnableLambda(lambda x: RedditFormatter().format_multiple_posts(x['posts']))
            | RunnableLambda(lambda formatted_posts: {
                'formatted_string': formatted_posts
            })
        )

        # Full Chain
        self.initial_parallel_chain = RunnableParallel({
            'search_params': RunnablePassthrough(),
            'all_posts': self.reddit_search_chain
        })

        self.formatting_parallel_chain = RunnableParallel({
            'formatted_string': (
                RunnableLambda(lambda x: {'posts': x['all_posts']})
                | RunnableLambda(lambda x: RedditFormatter().format_multiple_posts(x['posts']))
            ),
            'all_posts': itemgetter('all_posts'),
            'search_params': itemgetter('search_params')
        })

        self.summary_parallel_chain = RunnableParallel({
            'report': (
                RunnableLambda(lambda x: {
                    'topic': x['search_params']['search_term'], 
                    'posts_and_comments': x['formatted_string']
                })
                | self.create_chain_from_task_asset(
                    REDDIT_TASK_ASSETS['multi_post_summary_report']
                )
            ),
            'all_posts': itemgetter('all_posts')
        })

        self.selection_parallel_chain = RunnableParallel({
            'posts_used': (
                RunnableLambda(lambda x: {
                    'posts': [post for post in x['all_posts'] if post['post_id'] in x['report']['posts_used']]
                })
                | RunnableLambda(lambda filtered_posts: {
                    'formatted_string': RedditFormatter().format_multiple_posts(filtered_posts['posts'])
                })
            ),
            'report': itemgetter('report'),
            'all_posts': itemgetter('all_posts')
        })

        self.enhancement_parallel_chain = RunnableParallel({
            'report': (
                RunnableLambda(lambda x: {
                    'report': x['report'],
                    'posts_used': x['posts_used']
                })
                | self.create_chain_from_task_asset(
                    REDDIT_TASK_ASSETS['embed_post_urls_in_report']
                )
                | itemgetter('report')
            ),
            # We'll return just the report for now
            # 'all_posts': itemgetter('all_posts')
        })

        self.full_chain = (
            self.initial_parallel_chain
            | self.formatting_parallel_chain
            | self.summary_parallel_chain
            | self.selection_parallel_chain
            | self.enhancement_parallel_chain
        )

    def create_chain_from_task_asset(self, task_asset):
        """
        Converts a task_asset dictionary into a chain.

        Parameters:
        - task_asset (dict): A dictionary containing the keys 'prompt', 'model', 'temperature', 'max_tokens', and 'json_schema'.

        Returns:
        - A chain that can be invoked with the appropriate input.
        """
        
        prompt_template = ChatPromptTemplate.from_messages([
            ('system', TOP_LEVEL_PROMPT),
            ('system', task_asset['task']),
            ('user', task_asset['prompt'])
        ])
        
        llm = ChatOpenAI(   
            model=task_asset.get('model', 'gpt-4o'),
            temperature=task_asset.get('temperature', 0.0),
            max_tokens=task_asset.get('max_tokens', 4096),
            openai_api_key=self.openai_key,
            rate_limiter=RATE_LIMITER
        )

        if 'json_schema' in task_asset:
            llm = llm.with_structured_output(task_asset['json_schema'])
       
        return prompt_template | llm

    def invoke(self, search_params, **kwargs):
        """
        Invokes the full topic search chain with the given search parameters.

        Parameters:
        - search_params (dict): A dictionary containing 'search_term', 'user_concerns', 'total_posts', and 'max_comments'.
        - **kwargs: Additional keyword arguments to pass to the chain's invoke method.

        Returns:
        - The result of the full chain execution.
        """
        return self.full_chain.invoke(search_params, **kwargs)

    def get_runnable_chain(self):
        """
        Returns the full runnable chain for Reddit topic search.
        """
        return self.full_chain

class YoutubeTopicSearchChain:
    '''
    A class to perform topic search and analysis using YouTube data and OpenAI's language model.

    This class initializes with credentials for YouTube and OpenAI, and sets up a series of processing chains
    to search for YouTube videos, format them, summarize them, and enhance the final report with embedded URLs.

    Attributes:
    - youtube_api_key (str): YouTube API key.
    - youtube_search_chain (RunnableLambda): Chain to search YouTube videos based on user concerns.
    - youtube_formatting_chain (RunnableLambda): Chain to format YouTube videos.
    - select_and_format_videos_chain (RunnableLambda): Chain to select and format specific YouTube videos.
    - multi_video_summary_chain (RunnableLambda): Chain to summarize multiple YouTube videos.
    - embed_video_urls_chain (RunnableLambda): Chain to embed video URLs in the report.
    - full_chain (RunnableParallel): The complete processing chain combining all steps.
    '''

    def __init__(self, openai_key, youtube_api_key):
        '''
        Initializes the YoutubeTopicSearchChain with OpenAI and YouTube credentials.

        Parameters:
        - openai_key (str): OpenAI API key.
        - youtube_api_key (str): YouTube API key.
        '''
        self.youtube_api_key = youtube_api_key
        self.openai_key = openai_key

        # Initialize Chains
        self.youtube_search_chain = (
            RunnableLambda(lambda search_params: {
                'search_params': search_params,
                'all_videos': YoutubeDataCollector(api_key=self.youtube_api_key)\
                    .search_videos(
                        query=search_params['search_term'], 
                        max_results=5,
                        include_transcript=True
                    )
            })
            | RunnableLambda(lambda x: {
                'videos': x['all_videos'],
                'search_params': x['search_params']
            })
        )
        
        self.summarize_transcript_chain = (
            RunnableLambda(lambda x: {
                'videos': x['videos'],
                'search_params': x['search_params'],
                'summaries': self.create_chain_from_task_asset(YOUTUBE_TASK_ASSETS['summarize_transcript']).map().invoke(x)
            })
            | RunnableLambda(lambda x: {
                'videos': [
                    video | {
                        'transcript_summary': summary['transcript_summary'],
                        'quotes': summary['quotes']
                    }
                    for video, summary in zip(x['videos'], x['summaries'])
                ],
                'search_params': x['search_params'],
            })
        )

        self.youtube_formatting_chain = RunnableLambda(lambda x: {
            'topic': x['search_params']['search_term'],
            'videos_and_comments': YoutubeFormatter().format_multiple_videos(x['videos']),
            'all_videos': x['videos']
        })
        
        self.generate_report_chain = (
            RunnableLambda(lambda x: {
                'topic': x['topic'],
            'report_data': self.create_chain_from_task_asset(YOUTUBE_TASK_ASSETS['multi_video_summary_report']).invoke(x),
                'all_videos': x['all_videos']
            })
            | RunnableLambda(lambda x: {
                'topic': x['topic'],
                'report': x['report_data']['report'],
                'videos_used': x['report_data']['videos_used'],
                'all_videos': x['all_videos']
            })
        )

        self.select_and_format_videos_chain = RunnableLambda(lambda x: {
            'report': x['report'],
            'videos_used_string': (
                RunnableLambda(lambda x: {
                    'videos': [video for video in x['all_videos'] if video['video_id'] in x['videos_used']],
                })
                | RunnableLambda(lambda x: YoutubeFormatter().format_multiple_videos(x['videos']))
            ),
            'all_videos': x['all_videos']
        })
        
        self.embed_video_urls_chain = (
            RunnableLambda(lambda x: {
                'all_videos': x['all_videos'],
                'report': (
                    self.create_chain_from_task_asset(YOUTUBE_TASK_ASSETS['embed_video_urls_in_report']).invoke(x)
            )
            })
            | RunnableLambda(lambda x: {
                'report': x['report']['report'],
                'all_videos': x['all_videos']
            })
        )

        # Full Chain
        self.full_chain = (
            self.youtube_search_chain
            | self.summarize_transcript_chain
            | self.youtube_formatting_chain
            | self.generate_report_chain
            | self.select_and_format_videos_chain
            | self.embed_video_urls_chain
        )

    def create_chain_from_task_asset(self, task_asset):
        """
        Converts a task_asset dictionary into a chain.

        Parameters:
        - task_asset (dict): A dictionary containing the keys 'prompt', 'model', 'temperature', 'max_tokens', and 'json_schema'.

        Returns:
        - A chain that can be invoked with the appropriate input.
        """
        
        prompt_template = ChatPromptTemplate.from_messages([
            ('system', TOP_LEVEL_PROMPT),
            ('system', task_asset['task']),
            ('user', task_asset['prompt'])
        ])
        
        llm = ChatOpenAI(   
            model=task_asset.get('model', 'gpt-4o'),
            temperature=task_asset.get('temperature', 0.0),
            max_tokens=task_asset.get('max_tokens', 4096),
            openai_api_key=self.openai_key,
            rate_limiter=RATE_LIMITER
        )

        if 'json_schema' in task_asset:
            llm = llm.with_structured_output(task_asset['json_schema'])
       
        return prompt_template | llm

    def invoke(self, search_params, **kwargs):
        """
        Invokes the full topic search chain with the given search parameters.

        Parameters:
        - search_params (dict): A dictionary containing 'search_term', 'user_concerns', and 'total_videos'.
        - **kwargs: Additional keyword arguments to pass to the chain's invoke method.

        Returns:
        - The result of the full chain execution.
        """
        return self.full_chain.invoke(search_params, **kwargs)

    def get_runnable_chain(self):
        """
        Returns the full runnable chain for YouTube topic search.
        """
        return self.full_chain

def create_chain_from_task_asset(task_asset, openai_key):
    """
    Converts a task_asset dictionary into a chain.

    Parameters:
    - task_asset (dict): A dictionary containing the keys 'prompt', 'model', 'temperature', 'max_tokens', and 'json_schema'.

    Returns:
    - A chain that can be invoked with the appropriate input.
    """
    
    
    
    prompt_template = ChatPromptTemplate.from_messages([
        ('system', TOP_LEVEL_PROMPT),
        ('system', task_asset['task']),
        ('user', task_asset['prompt'])
    ])
    
    llm = ChatOpenAI(   
        model=task_asset.get('model', 'gpt-4o'),
        temperature=task_asset.get('temperature', 0.0),
        max_tokens=task_asset.get('max_tokens', 4096),
        openai_api_key=openai_key,
        rate_limiter=RATE_LIMITER
    )

    if 'json_schema' in task_asset:
        llm = llm.with_structured_output(task_asset['json_schema'])
    
    return prompt_template | llm

def build_chain_for_platforms(config, platforms):
    """
    Builds the processing chain based on the specified platforms.

    Parameters:
    - config (dict): Configuration containing API keys and credentials.
    - platforms (list): List of platforms to include in the report ('reddit', 'youtube').

    Returns:
    - A RunnableParallel chain for the specified platforms.
    """
    chains = {}

    if 'reddit' in platforms:
        chains['reddit_report'] = RedditTopicSearchChain(
            openai_key=config['openai_key'],
            reddit_user_agent=config['reddit_user_agent'],
            reddit_client_id=config['reddit_client_id'],
            reddit_client_secret=config['reddit_client_secret']
        ).full_chain

    if 'youtube' in platforms:
        chains['youtube_report'] = YoutubeTopicSearchChain(
            openai_key=config['openai_key'],
            youtube_api_key=config['youtube_api_key']
        ).full_chain

    if len(chains) > 1:
        return RunnableParallel(chains) | create_chain_from_task_asset(
            GENERAL_TASK_ASSETS['combine_platform_reports'],
            config['openai_key']
        )
    elif len(chains) == 1:
        return list(chains.values())[0]
    else:
        raise ValueError("No valid platforms specified in 'platforms'.")

if __name__ == '__main__':
    load_dotenv()

    # Load each Reddit credential separately
    REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT')
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')

    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

    config = {
        'openai_key': OPENAI_API_KEY,
        'youtube_api_key': YOUTUBE_API_KEY,
        'reddit_user_agent': REDDIT_USER_AGENT,
        'reddit_client_id': REDDIT_CLIENT_ID,
        'reddit_client_secret': REDDIT_CLIENT_SECRET
    }

    # Example usage of the topic analysis pipeline with user concerns
    search_parameters = {
        'search_term': 'Zelensky',
        'user_concerns': ['politics', 'war'],
        'platforms': ['reddit', 'youtube']  # Specify platforms here
    }

    start_time = time.time()
    chain = build_chain_for_platforms(config, search_parameters['platforms'])
    combined_result = chain.invoke(search_parameters)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
    
    print(combined_result)