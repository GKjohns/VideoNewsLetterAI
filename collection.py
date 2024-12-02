import praw
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
import os

import random
from youtube_transcript_api import YouTubeTranscriptApi
import requests

class RedditDataCollector:

    '''
    A Python class that interfaces with Reddit using the PRAW library to collect and analyze Reddit data.

    Attributes:
        reddit (praw.Reddit): An instance of the Reddit class from PRAW which handles communication with the Reddit API.
        VALID_SORT_METHODS (tuple): A tuple containing the valid sorting methods ('hot', 'top', 'new') for fetching posts.

    Methods:
        __init__(self, session_params): Initializes the RedditDataCollector instance with specific session parameters.
        extract_image_urls(self, post): Extracts and returns image URLs from a Reddit post if it contains direct images or a gallery.
        parse_post(self, post, **kwargs): Extracts data from a Reddit post and returns it as a dictionary.
        parse_comment(self, comment, **kwargs): Extracts data from a Reddit comment and returns it as a dictionary.
        get_posts_search(self, search_term, sort_by, subreddit_name, limit): Searches posts in a subreddit based on a search term.
        get_posts_subreddit(self, subreddit_name, sort_by, limit): Retrieves posts from a subreddit sorted by a specified method.
        get_flat_balanced_comments(self, submission_id, limit): Retrieves a balanced sample of comments from a post.
        format_comment_tree(self, comment_id, comments_by_id, children, depth): Formats a single comment and its children into a readable string.
        format_comments(self, comments): Formats a list of comments into a structured tree representation.
        get_comments(self, post_id, limit): Retrieves comments from a specific post.
        get_subreddits_search(self, search_term, limit): Searches for subreddits related to a specific search term.
        full_search_topics(self, search_terms, limit): Conducts a comprehensive search across multiple terms and sorts.
        get_post_with_top_comments(self, post_or_id, k): Retrieves a post and its top k comments.
        get_posts_search_with_concerns(self, search_term, user_concerns, limit, comments_limit, concern_ratio): Performs multiple searches and combines the results.

    Note:
        This class requires a valid Reddit API session setup, passed as `session_params` to the constructor.
    '''
    
    VALID_SORT_METHODS = ('hot', 'top', 'new')
    
    def __init__(self, session_params):
        if not session_params:
            raise ValueError("Reddit session parameters must be provided.")
        if not isinstance(session_params, dict):
            raise ValueError("Reddit session parameters must be a dictionary.")
        
        self.reddit = praw.Reddit(**session_params)

    def extract_image_urls(self, post):

        image_urls = []

        # Check if the post is a direct image post
        if hasattr(post, 'post_hint') and post.post_hint == 'image':
            image_urls.append(post.url)
        
        # Check if the post is a gallery
        elif hasattr(post, 'is_gallery') and post.is_gallery:
            # Reddit galleries have a media_metadata attribute containing info about each item
            for item_id in post.media_metadata:
                item = post.media_metadata[item_id]
                # Check if the item is an image (it could also be a video)
                if 'e' in item and item['e'] == 'Image':
                    image_url = item['s']['u']
                    image_urls.append(image_url)

        return image_urls

    def parse_post(self, post, **kwargs):
        
        rows = {
            'datestamp': str(pd.to_datetime(post.created * 1e9)).split(' ')[0],
            'timestamp': post.created,
            'title': post.title,
            'author': post.author.name if post.author else None,
            'score': post.score,
            'subreddit': post.subreddit.display_name,
            'body': post.selftext,
            'upvote_ratio': post.upvote_ratio,
            'comment_count': post.num_comments,
            'post_id': post.id,
            'post_url': post.url,
            'full_url': post.url,  # Add full URL for the post
            'image_urls': self.extract_image_urls(post)
        }
        
        rows.update(kwargs)
        
        return rows

    def parse_comment(self, comment, **kwargs):
        rows = {
            'datestamp': str(pd.to_datetime(comment.created * 1e9)).split(' ')[0],
            'timestamp': comment.created,
            'comment_id': comment.id,
            'body': comment.body,
            'author': comment.author.name if comment.author else None,
            'subreddit': comment.subreddit.display_name,
            'score': comment.score,
            'parent_post_id': comment.submission.id,
            'full_url': f'https://www.reddit.com{comment.permalink}',
            'parent_comment_id': comment.parent_id.split('_')[-1] if comment.parent_id else None
        }
        
        rows.update(kwargs)
        
        return rows

    def get_posts_search(self, search_term, sort_by='hot', subreddit_name='all', limit=10, comments_limit=0):
        subreddit = self.reddit.subreddit(subreddit_name)
        search_results = subreddit.search(query=search_term, limit=limit, sort=sort_by)
        
        rows = []
        for row in search_results:
            if row.stickied:
                continue
            
            parsed_post = self.parse_post(
                post=row,
                search_term=search_term,
                sort_method=sort_by
            )
            
            # Get top comments for the post if comments_limit > 0
            if comments_limit > 0:
                top_comments = self.get_comments(row.id, limit=comments_limit)
                parsed_post['comments'] = top_comments
            
            rows.append(parsed_post)
            
        return rows

    def get_posts_subreddit(self, subreddit_name, sort_by='hot', limit=10, comments_limit=0):
        if sort_by not in self.VALID_SORT_METHODS:
            raise ValueError(f"Invalid sort method. Must be one of {self.VALID_SORT_METHODS}")
        
        subreddit = self.reddit.subreddit(subreddit_name)
        top_posts = getattr(subreddit, sort_by)(limit=limit)
        
        rows = []
        for post in top_posts:
            if post.stickied:
                continue
            parsed_post = self.parse_post(post, sort_method=sort_by)
            
            # Get top comments for the post if comments_limit > 0
            if comments_limit > 0:
                top_comments = self.get_comments(post.id, limit=comments_limit)
                parsed_post['comments'] = top_comments
            
            rows.append(parsed_post)
                
        return rows

    def get_flat_balanced_comments(self, submission_id, limit=10):
        '''
        Fetches a balanced sample of comments from a Reddit post.
        The comments are selected in a balanced way to ensure a variety of comment threads are included
        '''

        submission = self.reddit.submission(id=submission_id)
        submission.comment_sort = 'top'
        submission.comments.replace_more(limit=0)

        # if limit is None, return all comments
        if limit is None:
            parsed_comments = [self.parse_comment(comment) for comment in submission.comments.list()]
            return parsed_comments

        # Calculate balanced distribution
        k = int(limit ** .5)
        comments_collected = 0
        comments_list = []

        for top_level_comment in submission.comments:
            if comments_collected >= limit:
                break
            comments_list.append(top_level_comment)
            comments_collected += 1

            # Fetch replies to the top level comment, limiting based on remaining capacity
            top_level_comment.replies.replace_more(limit=0)
            replies_to_add = min(k - 1, limit - comments_collected, len(top_level_comment.replies))
            
            for reply in top_level_comment.replies[:replies_to_add]:
                comments_list.append(reply)
                comments_collected += 1
                if comments_collected >= limit:
                    break
        
        parsed_comments = [self.parse_comment(comment) for comment in comments_list]
        
        return parsed_comments

    def format_comment_tree(self, comment_id, comments_by_id, children, depth=0):
        """Formats a single comment and its children into a string."""
        if comment_id not in comments_by_id:
            return ""
        comment = comments_by_id[comment_id]
        indent = "  " * depth
        # Clean and truncate the comment body for display
        body = comment['body'].replace('\n', ' ')[:30] + "..."
        result = f"{indent}- {comment['author']}: {body}\n"
        for child_id in children[comment_id]:
            result += self.format_comment_tree(child_id, comments_by_id, children, depth + 1)
        return result

    def format_comments(self, comments):
        
        if not comments:
            return "No comments to display."
        
        comments_by_id = {c['comment_id']: c for c in comments}
        children = defaultdict(list)

        # Populate the children mapping
        for comment in comments:
            parent_id = comment['parent_comment_id']
            children[parent_id].append(comment['comment_id'])

        # Format comments
        root_post_id = comments[0]['parent_post_id'] if comments else None
        return "".join(self.format_comment_tree(cid, comments_by_id, children, 1) for cid in children[root_post_id]).strip()



    def get_comments(self, post_id, limit=10):
        
        post = self.reddit.submission(id=post_id)
        post.comments.replace_more(limit=0)    
        
        comments = post.comments.list()
        comments = sorted(comments, key=lambda comment: comment.score, reverse=True)
        if limit is not None:
            comments = comments[:limit]

        rows = []
        for comment in comments:
            
            if comment.stickied:
                continue
            
            parsed_comment = self.parse_comment(comment)
            rows.append(parsed_comment)
            
        return rows                 

    def get_subreddits_search(self, search_term, limit=10):
        
        search_results = self.reddit.subreddits.search(search_term, limit=limit)
        
        return list(search_results)

    
        
    def full_search_topics(self, search_terms, limit=400, comments_limit=0):
        results_per_term = limit // len(search_terms) // len(self.VALID_SORT_METHODS)
        rows = []
        for term in search_terms:
            for sort_method in self.VALID_SORT_METHODS:
                search_results = self.get_posts_search(term, sort_by=sort_method, limit=results_per_term, comments_limit=comments_limit)
                rows.extend(search_results)
            
        return rows
    
    def __repr__(self) -> str:
        return f'<RedditDataCollector(user="{self.reddit.user.me()}", read_only={self.reddit.read_only})>'

    def __str__(self) -> str:
        return f'RedditDataCollector (User: {self.reddit.user.me()}, Read-only: {self.reddit.read_only})'

    def get_post_with_top_comments(self, post_or_id, k=10):
        """
        Retrieves a post and its top k comments.

        Args:
            post_or_id: A PRAW Submission object or a post ID string.
            k (int): The number of top comments to retrieve. Defaults to 10.

        Returns:
            dict: A dictionary containing the post data and top k comments.
        """
        if isinstance(post_or_id, str):
            post = self.reddit.submission(id=post_or_id)
        else:
            post = post_or_id

        # Parse the post data
        post_data = self.parse_post(post)

        # Get the top k comments
        post.comment_sort = 'top'
        post.comments.replace_more(limit=0)
        top_comments = sorted(post.comments.list(), key=lambda c: c.score, reverse=True)[:k]
        parsed_comments = [self.parse_comment(comment) for comment in top_comments]

        # Combine post data and comments
        result = post_data
        result['comments'] = parsed_comments

        return result

    def get_posts_search_with_concerns(self, search_term, user_concerns, limit=20, comments_limit=5, concern_ratio=0.2):
        # Calculate the number of posts to fetch for each search
        main_search_limit = max(1, int(limit * (1 - concern_ratio)))
        
        if user_concerns:
            concern_search_limit = max(1, int(limit * concern_ratio / len(user_concerns)))
        else:
            concern_search_limit = 0
            main_search_limit = limit

        # Perform the main search
        main_posts = self.get_posts_search(
            search_term=search_term,
            limit=main_search_limit,
            comments_limit=comments_limit,
        )

        # Perform searches with concerns
        concern_posts = []
        for concern in user_concerns:
            concern_search_term = f"{search_term} {concern}"
            concern_results = self.get_posts_search(
                search_term=concern_search_term,
                limit=concern_search_limit,
                comments_limit=comments_limit
            )
            concern_posts.extend(concern_results)

        # Combine and shuffle the results
        all_posts = main_posts + concern_posts
        random.shuffle(all_posts)

        # Deduplicate posts based on post_id
        seen_ids = set()
        unique_posts = []
        for post in all_posts:
            if post['post_id'] not in seen_ids:
                seen_ids.add(post['post_id'])
                unique_posts.append(post)

        return unique_posts[:limit]

class YoutubeDataCollector:
    '''
    A Python class that interfaces with YouTube using the requests library to collect and analyze YouTube data.

    Attributes:
        api_key (str): The API key for accessing the YouTube Data API.
    
    Methods:
        __init__(self, api_key): Initializes the YoutubeDataCollector instance with a specific API key.
        search_videos(self, query, max_results): Searches for videos based on a query.
        get_video_details(self, video_id): Retrieves details of a specific video.
        get_comments(self, video_id, max_results): Retrieves comments from a specific video.
        get_video_transcripts(self, video_ids): Retrieves transcripts for a list of video IDs.
    '''

    def __init__(self, api_key):
        if not api_key:
            raise ValueError("YouTube API key must be provided.")
        
        self.api_key = api_key

    def search_videos(self, query, max_results=10, include_transcript=True):
        url = 'https://www.googleapis.com/youtube/v3/search'
        params = {
            'part': 'snippet',
            'q': query,
            'type': 'video',
            'maxResults': max_results,
            'key': self.api_key
        }
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error for bad responses

        videos = []
        video_ids = []
        for item in response.json().get('items', []):
            video_id = item['id']['videoId']
            video_data = {
                'video_id': video_id,
                'title': item['snippet']['title'],
                'description': item['snippet']['description'],
                'channel_title': item['snippet']['channelTitle'],
                'publish_time': item['snippet']['publishTime'],
                'like_count': None,  # Placeholder, will be updated with actual data
                'comment_count': None,  # Placeholder, will be updated with actual data
                'view_count': None,  # Placeholder, will be updated with actual data
            }
            videos.append(video_data)
            video_ids.append(video_id)

        # Fetch additional details for each video
        for video in videos:
            details = self.get_video_details(video['video_id'])
            if details:
                video['like_count'] = details.get('like_count')
                video['comment_count'] = details.get('comment_count')
                video['view_count'] = details.get('view_count')  # Add view count

        if include_transcript:
            transcripts = self.get_video_transcripts(video_ids)
            for video in videos:
                video['transcript'] = transcripts.get(video['video_id'], "Transcript not available")

        return videos

    def get_video_details(self, video_id, include_transcript=False):
        url = 'https://www.googleapis.com/youtube/v3/videos'
        params = {
            'part': 'snippet,contentDetails,statistics',
            'id': video_id,
            'key': self.api_key
        }
        response = requests.get(url, params=params)
        response.raise_for_status()

        items = response.json().get('items', [])
        if not items:
            return None

        item = items[0]
        video_details = {
            'title': item['snippet']['title'],
            'description': item['snippet']['description'],
            'channel_title': item['snippet']['channelTitle'],
            'publish_time': item['snippet']['publishedAt'],
            'view_count': item['statistics'].get('viewCount'),
            'like_count': item['statistics'].get('likeCount'),
            'comment_count': item['statistics'].get('commentCount')
        }

        if include_transcript:
            transcript = self.get_video_transcripts([video_id])
            video_details['transcript'] = transcript.get(video_id, "Transcript not available")

        return video_details

    def get_comments(self, video_id, max_results=10):
        url = 'https://www.googleapis.com/youtube/v3/commentThreads'
        params = {
            'part': 'snippet',
            'videoId': video_id,
            'maxResults': max_results,
            'key': self.api_key
        }
        response = requests.get(url, params=params)
        response.raise_for_status()

        comments = []
        for item in response.json().get('items', []):
            comment_data = {
                'author': item['snippet']['topLevelComment']['snippet']['authorDisplayName'],
                'text': item['snippet']['topLevelComment']['snippet']['textDisplay'],
                'like_count': item['snippet']['topLevelComment']['snippet']['likeCount'],
                'published_at': item['snippet']['topLevelComment']['snippet']['publishedAt']
            }
            comments.append(comment_data)

        return comments

    def get_video_transcripts(self, video_ids):
        """
        Retrieves transcripts for a list of video IDs and concatenates the text.

        Args:
            video_ids (list): A list of YouTube video IDs.

        Returns:
            dict: A dictionary where keys are video IDs and values are concatenated transcript texts.
        """
        transcripts = {}
        for video_id in video_ids:
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                # Concatenate all text entries into a single string
                concatenated_text = ' '.join([entry['text'] for entry in transcript])
                transcripts[video_id] = concatenated_text
            except Exception as e:
                transcripts[video_id] = str(e)  # Store the error message if transcript retrieval fails

        return transcripts

    def __repr__(self) -> str:
        return f'<YoutubeDataCollector api_key_provided={bool(self.api_key)}>'

    def __str__(self) -> str:
        return self.__repr__()

if __name__ == '__main__':
    # Load environment variables from .env file
    load_dotenv() 

    youtube_collector = YoutubeDataCollector(os.getenv('YOUTUBE_API_KEY'))
    
    print(json.dumps(youtube_collector.search_videos('AI', include_transcript=False)[0], indent=4))