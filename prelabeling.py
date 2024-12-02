class RedditFormatter:
    
    def format_comment(self, comment):
        clipped_body = comment["body"][:500].strip()
        author = comment.get("author", "unknown")
        
        return f'[score {comment["score"]}] /u/{author}: {clipped_body}'
    
    def format_post(self, post, n_comments=0):
        lines = []
        lines.append('# Reddit Post')
        lines.append(f'Subreddit: {post["subreddit"]}')
        lines.append(f'Title: {post["title"]}')
        lines.append(f'Author: {post["author"]}')
        lines.append(f'Score: {post["score"]}')
        if post['body']:
            lines.append(f'Body (clipped): {post["body"][:1000]}')
        
        if n_comments > 0 and post['comments']:
            lines.append('\n# Comments')
            comments = post['comments'] if len(post['comments']) < n_comments else post['comments'][:n_comments]
            for comment in comments:
                lines.append(self.format_comment(comment))
                
        return '\n'.join(lines)

    def format_multiple_posts(self, posts, max_posts=30, max_title_length=100, max_body_length=200, max_comments=3):
        if not posts:
            return "No posts to display."

        formatted_posts = []
        for post in posts[:max_posts]:
            lines = []
            lines.append(f"Post ID: {post['post_id']}:")
            lines.append(f"r/{post['subreddit']} | u/{post['author']} | Score: {post['score']}")
            lines.append(f"Title: {post['title'][:max_title_length]}{'...' if len(post['title']) > max_title_length else ''}")
            
            if post['body']:
                body = post['body'][:max_body_length]
                lines.append(f"Body: {body}{'...' if len(post['body']) > max_body_length else ''}")
            
            if post['comments']:
                lines.append("Top comments:")
                # Sort comments by score in descending order and get top k
                top_comments = sorted(post['comments'], key=lambda x: x['score'], reverse=True)[:max_comments]
                for comment in top_comments:
                    author = comment.get("author", "unknown")
                    body = comment['body'][:100]  # Clip comment to 100 characters
                    lines.append(f"- u/{author} (ID: {comment['comment_id']}, Score: {comment['score']}): {body}{'...' if len(comment['body']) > 100 else ''}")
            formatted_posts.append('\n'.join(lines))

        return '\n\n---------\n\n'.join(formatted_posts)


class YoutubeFormatter:
    
    def format_video(self, video, max_description_length=200, max_transcript_length=500):
        lines = []
        lines.append(f'# YouTube Video: {video["title"]}')
        lines.append(f'{video["channel_title"]} | Published: {video["publish_time"]}')
        lines.append(f'Description: {video["description"][:max_description_length]}{"..." if len(video["description"]) > max_description_length else ""}')
        
        # Use transcript_summary if available
        if 'transcript_summary' in video and video['transcript_summary']:
            lines.append(f'Transcript Summary: {video["transcript_summary"]}')
        
        # Use quotes if available
        if 'quotes' in video and video['quotes']:
            lines.append('Quotes:')
            for quote in video['quotes']:
                lines.append(f'     - "{quote}"')
        
        return '\n'.join(lines)
    
    def format_comment(self, comment, max_text_length=100):
        text = comment['text'][:max_text_length]
        if len(comment['text']) > max_text_length:
            text += '...'
        return f'- {comment["author"]}: {text} (Likes: {comment["like_count"]})'
    
    def format_video_with_comments(self, video, comments, max_comments=5):
        lines = [self.format_video(video)]
        if comments:
            lines.append('\n# Comments')
            for comment in comments[:max_comments]:
                lines.append(self.format_comment(comment))
        
        return '\n'.join(lines)

    def format_multiple_videos(self, videos, max_videos=30, max_title_length=100, max_description_length=200, max_transcript_length=500, max_comments=3):
        if not videos:
            return "No videos to display."

        formatted_videos = []
        for video in videos[:max_videos]:
            lines = []
            lines.append(f"Video ID: {video['video_id']}:")
            lines.append(f"Title: {video['title'][:max_title_length]}{'...' if len(video['title']) > max_title_length else ''}")
            lines.append(f"Channel: {video['channel_title']}")
            lines.append(f"Published: {video['publish_time']}")
            lines.append(f"Description: {video['description'][:max_description_length]}{'...' if len(video['description']) > max_description_length else ''}")
            
           # Use transcript_summary if available
            if 'transcript_summary' in video and video['transcript_summary']:
                lines.append(f'Transcript Summary: {video["transcript_summary"]}')
            
            # Use quotes if available
            if 'quotes' in video and video['quotes']:
                lines.append('Quotes:')
                for quote in video['quotes']:
                    lines.append(f'     - {quote}')
            
            if 'comments' in video:
                lines.append("Top comments:")
                top_comments = sorted(video['comments'], key=lambda x: x['like_count'], reverse=True)[:max_comments]
                for comment in top_comments:
                    lines.append(self.format_comment(comment))
            
            formatted_videos.append('\n'.join(lines))

        return '\n\n---------\n\n'.join(formatted_videos)