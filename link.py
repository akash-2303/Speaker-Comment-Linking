import json
import argparse
from datetime import datetime, timedelta
import heapq
import os
import re

def load_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# def categorize_comments_by_debate(comments, debates):
#     categorized = {debate: [] for debate in debates}
#     for comment in comments:
#         if 'publishedAt' not in comment:
#             continue
#         date_match = re.search(r'(\d{4})-(\d{2})-\d{2}T', comment['publishedAt'])
#         if date_match:
#             comment_date = datetime.strptime(date_match.group(0)[:-1], "%Y-%m-%dT")
#             closest_debate = min(debates, key=lambda d: abs(d - comment_date))
#             categorized[closest_debate].append(comment)
#     return categorized

def categorize_comments_by_debate(comments, debates):
    categorized = {debate: [] for debate in debates}
    for comment in comments:
        if 'publishedAt' not in comment:
            continue
        published_at = comment['publishedAt']
        if not isinstance(published_at, str):
            print(f"Unexpected data type for 'publishedAt': {published_at} (type: {type(published_at)})")
            continue

        try:
            comment_date = datetime.fromisoformat(published_at.rstrip('Z'))
            closest_debate = min(debates, key=lambda d: abs(d - comment_date))
            categorized[closest_debate].append(comment)
        except ValueError as e:
            print(f"Error parsing date from comment 'publishedAt': {published_at}, Error: {str(e)}")
    return categorized

def find_closest_comments(speaker_time, user_comments, time_threshold=timedelta(minutes=5), k=3):
    relevant_comments = []
    for comment in user_comments:
        comment_time = datetime.fromisoformat(comment['publishedAt'].replace('Z', ''))
        time_diff = abs(comment_time - speaker_time)
        
        if time_diff <= time_threshold:
            relevant_comments.append((time_diff, comment))
    
    if not relevant_comments:
        relevant_comments = heapq.nsmallest(k, 
                                            [(abs(comment_time - speaker_time), comment)
                                             for comment in user_comments], key=lambda x: x[0])
    return [comment for _, comment in relevant_comments]

def link_speaker_to_comments(speaker_segments, user_comments, transcript_file_name):
    combined_data = []
    for segment in speaker_segments:
        speaker_time = datetime.strptime(segment['start'], '%H:%M:%S')
        matched_comments = find_closest_comments(speaker_time, user_comments)
        
        combined_data.append({
            'transcript': transcript_file_name,
            'speaker': segment['speaker'],
            'start': segment['start'],
            'end': segment['end'],
            'message': segment['message'],
            'linked_comments': matched_comments
        })
    return combined_data

def main(transcript_dir, comments_file, output_file):
    debates = {
        "2024-06-27": datetime(2024, 6, 27),
        "2024-09-10": datetime(2024, 9, 10),
        "2024-10-01": datetime(2024, 10, 1)
    }
    debate_files = {
        "2024-06-27": "BidenVSTrump_with_speakers.json",
        "2024-09-10": "harris_vs_trump_combined.json",
        "2024-10-01": "vance_vs_walz_combined.json"
    }

    comments = load_data(comments_file)
    categorized_comments = categorize_comments_by_debate(comments, debates.values())

    all_debates_data = []
    for debate_date, debate_time in debates.items():
        transcript_file_name = debate_files[debate_date]
        transcript_path = os.path.join(transcript_dir, transcript_file_name)
        
        if os.path.exists(transcript_path):
            speaker_segments = load_data(transcript_path)
            linked_data = link_speaker_to_comments(speaker_segments, categorized_comments[debate_time], transcript_file_name)
            all_debates_data.extend(linked_data)
        else:
            print(f"Transcript file {transcript_file_name} for debate on {debate_date} not found at path: {transcript_path}.")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_debates_data, f, indent=4)
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Link speaker comments to user comments within a time threshold.")
    parser.add_argument("--transcript_dir", type=str, required=True, help="Directory containing transcript JSON files.")
    parser.add_argument("--comments_file", type=str, required=True, help="File path for user comments JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Output file for the combined data.")
    args = parser.parse_args()
    main(args.transcript_dir, args.comments_file, args.output_file)
