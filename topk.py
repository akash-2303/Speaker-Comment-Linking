import json
import spacy
import argparse
from datetime import datetime, timedelta
from scipy.spatial.distance import cosine
import numpy as np
import os
import pandas as pd
import sqlite3

# Load the spaCy model
nlp = spacy.load("en_core_web_md")

# Data loading function for JSON
def load_data_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Data loading function for CSV with error handling
def load_data_csv(file_path):
    chunks = []
    try:
        for chunk in pd.read_csv(file_path, encoding='utf-8', chunksize=1000, engine='python'):
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True)
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on failure

# Vectorizing transcript segments
def vectorize_transcript(transcript_segments):
    for segment in transcript_segments:
        segment_vector = nlp(segment['message']).vector
        if np.isnan(segment_vector).any():
            segment['vector'] = None
        else:
            segment['vector'] = segment_vector
        segment['start_time'] = datetime.strptime(segment['start'], '%H:%M:%S')
        segment['end_time'] = datetime.strptime(segment['end'], '%H:%M:%S')
    return transcript_segments

# Vectorizing comments and tracking skipped ones from a DataFrame
def vectorize_comments(df):
    df['vector'] = df['text'].apply(lambda x: nlp(str(x)).vector if pd.notnull(x) else None)
    df['vector'] = df['vector'].apply(lambda x: None if x is None or np.isnan(x).any() else x)
    df['published_at'] = pd.to_datetime(df['published_at'], utc=True)
    skipped_comments = df[df['vector'].isnull()]
    return df, skipped_comments

# Initialize SQLite database
def init_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS similarities
                 (comment_id INTEGER, transcript_id INTEGER, similarity REAL, PRIMARY KEY (comment_id, transcript_id))''')
    conn.commit()
    return conn

# Store similarity in the database
def store_similarity(conn, comment_id, transcript_id, similarity):
    c = conn.cursor()
    c.execute('INSERT OR REPLACE INTO similarities (comment_id, transcript_id, similarity) VALUES (?, ?, ?)',
              (comment_id, transcript_id, similarity))
    conn.commit()

# Retrieve similarity from the database
def get_similarity(conn, comment_id, transcript_id):
    c = conn.cursor()
    c.execute('SELECT similarity FROM similarities WHERE comment_id = ? AND transcript_id = ?', (comment_id, transcript_id))
    result = c.fetchone()
    return result[0] if result else None

# Compute valid cosine similarity
def valid_cosine(u, v):
    return 1 - cosine(u, v) if not np.isnan(u).any() and not np.isnan(v).any() else float('inf')

# Filtering based on relevance to any transcript segment
def filter_relevant_comments(comments, transcript_segments, conn, similarity_threshold=0.5):
    relevant_comments = []
    for index, comment in comments.iterrows():
        if comment['vector'] is None:
            continue

        is_relevant = False
        for seg_id, segment in enumerate(transcript_segments):
            if 'vector' not in segment or segment['vector'] is None:
                continue

            # Get precomputed similarity or compute and store it
            similarity = get_similarity(conn, index, seg_id)
            if similarity is None:
                similarity = valid_cosine(comment['vector'], segment['vector'])
                store_similarity(conn, index, seg_id, similarity)

            if 1 - similarity >= similarity_threshold:
                is_relevant = True
                break

        if is_relevant:
            relevant_comments.append(comment)
    return pd.DataFrame(relevant_comments)  # Ensure we return a DataFrame

# Create sliding windows of comment averages
def create_time_windows(comments, window_size=5, step=1):
    min_time = comments['published_at'].min()
    max_time = comments['published_at'].max()
    
    windows = []
    current_start = min_time
    
    while current_start <= max_time:
        current_end = current_start + timedelta(minutes=window_size)
        window_comments = comments.loc[
            (comments['published_at'] >= current_start) & (comments['published_at'] < current_end), 'vector'
        ]
        
        if not window_comments.empty:
            avg_vector = np.mean(list(window_comments), axis=0)
            if np.isnan(avg_vector).any():
                windows.append({
                    'start': current_start,
                    'end': current_end,
                    'avg_vector': None
                })
            else:
                windows.append({
                    'start': current_start,
                    'end': current_end,
                    'avg_vector': avg_vector
                })
        
        current_start += timedelta(minutes=step)
    
    return windows

# Link comments to the closest transcript segment or time window
def link_comments(transcript_segments, comments, windows, conn):
    linked_data = []
    
    for index, comment in comments.iterrows():
        if comment['vector'] is None:
            continue

        # Find the closest segment with a vector
        closest_segment = None
        min_cosine_distance = float('inf')
        for seg_id, segment in enumerate(transcript_segments):
            if 'vector' not in segment or segment['vector'] is None:
                continue

            # Get precomputed similarity or compute and store it
            similarity = get_similarity(conn, index, seg_id)
            if similarity is None:
                similarity = valid_cosine(comment['vector'], segment['vector'])
                store_similarity(conn, index, seg_id, similarity)

            cosine_distance = 1 - similarity
            if cosine_distance < min_cosine_distance:
                min_cosine_distance = cosine_distance
                closest_segment = segment

        # Find the closest time window
        closest_window = None
        min_window_distance = float('inf')
        for win in windows:
            if win['avg_vector'] is None:
                continue

            window_distance = valid_cosine(win['avg_vector'], comment['vector'])
            if window_distance < min_window_distance:
                min_window_distance = window_distance
                closest_window = win

        # Choose the closest item
        if closest_segment and (not closest_window or min_cosine_distance <= min_window_distance):
            closest_item = closest_segment
            item_type = 'transcript'
        else:
            closest_item = closest_window
            item_type = 'window'

        if closest_item:
            linked_data.append({
                'comment_text': comment['text'],
                'comment_time': comment['published_at'].isoformat(),
                'linked_item': {
                    'type': item_type,
                    'start': closest_item['start_time'].strftime('%H:%M:%S') if item_type == 'transcript' else closest_item['start'].isoformat(),
                    'end': closest_item['end_time'].strftime('%H:%M:%S') if item_type == 'transcript' else closest_item['end'].isoformat(),
                    'message': closest_item.get('message', 'Time window discussion') if item_type == 'transcript' else 'Time window discussion'
                }
            })
    
    return linked_data

# Main function to process multiple transcripts and comments
def main(transcript_dir, comments_file, output_file, db_file, filter_relevant, similarity_threshold):
    conn = init_db(db_file)

    # Load and vectorize comments from CSV file
    comments_df = load_data_csv(comments_file)
    if comments_df.empty:
        print("Failed to load comments.")
        return
    
    comments, skipped_comments = vectorize_comments(comments_df)
    all_debates_data = []

    # Loop through each transcript file in the transcript directory
    for filename in os.listdir(transcript_dir):
        if filename.endswith(".json"):
            transcript_path = os.path.join(transcript_dir, filename)
            print(f"Processing transcript: {transcript_path}")
            transcript_segments = load_data_json(transcript_path)
            transcript_segments = vectorize_transcript(transcript_segments)

            # Filter relevant comments if enabled
            if filter_relevant:
                relevant_comments = filter_relevant_comments(comments, transcript_segments, conn, similarity_threshold)
            else:
                relevant_comments = comments

            windows = create_time_windows(relevant_comments)
            linked_data = link_comments(transcript_segments, relevant_comments, windows, conn)
            all_debates_data.extend(linked_data)

    # Save all linked data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_debates_data, f, indent=4)
    
    conn.close()
    print(f"Linked data saved to {output_file}")
    print(f"Skipped comments: {len(skipped_comments)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Link comments to the closest transcript segment or time window.")
    parser.add_argument("--transcript_dir", type=str, required=True, help="Path to the transcript directory.")
    parser.add_argument("--comments_file", type=str, required=True, help="Path to the comments CSV file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the linked output JSON file.")
    parser.add_argument("--db_file", type=str, required=True, help="Path to the SQLite database file.")
    parser.add_argument("--filter_relevant", action="store_true", help="Enable filtering to retain only relevant comments.")
    parser.add_argument("--similarity_threshold", type=float, default=0.5, help="Threshold for relevance filtering (0 to 1).")
    
    args = parser.parse_args()
    main(args.transcript_dir, args.comments_file, args.output_file, args.db_file, args.filter_relevant, args.similarity_threshold)
