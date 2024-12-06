import json
import os
from pyannote.audio import Pipeline
from dotenv import load_dotenv
import datetime

load_dotenv()

def get_diarization(audio_path):
    """
    Perform speaker diarization on the audio file.
    """
    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    if hf_token:
        print("Token loaded successfully.")
    else:
        print("Token not loaded. Check your .env file.")

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    diarization = pipeline(audio_path, num_speakers=5)
    
    speaker_segments = []
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append({
            'start': turn.start,
            'end': turn.end,
            'speaker': speaker
        })
    
    return speaker_segments

def match_transcript_to_speakers(transcript_list, speaker_segments):
    """
    Match the transcript text with speaker labels based on timestamps.
    Concatenate messages from the same speaker.
    """
    combined_data = []
    current_speaker = None
    current_message = ""
    current_start = None
    
    for transcript in transcript_list:
        transcript_time = transcript['start']
        text = transcript['text']
        
        # Find the closest speaker segment
        for segment in speaker_segments:
            if segment['start'] <= transcript_time <= segment['end']:
                speaker = segment['speaker']
                if speaker == current_speaker:
                    # Same speaker, concatenate the message and timestamps
                    current_message += " " + text
                else:
                    # Different speaker or first segment, save the previous speaker's data
                    if current_speaker is not None:
                        combined_data.append({
                            'speaker': current_speaker,
                            'start': str(datetime.timedelta(seconds=int(current_start))),
                            'end': str(datetime.timedelta(seconds=int(transcript_time))),
                            'message': current_message
                        })
                    # Start a new segment for the new speaker
                    current_speaker = speaker
                    current_message = text
                    current_start = transcript_time
                break
    
    # Save the last speaker's data
    if current_speaker is not None:
        combined_data.append({
            'speaker': current_speaker,
            'start': str(datetime.timedelta(seconds=int(current_start))),
            'end': str(datetime.timedelta(seconds=int(transcript_list[-1]['start']))),
            'message': current_message
        })
    
    return combined_data

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    audio_path = os.path.join(current_dir, "HarrisVSTrump.wav")
    # Step 1: Load transcript from the output JSON file created by transcript.py
    # transcript_file = "C:\\Users\\Akash Balaji\\Downloads\\transcript\\BidesVSTrump.json"  # replace with your actual file
    transcript_file = os.path.join(current_dir, "HarrisVsTrump.json")
    with open(transcript_file, "r", encoding="utf-8") as f:
        transcript_list = json.load(f)
    
    # Step 2: Perform speaker diarization
    speaker_segments = get_diarization(audio_path)
    
    # Step 3: Match transcript to speaker segments and concatenate
    combined_data = match_transcript_to_speakers(transcript_list, speaker_segments)
    
    # Step 4: Save the combined transcript with speakers to a JSON file
    output_filename = "combined_transcript.json"
    with open(output_filename, "w", encoding="utf-8") as json_file:
        json.dump(combined_data, json_file, indent=4)
    
    print(f"Combined transcript with speaker info saved to {output_filename}")
