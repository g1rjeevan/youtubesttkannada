import torch
import yt_dlp
import subprocess
import os
import json

import audiotranslate
from datetime import datetime
from transformers import pipeline

def extract_audio_from_stream_old(url, output_dir="downloads"):
    """
    Extract audio from a YouTube live stream or video.
    
    Args:
        url (str): YouTube video/stream URL
        output_dir (str): Directory to save the audio file
    
    Returns:
        str: Path to the saved audio file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"audio_stream_{timestamp}.mp3"
    output_path = os.path.join(output_dir, output_filename)
    
    # Configure yt-dlp options
    ydl_opts = {
        'format': 'mp3/192',  # Select best audio quality
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',  # Audio quality in kbps
        }],
        'outtmpl': output_path,  # Output template
        'verbose': True,  # Show detailed progress
    }
    
    try:
        # Download and extract audio
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Starting audio extraction from: {url}")
            ydl.download([url])
            
        print(f"Audio successfully saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        return None

def get_available_formats(url):
    """
    Get available formats for the video URL
    """
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            formats = info.get('formats', [])
            
            with open('data.json', 'w', encoding='utf-8') as f:
                json.dump(formats, f, ensure_ascii=False, indent=4)
            print("audio_formats::", json.dumps(formats))

            # Filter audio-only formats
            audio_formats = [f for f in formats if f.get('vcodec') == 'none']
            
            if not audio_formats:
                # If no audio-only formats, get formats with audio
                audio_formats = [f for f in formats if f.get('acodec') != 'none']
            
            return audio_formats
    except Exception as e:
        print(f"Error getting formats: {str(e)}")
        return []

def extract_audio_from_stream(url, output_dir="downloads"):
    """
    Extract audio from a YouTube live stream or video.
    
    Args:
        url (str): YouTube video/stream URL
        output_dir (str): Directory to save the audio file
    
    Returns:
        str: Path to the saved audio file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get available formats
    audio_formats = get_available_formats(url)
    
    if not audio_formats:
        print("No suitable audio formats found!")
        return None
    
    # Sort formats by quality (bitrate)
    audio_formats = sorted(
        audio_formats,
        key=lambda x: float(x.get('abr', 0) or 0),
        reverse=True
    )
    # audio_formats.sort(key=lambda x: float(x.get('abr', 0) or 0, reverse=True)
    best_audio = audio_formats[0]
    format_id = best_audio['format_id']
    
    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"audio_stream_{timestamp}"
    output_path = os.path.join(output_dir, output_filename)
    
    # Configure yt-dlp options
    ydl_opts = {
        'format': format_id,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_path,
        'verbose': True,
    }

    
        # Try to find FFmpeg automatically
    ffmpeg_path = "/usr/local/bin/ffmpeg"
    
        # Add FFmpeg location if provided
    if ffmpeg_path:
        ydl_opts['ffmpeg_location'] = ffmpeg_path
    
    try:
        # Download and extract audio
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Starting audio extraction from: {url}")
            print(f"Using format: {best_audio.get('format', 'unknown')} "
                  f"(bitrate: {best_audio.get('abr', 'unknown')}kbps)")
            ydl.download([url])
            
        print(f"Audio successfully saved to: {output_path}")
        return output_path+".mp3"
        
    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        return None

def list_available_formats(url):
    """
    Print all available formats for debugging
    """
    formats = get_available_formats(url)
    print("\nAvailable audio formats:")
    for f in formats:
        print(f"Format ID: {f.get('format_id', 'N/A')}")
        print(f"Format: {f.get('format', 'N/A')}")
        print(f"Audio Bitrate: {f.get('abr', 'N/A')}kbps")
        print("-" * 50)

def find_ffmpeg():
    """Try to find FFmpeg installation"""
    possible_paths = [
        "C:\\ffmpeg\\bin\\ffmpeg.exe"
    ]
    
    for path in possible_paths:
        if os.path.isfile(path):
            return path
            
    return None

def main():
    url = input("Enter YouTube URL: ")
    output_dir = input("Enter output directory (press Enter for default 'downloads'): ").strip()
    
    if not output_dir:
        output_dir = "downloads"
    
    # Optionally list formats for debugging
    list_formats = input("Would you like to see available formats? (y/n): ").lower().strip() == 'y'
    if list_formats:
        list_available_formats(url)

    result = extract_audio_from_stream(url, output_dir)
    
    if result:
        # Check file size to determine which method to use
        # file_size = os.path.getsize(result)
        #
        # if file_size > 10 * 1024 * 1024:  # If file is larger than 10MB
        #     print("Large audio file detected, using chunked processing...")
        #     text = audiotranslate.transcribe_long_kannada_audio(result)
        # else:
        #     text = audiotranslate.transcribe_kannada_audio(result)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        transcribe = pipeline(task="automatic-speech-recognition", model="vasista22/whisper-kannada-medium",
                              chunk_length_s=30, device=device)
        transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language="kn",
                                                                                                 task="transcribe")

        text = transcribe(result)["text"]
        if text:
            print("\nTranscribed text:")
            print(text)
        else:
            print("\nTranscription failed!")
        print(f"\nExtraction completed successfully!")
    else:
        print("\nExtraction failed!")

if __name__ == "__main__":
    main()