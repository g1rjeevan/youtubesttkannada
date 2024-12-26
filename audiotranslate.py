import speech_recognition as sr
from pydub import AudioSegment
import os

# Set ffmpeg paths - modify these paths according to your installation
print("os.name::",os.name)
if os.name == 'nt':  # Windows
    AudioSegment.converter = r"/mnt/c/ffmpeg/bin/ffmpeg.exe"
    AudioSegment.ffmpeg = r"/mnt/c/ffmpeg/bin/ffmpeg.exe"
    AudioSegment.ffprobe = r"/mnt/c/ffmpeg/bin/ffprobe.exe"
else:  # Linux/Unix
    AudioSegment.converter = "/usr/bin/ffmpeg"
    AudioSegment.ffmpeg = "/usr/bin/ffmpeg"
    AudioSegment.ffprobe = "/usr/bin/ffprobe"

def convert_to_wav(input_path, output_path):
    """Convert audio file to WAV format"""
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1)  # Convert to mono
        audio = audio.set_frame_rate(16000)  # Set sample rate to 16kHz
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        print(f"Error converting audio: {e}")
        return False

def transcribe_kannada_audio(audio_file_path):
    """
    Transcribe Kannada audio using speech recognition
    
    Args:
        audio_file_path (str): Path to the audio file
    Returns:
        str: Transcribed Kannada text
    """
    # Initialize recognizer
    recognizer = sr.Recognizer()
    
    try:
        # Check if file needs conversion
        if not audio_file_path.lower().endswith('.wav'):
            print("Converting audio to WAV format...")
            wav_path = audio_file_path.rsplit('.', 1)[0] + '.wav'
            if not convert_to_wav(audio_file_path, wav_path):
                return None
            audio_file_path = wav_path
        print("audio_file_path::",audio_file_path)
        with sr.AudioFile(audio_file_path) as audio_file:
            # Adjust for ambient noise
            print("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(audio_file, duration=0.5)
            
            # Record the audio file
            print("Processing audio...")
            audio = recognizer.record(audio_file)
            
            # Perform the transcription with Kannada language
            print("Transcribing...")
            text = recognizer.recognize_google(
                audio,
                language="kn-IN",  # Kannada language code
                show_all=False
            )
            
            return text
            
    except sr.UnknownValueError:
        print("Speech Recognition could not understand the audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Speech Recognition service; {e}")
        return None
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def transcribe_long_kannada_audio(audio_file_path, chunk_duration=30):
    """
    Transcribe long Kannada audio files by splitting into chunks
    
    Args:
        audio_file_path (str): Path to the audio file
        chunk_duration (int): Duration of each chunk in seconds
    Returns:
        str: Transcribed Kannada text
    """
    recognizer = sr.Recognizer()
    full_text = []
    
    try:
        with sr.AudioFile(audio_file_path) as audio_file:
            # Get total duration
            audio_duration = audio_file.duration_seconds
            
            # Process each chunk
            for i in range(0, int(audio_duration), chunk_duration):
                print(f"Processing chunk {i//chunk_duration + 1}...")
                audio_file.seek(i)
                chunk = recognizer.record(
                    audio_file, 
                    duration=min(chunk_duration, audio_duration - i)
                )
                
                try:
                    chunk_text = recognizer.recognize_google(
                        chunk,
                        language="kn-IN",
                        show_all=False
                    )
                    full_text.append(chunk_text)
                except sr.UnknownValueError:
                    full_text.append("[ಅಸ್ಪಷ್ಟ]")  # [unclear] in Kannada
                except sr.RequestError as e:
                    print(f"API error in chunk {i//chunk_duration + 1}: {e}")
                    full_text.append("[ದೋಷ]")  # [error] in Kannada
                    
        return " ".join(full_text)
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def main():
    # Example usage
    audio_path = input("Enter the path to your audio file: ")
    
    if not os.path.exists(audio_path):
        print("Error: Audio file not found!")
        return
    
    # Check file size to determine which method to use
    file_size = os.path.getsize(audio_path)
    
    if file_size > 10 * 1024 * 1024:  # If file is larger than 10MB
        print("Large audio file detected, using chunked processing...")
        text = transcribe_long_kannada_audio(audio_path)
    else:
        text = transcribe_kannada_audio(audio_path)
    
    if text:
        print("\nTranscribed text:")
        print(text)
    else:
        print("\nTranscription failed!")

if __name__ == "__main__":
    main()