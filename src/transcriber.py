import os
import torch
import whisper

class AudioTranscriber:
    def __init__(self, input_folder):
        """Initialize the transcriber with the specified input folder."""
        self.input_folder = os.path.abspath(input_folder)
        self.whisper_model = None  # Model must be loaded externally
        self.transcriptions_dict = {}

    def transcribe_audio(self, audio_file):
        """Transcribe a single audio file using the Whisper model."""
        try:
            if not os.path.exists(audio_file):
                print(f"File not found: {audio_file}")
                return None

            if os.path.getsize(audio_file) == 0:
                print(f"Empty file: {audio_file}")
                return None

            transcription = self.whisper_model.transcribe(audio_file)
            return transcription["text"]

        except Exception as e:
            print(f"Error transcribing {audio_file}: {str(e)}")
            return None

    # def transcribe_all_audios(self, audio_files_dict):
    #     """Transcribe all downloaded audio files."""
    #     for url, audio_path in audio_files_dict.items():
    #         if not audio_path.endswith(".mp3"):
    #             print(f"Skipping non-MP3 file: {audio_path}")
    #             continue

    #         transcription = self.transcribe_audio(audio_path)

    #         if transcription:
    #             self.transcriptions_dict[url] = {
    #                 "url": url,
    #                 "audio_path": audio_path,
    #                 "transcription": transcription
    #             }
    #         else:
    #             print(f"Failed to transcribe: {audio_path}")

    #     return self.transcriptions_dict

    def transcribe_all_audios(self, audio_files_dict):
        """Transcribe all downloaded audio files and return structured data."""
        audio_data = []  # Store formatted results

        for url, audio_path in audio_files_dict.items():
            if not audio_path.endswith(".mp3"):
                print(f"Skipping non-MP3 file: {audio_path}")
                continue

            transcription = self.transcribe_audio(audio_path)

            if transcription:
                audio_data.append({
                    "url": url,
                    "audio_path": audio_path,
                    "transcription": transcription
                })
            else:
                print(f"Failed to transcribe: {audio_path}")

        return audio_data
