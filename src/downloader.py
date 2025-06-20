from yt_dlp import YoutubeDL
import os
import re

class YouTubeAudioDownloader:
    def __init__(self, output_folder):
        self.output_folder = os.path.abspath(output_folder)
        self.audio_files_dict = {}

    def get_safe_filename(self, filename):
        """Sanitize a filename to ensure it is safe for the filesystem."""
        safe_filename = re.sub(r'[^\w\-.]', '_', filename)
        safe_filename = re.sub(r'_+', '_', safe_filename)
        safe_filename = safe_filename[:50].strip('_')
        return safe_filename

    # def download_audio(self, video_url):
    #     """Download audio from a YouTube video."""
    #     try:
    #         # ydl_opts = {
    #         #     'format': 'bestaudio/best',
    #         #     'postprocessors': [{
    #         #         'key': 'FFmpegExtractAudio',
    #         #         'preferredcodec': 'mp3',
    #         #         'preferredquality': '192',
    #         #     }],
    #         #     'outtmpl': os.path.join(self.output_folder, '%(title)s.%(ext)s'),
    #         # }
    #         ydl_opts = {
    #             'format': 'bestaudio/best',
    #             'postprocessors': [{
    #                 'key': 'FFmpegExtractAudio',
    #                 'preferredcodec': 'mp3',
    #                 'preferredquality': '192',
    #             }],
    #             'ffmpeg_location': "/path/to/ffmpeg",  # Replace with your actual ffmpeg path
    #             'outtmpl': os.path.join(self.output_folder, '%(title)s.%(ext)s'),
    #         }


    #         with YoutubeDL(ydl_opts) as ydl:
    #             info = ydl.extract_info(video_url, download=True)
    #             filename = ydl.prepare_filename(info)
    #             base, ext = os.path.splitext(filename)
    #             new_file = base + '.mp3'

    #             print(f"Downloaded: {new_file}")
    #             self.audio_files_dict[video_url] = new_file
    #             return new_file
    #     except Exception as e:
    #         print(f"Error downloading {video_url}: {str(e)}")
    #         return None



    def download_audio(self, video_url):
        """Download audio from a YouTube video, skipping if already downloaded."""
        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': os.path.join(self.output_folder, '%(title)s.%(ext)s'),
            }

            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)  # Fetch metadata without downloading
                filename = ydl.prepare_filename(info).replace(".webm", ".mp3").replace(".m4a", ".mp3")  # Adjust for mp3 format

                full_path = os.path.join(self.output_folder, filename)

                if os.path.exists(full_path):
                    print(f"Skipping {filename}, already downloaded.")
                    self.audio_files_dict[video_url] = full_path
                    return full_path

                # Proceed with download
                ydl.download([video_url])
                print(f"Downloaded: {full_path}")
                self.audio_files_dict[video_url] = full_path
                return full_path
        except Exception as e:
            print(f"Error downloading {video_url}: {str(e)}")
            return None

    def download_multiple_audios(self, video_urls):
        """Download multiple videos' audio."""
        for url in video_urls:
            print(f"Processing: {url}")
            audio_file = self.download_audio(url)
            if audio_file is None:
                print(f"Failed to download {url}")
        return self.audio_files_dict
