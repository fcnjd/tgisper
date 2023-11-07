import logging
import os
import sys
from typing import BinaryIO

import ffmpeg
import numpy as np
import telebot
from faster_whisper import WhisperModel
from telebot.apihelper import ApiException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration from environment variables
bot_id = os.getenv("BOT_ID")
if not bot_id:
    logger.error("The BOT_ID environment variable not set.")
    sys.exit(1)

model_name = os.getenv("ASR_MODEL", "small")

# Initialize the bot and model
bot = telebot.TeleBot(bot_id)
model = WhisperModel(model_name, device="cpu", compute_type="float32")
SAMPLE_RATE = 16000


def main():
    # Main function to start the bot polling
    logger.info("Starting bot polling.")
    try:
        bot.infinity_polling(timeout=10, long_polling_timeout=5)
    except ApiException as e:
        logger.error(f"APIException occurred: {e}")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")

@bot.message_handler(commands=["help", "start"])
def send_welcome(message):
    """Send a welcome message in response to start/help commands."""
    bot.reply_to(
        message,
        "Hello! I'm a voice recognition bot 🎤 Send me a voice message and I'll transcribe it for you."
    )

@bot.message_handler(content_types=["voice","audio"], chat_types=["private", "group", "supergroup"])
def handle_audio_message(message):
    """Handle voice and audio messages by transcribing them."""
    file_info = bot.get_file(getattr(message.voice or message.audio, 'file_id'))
    file_path = file_info.file_path
    file_downloaded = bot.download_file(file_path)
    audio_data = load_audio(file_downloaded)
    transcribe_and_send(message, audio_data)

def load_audio(binary_file: BinaryIO, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Read an audio file object as mono waveform, resampling as necessary.

    :param binary_file: The audio file like object.
    :param sr: The sample rate to resample the audio if necessary.
    :return: A NumPy array containing the audio waveform.
    """
    try:
        out, _ = (ffmpeg
                  .input("pipe:0", format="s16le")
                  .output("pipe:1", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                  .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=binary_file))
    except ffmpeg.Error as e:
        logger.error(f"Failed to load audio: {e.stderr.decode().strip()}")
        raise RuntimeError("Failed to load audio") from e
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0

@bot.message_handler(content_types=['video', 'video_note'], chat_types=["private", "group", "supergroup"])
def handle_video_message(message):
    file_info = bot.get_file(getattr(message.video or message.video_note, 'file_id'))
    file_path = file_info.file_path
    file_downloaded = bot.download_file(file_path)
    audio_downloaded = convert_video_to_audio(file_downloaded)
    audio_data = load_audio(audio_downloaded)
    transcribe_and_send(message, audio_data)

def convert_video_to_audio(video_data: bytes) -> np.ndarray:
    try:
        # Convert video to audio using ffmpeg
        out, _ = (ffmpeg
                  .input('pipe:0', threads=0)
                  .output('pipe:1', format='s16le', acodec='pcm_s16le', ac=1, ar=SAMPLE_RATE)
                  .run(cmd='ffmpeg', capture_stdout=True, capture_stderr=True, input=video_data))
    except ffmpeg.Error as e:
        logger.error(f"Failed to convert video to audio: {e.stderr.decode().strip()}")
        raise RuntimeError("Failed to convert video to audio") from e
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0

def transcribe_and_send(message, audio_data: np.ndarray):
    try:
        segments, _ = model.transcribe(audio=audio_data, vad_filter=True, beam_size=1)
        text = "".join([segment.text for segment in segments])
        split_and_send_message(text, message)
    except Exception as e:
        handle_transcription_error(e, message)

# Check if the message is too long and split it into multiple messages if necessary
def split_and_send_message(text: str, message):
    split_length = 4096
    if len(text) <= split_length:
        bot.reply_to(message, text)
        return
    for start in range(0, len(text), split_length):
        bot.reply_to(message, text[start:start+split_length])

def handle_transcription_error(e: Exception, message):
    logger.error(f"Failed to process message: {e}")
    # For security and simplicity, only a generic message is sent.
    bot.reply_to(message, "Sorry, an error occurred while processing your message.")

if __name__ == "__main__":
    main()
