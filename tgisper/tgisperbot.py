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
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.getLogger("faster_whisper").setLevel(logging.DEBUG)

# Constants
SAMPLE_RATE = 16000
MAX_MESSAGE_LENGTH = 4096

# Load configuration from environment variables
bot_id = os.getenv("BOT_ID")
if not bot_id:
    logger.error("The BOT_ID environment variable not set.")
    sys.exit(1)

model_name = os.getenv("ASR_MODEL", "small")
compute_type = os.getenv("COMPUTE_TYPE", "INT8")

# Initialize the bot and model
bot = telebot.TeleBot(bot_id)
model = WhisperModel(model_name, device="cpu", compute_type=compute_type)


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

def load_audio(binary_file: BinaryIO, sr: int = SAMPLE_RATE):
    """
    Read an audio file object as mono waveform, resampling as necessary.
    Modified from https://github.com/openai/whisper/blob/main/whisper/audio.py
    to accept a binary object.


    :param binary_file: The audio file like object.
    :param sr: The sample rate to resample the audio if necessary.
    :return: A NumPy array containing the audio waveform.
    """
    try:
        # This launches a subprocess to decode audio while down-mixing and
        # resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input("pipe:", threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=binary_file)
        )
    except ffmpeg.Error as e:
        logger.error(f"Failed to load audio: {e.stderr.decode().strip()}")
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

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
        # Transcribe and request word timestamps
        segments, _ = model.transcribe(
            audio=audio_data,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=1000),
            beam_size=5,
            word_timestamps=True)
        # Initialize an empty message to build our transcription
        full_transcript = ""
        last_message = None
        # Process segments and append words to the full transcription
        for segment in segments:
            for word in segment.words:
                # Build the string for the current word with timestamps
                word_str = "[{:.2f}s -> {:.2f}s] {}".format(word.start, word.end, word.word)
                # Check if adding the next word exceeds the Telegram message limit
                if len(full_transcript) + len(word_str) < MAX_MESSAGE_LENGTH:
                    full_transcript += word_str + " "
                else:
                    # If limit reached, send a new message
                    last_message = bot.reply_to(message, text=full_transcript)
                    full_transcript = word_str + " " # Start a new message with the current word
        # If the transcript contains data, send or edit the last message.
        if full_transcript:
            if last_message:
                bot.edit_message_text(chat_id=last_message.chat.id,
                                      message_id=last_message.message_id,
                                      text=full_transcript)
            else:
                bot.reply_to(message, text=full_transcript)
    except Exception as e:
        handle_transcription_error(e, message)

def handle_transcription_error(e: Exception, message):
    logger.error(f"Failed to process message: {e}")
    # For security and simplicity, only a generic message is sent.
    bot.reply_to(message, "Sorry, an error occurred while processing your message.")

if __name__ == "__main__":
    main()
