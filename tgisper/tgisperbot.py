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
    """Handle voice messages by transcribing them."""
    try:
        voice_meta = bot.get_file(message.voice.file_id)
        voice_audio = load_audio(bot.download_file(voice_meta.file_path))
        segments, _ = model.transcribe(audio=voice_audio, vad_filter=True, beam_size=1)
        text = "".join([segment.text for segment in segments])
        split_length = 4096
        for start in range(0, len(text), split_length):
            bot.send_message(message.chat.id, text[start:start + split_length])
    except Exception as e:
        logger.error(f"Failed to process voice message: {e}")
        bot.reply_to(message, "Sorry, an error occurred while processing your voice message.")


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

if __name__ == "__main__":
    main()
