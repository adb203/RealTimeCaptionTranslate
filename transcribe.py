import argparse
import io
import os
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from tempfile import TemporaryFile
from time import sleep
from sys import platform
import argostranslate.package
import argostranslate.translate

def execute_transcription():
    parser = argparse.ArgumentParser(description="Real-Time Speech Transcription and Translation")
    parser.add_argument("--model_size", default="medium", help="Select the Whisper model size",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english_mode", action='store_true',
                        help="Activate for non-English transcription.")
    parser.add_argument("--audio_sensitivity", default=1000,
                        help="Mic sensitivity level for voice detection.", type=int)
    parser.add_argument("--recording_delay", default=2,
                        help="Delay in seconds for real-time recording.", type=float)
    parser.add_argument("--silence_duration", default=3,
                        help="Duration of silence to indicate end of speech segment.", type=float)  
    parser.add_argument("--target_language", default="ko", help="Target language for transcription", type=str)

    if 'linux' in platform:
        parser.add_argument("--linux_mic", default='pulse',
                            help="Preferred Linux microphone. Type 'list' to view options.", type=str)
    args = parser.parse_args()
    
    speech_language = args.target_language
    last_phrase_timestamp = None
    recent_audio_data = bytes()
    audio_queue = Queue()
    audio_listener = sr.Recognizer()
    audio_listener.energy_threshold = args.audio_sensitivity
    audio_listener.dynamic_energy_threshold = False

    translation_from = "ko"
    translation_to = "en"

    argostranslate.package.update_package_index()
    translation_packages = argostranslate.package.get_available_packages()
    package_to_download = next(
        (pkg for pkg in translation_packages 
         if pkg.from_code == translation_from and pkg.to_code == translation_to),
        None
    )
    argostranslate.package.install_from_path(package_to_download.download())

    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")   
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)
        
    whisper_model = args.model_size
    if whisper_model != "large" and not args.non_english_mode:
        whisper_model += ".en"
    voice_model = whisper.load_model(whisper_model)

    recording_gap = args.recording_delay
    silence_window = args.silence_duration

    temporary_file_path = TemporaryFile().name
    transcribed_content = ['']

    with source:
        audio_listener.adjust_for_ambient_noise(source)

    def audio_capture_callback(_, audio_chunk:sr.AudioData):
        audio_data = audio_chunk.get_raw_data()
        audio_queue.put(audio_data)

    audio_listener.listen_in_background(source, audio_capture_callback, phrase_time_limit=recording_gap)

    print("Start speaking for transcription.\n")

    try:
        while True:
            current_time = datetime.utcnow()
            if not audio_queue.empty():
                phrase_end = False
                if last_phrase_timestamp and current_time - last_phrase_timestamp > timedelta(seconds=silence_window):
                    recent_audio_data = bytes()
                    phrase_end = True
                last_phrase_timestamp = current_time

                while not audio_queue.empty():
                    audio_data = audio_queue.get()
                    recent_audio_data += audio_data

                audio_chunk = sr.AudioData(recent_audio_data, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_audio = io.BytesIO(audio_chunk.get_wav_data())

                with open(temporary_file_path, 'wb') as file:
                    file.write(wav_audio.read())

                whisper_result = voice_model.transcribe(temporary_file_path, fp16=torch.cuda.is_available(), language=speech_language)
                speech_transcription = whisper_result['text'].strip()
                translated_text = argostranslate.translate.translate(speech_transcription, translation_from, translation_to)

                transcript_entry = f"Original: {speech_transcription}\nTranslation: {translated_text}"
                if phrase_end:
                    transcribed_content.append(transcript_entry)
                else:
                    transcribed_content[-1] = transcript_entry

                with open('transcription_output.txt', 'a', encoding='utf-8') as transcript_file:
                    transcript_file.write(transcript_entry + "\n\n")

                os.system('cls' if os.name=='nt' else 'clear')
                print('\n'.join(transcribed_content))
                sleep(0.25)
    except KeyboardInterrupt:
        print("\n\nFinal Transcription:")
        print('\n'.join(transcribed_content))

if __name__ == "__main__":
    execute_transcription()
