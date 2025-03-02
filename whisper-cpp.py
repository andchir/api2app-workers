import os
import sys
import subprocess
import ffmpeg

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.queue_manager import polling_queue, upload_queue_files, send_queue_error, send_queue_result_dict
from utils.upload_file import delete_old_files

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WHISPER_CLI_PATH = '/home/andrew/PycharmProjects/whisper.cpp/build/bin/whisper-cli'
WHISPER_MODEL_PATH = '/home/andrew/PycharmProjects/whisper.cpp/models/ggml-medium.bin'
MAX_DURATION = 30 * 60 # 30 minutes maximum


def processing(queue_item):
    print()
    print('New task:', queue_item['uuid'])
    upload_dir_path = os.path.join(ROOT_DIR, 'uploads', 'whisper')

    # Download all files to a local folder
    image_file_path, audio_file_path, video_file_path, image_file_path2 = upload_queue_files(queue_item, upload_dir_path)

    if not audio_file_path or not os.path.isfile(audio_file_path):
        print('Send error message - File not found.')
        send_queue_error(queue_item['uuid'], 'File not found.')
        return None

    # Options
    options = queue_item['data'] if 'data' in queue_item else dict()
    language = options['language'] if 'language' in options and type(options['language']) is str else 'en'
    output_type = options['output_type'] \
        if 'output_type' in options \
        and options['output_type'] in ['txt', 'csv', 'srt', 'json', 'json-full'] else 'txt'

    # Converting audio file to WAV format
    audio_wav_path = os.path.join(upload_dir_path, queue_item['uuid'] + '.wav')
    (
        ffmpeg
        .input(audio_file_path)
        .output(audio_wav_path, ar=16000, ac=1, acodec='pcm_s16le', t=MAX_DURATION)
        .run(capture_stdout=True, capture_stderr=True)
    )

    output_file_path = os.path.join(upload_dir_path, queue_item['uuid'])
    print()
    print('Processing...')
    # Creating a transcription using whisper-cli
    command = [
        WHISPER_CLI_PATH,
        '--model', WHISPER_MODEL_PATH,
        '--threads', '4',
        '--processors', '1',
        f'--output-{output_type}',
        '--language', language,
        '--duration', '0',
        '--file', audio_wav_path,
        '--output-file', output_file_path
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    if 'saving output to' in str(result):
        if output_type in ['json', 'json-full']:
            output_file_path += '.json'
        else:
            output_file_path += '.' + output_type
        if not os.path.isfile(output_file_path):
            print(f'Output file not found. Send error message - Processing error.')
            print()
            send_queue_error(queue_item['uuid'], 'Processing error. Please try again later.')
            return

        with open(output_file_path, 'r') as file:
            result_str = file.read()

        print('Sending the result...')
        res = send_queue_result_dict(queue_item['uuid'], {'result': result_str.strip()})
        print()
        print('Completed.')

    else:
        send_queue_error(queue_item['uuid'], 'Processing error. Please try again later.')

    # Delete old files
    deleted_input = delete_old_files(upload_dir_path, max_hours=2)
    print('Deleted old files: ', deleted_input)
    print()


if __name__ == '__main__':
    # Waiting for new tasks (polling)
    polling_queue('ea474c9a-c631-4bac-9141-ba26b0ff56c5', processing)
