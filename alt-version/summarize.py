import sys
import threading
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QLineEdit, QLabel
from PyQt5.QtCore import pyqtSignal, QObject, Qt
from PyQt5.QtGui import QFont, QIcon

from pytube import YouTube
from pydub import AudioSegment
import os
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BartForConditionalGeneration, BartTokenizer

class WorkerSignals(QObject):
    finished = pyqtSignal(str, str)
    error = pyqtSignal(str)

class Worker(QObject):
    def __init__(self, url):
        super().__init__()
        self.url = url
        self.signals = WorkerSignals()

    def process_video(self):
        try:
            summary, transcription = process_video(self.url)
            self.signals.finished.emit(summary, transcription)
        except Exception as e:
            self.signals.error.emit(str(e))

# Load the pipelines and models
questionans = pipeline("question-answering")
transcriber = pipeline(model="openai/whisper-large-v2")
model_gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer_gpt2 = AutoTokenizer.from_pretrained("gpt2")
summarizer = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer_summarizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

def download_youtube_video(url, output_path='video.mp4'):
    yt = YouTube(url)
    stream = yt.streams.filter(only_audio=True).first()
    stream.download(filename=output_path)
    return output_path

def convert_video_to_audio(video_path, audio_output_path):
    audio = AudioSegment.from_file(video_path)
    audio.export(audio_output_path, format="mp3")

def process_video(url):
    video_path = 'downloaded_video.mp4'
    audio_output_path = 'extracted_audio.mp3'
    
    downloaded_video_path = download_youtube_video(url, video_path)
    convert_video_to_audio(downloaded_video_path, audio_output_path)
    
    transcription = transcriber(audio_output_path)
    
    inject = "summarize the above text properly in around 300 words"
    prompt = transcription['text'] + inject
    input_ids = tokenizer_gpt2(prompt, return_tensors="pt").input_ids
    gen_tokens = model_gpt2.generate(
        input_ids,
        do_sample=True,
        temperature=0.9,
        max_length=500,
    )
    gen_text = tokenizer_gpt2.batch_decode(gen_tokens)[0]
    
    input_ids_summary = tokenizer_summarizer(gen_text, return_tensors='pt').input_ids
    summary_ids = summarizer.generate(input_ids_summary, max_length=100, min_length=50, length_penalty=2.0)
    summary = tokenizer_summarizer.decode(summary_ids[0], skip_special_tokens=True)
    
    if os.path.exists(video_path):
        os.remove(video_path)
    if os.path.exists(audio_output_path):
        os.remove(audio_output_path)
    
    return summary, transcription['text']

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YouTube Video Summarizer and Q&A")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #333;
            }
            QLineEdit {
                padding: 8px;
                font-size: 14px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            QPushButton {
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
                color: white;
                background-color: #4CAF50;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QTextEdit {
                font-size: 14px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()

        title_label = QLabel("YouTube Video Summarizer and Q&A")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)
        layout.addWidget(self.chat_area)

        url_layout = QHBoxLayout()
        url_label = QLabel("YouTube URL:")
        url_layout.addWidget(url_label)
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter YouTube URL")
        url_layout.addWidget(self.url_input)
        self.process_button = QPushButton("Process Video")
        self.process_button.clicked.connect(self.process_video)
        url_layout.addWidget(self.process_button)
        layout.addLayout(url_layout)

        question_layout = QHBoxLayout()
        question_label = QLabel("Question:")
        question_layout.addWidget(question_label)
        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText("Ask a question about the video")
        question_layout.addWidget(self.question_input)
        self.ask_button = QPushButton("Ask")
        self.ask_button.clicked.connect(self.ask_question)
        question_layout.addWidget(self.ask_button)
        layout.addLayout(question_layout)

        main_widget.setLayout(layout)

        self.transcription = ""

    def process_video(self):
        url = self.url_input.text()
        self.chat_area.append(f"<b>Processing URL:</b> {url}")
        self.process_button.setEnabled(False)
        
        self.worker = Worker(url)
        self.worker_thread = threading.Thread(target=self.worker.process_video)
        self.worker.signals.finished.connect(self.on_video_processed)
        self.worker.signals.error.connect(self.on_error)
        self.worker_thread.start()

    def on_video_processed(self, summary, transcription):
        self.chat_area.append(f"<b>Summary:</b> {summary}")
        self.transcription = transcription
        self.process_button.setEnabled(True)

    def on_error(self, error_message):
        self.chat_area.append(f"<b>Error:</b> {error_message}")
        self.process_button.setEnabled(True)

    def ask_question(self):
        question = self.question_input.text()
        self.chat_area.append(f"<b>Q:</b> {question}")
        answer = questionans(question=question, context=self.transcription)['answer']
        self.chat_area.append(f"<b>A:</b> {answer}")
        self.question_input.clear()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())