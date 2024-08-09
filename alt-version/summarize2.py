import sys
import threading
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QLineEdit, QLabel, QFileDialog, QTabWidget
from PyQt5.QtCore import pyqtSignal, QObject, Qt

from pytube import YouTube
from pydub import AudioSegment
import PyPDF2
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BartForConditionalGeneration, BartTokenizer

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

def process_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        content = ""
        for page in pdf_reader.pages:
            content += page.extract_text() + "\n"
    
    inject = "summarize the above text properly in around 300 words"
    prompt = content + inject
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
    
    return summary, content

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Content Summarizer and Q&A")
        self.setGeometry(100, 100, 800, 600)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()

        self.tab_widget = QTabWidget()
        self.youtube_tab = QWidget()
        self.pdf_tab = QWidget()
        self.tab_widget.addTab(self.youtube_tab, "YouTube")
        self.tab_widget.addTab(self.pdf_tab, "PDF")
        layout.addWidget(self.tab_widget)

        self.setup_youtube_tab()
        self.setup_pdf_tab()

        main_widget.setLayout(layout)

        self.content = ""

    def setup_youtube_tab(self):
        layout = QVBoxLayout()

        self.yt_chat_area = QTextEdit()
        self.yt_chat_area.setReadOnly(True)
        layout.addWidget(self.yt_chat_area)

        url_layout = QHBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter YouTube URL")
        url_layout.addWidget(self.url_input)
        self.process_button = QPushButton("Process Video")
        self.process_button.clicked.connect(self.process_video)
        url_layout.addWidget(self.process_button)
        layout.addLayout(url_layout)

        question_layout = QHBoxLayout()
        self.yt_question_input = QLineEdit()
        self.yt_question_input.setPlaceholderText("Ask a question about the video")
        question_layout.addWidget(self.yt_question_input)
        self.yt_ask_button = QPushButton("Ask")
        self.yt_ask_button.clicked.connect(lambda: self.ask_question(self.yt_question_input, self.yt_chat_area))
        question_layout.addWidget(self.yt_ask_button)
        layout.addLayout(question_layout)

        self.youtube_tab.setLayout(layout)

    def setup_pdf_tab(self):
        layout = QVBoxLayout()

        self.pdf_chat_area = QTextEdit()
        self.pdf_chat_area.setReadOnly(True)
        layout.addWidget(self.pdf_chat_area)

        pdf_layout = QHBoxLayout()
        self.pdf_path_input = QLineEdit()
        self.pdf_path_input.setPlaceholderText("Select PDF file")
        pdf_layout.addWidget(self.pdf_path_input)
        self.pdf_browse_button = QPushButton("Browse")
        self.pdf_browse_button.clicked.connect(self.browse_pdf)
        pdf_layout.addWidget(self.pdf_browse_button)
        self.pdf_process_button = QPushButton("Process PDF")
        self.pdf_process_button.clicked.connect(self.process_pdf)
        pdf_layout.addWidget(self.pdf_process_button)
        layout.addLayout(pdf_layout)

        question_layout = QHBoxLayout()
        self.pdf_question_input = QLineEdit()
        self.pdf_question_input.setPlaceholderText("Ask a question about the PDF")
        question_layout.addWidget(self.pdf_question_input)
        self.pdf_ask_button = QPushButton("Ask")
        self.pdf_ask_button.clicked.connect(lambda: self.ask_question(self.pdf_question_input, self.pdf_chat_area))
        question_layout.addWidget(self.pdf_ask_button)
        layout.addLayout(question_layout)

        self.pdf_tab.setLayout(layout)

    def browse_pdf(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select PDF File", "", "PDF Files (*.pdf)")
        if file_path:
            self.pdf_path_input.setText(file_path)

    def process_video(self):
        url = self.url_input.text()
        self.yt_chat_area.append(f"Processing URL: {url}")
        self.process_button.setEnabled(False)
        
        try:
            summary, transcription = process_video(url)
            self.yt_chat_area.append(f"Summary: {summary}")
            self.content = transcription
        except Exception as e:
            self.yt_chat_area.append(f"Error: {str(e)}")
        finally:
            self.process_button.setEnabled(True)

    def process_pdf(self):
        pdf_path = self.pdf_path_input.text()
        self.pdf_chat_area.append(f"Processing PDF: {pdf_path}")
        self.pdf_process_button.setEnabled(False)
        
        try:
            summary, content = process_pdf(pdf_path)
            self.pdf_chat_area.append(f"Summary: {summary}")
            self.content = content
        except Exception as e:
            self.pdf_chat_area.append(f"Error: {str(e)}")
        finally:
            self.pdf_process_button.setEnabled(True)

    def ask_question(self, question_input, chat_area):
        question = question_input.text()
        chat_area.append(f"Q: {question}")
        answer = questionans(question=question, context=self.content)['answer']
        chat_area.append(f"A: {answer}")
        question_input.clear()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    