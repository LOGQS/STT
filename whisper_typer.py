import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import json
import threading
import queue
import sounddevice as sd
import numpy as np
import customtkinter as ctk
from faster_whisper import WhisperModel
from typing import Optional
import pyautogui
from pynput import keyboard as kb
from pynput.keyboard import Key
import warnings
from collections import deque
import time


warnings.filterwarnings("ignore", category=UserWarning)

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.001

class AudioBuffer:
    def __init__(self, max_size=3):
        self.buffer = deque(maxlen=max_size)
        self.silence_threshold = 0.001
        self.noise_threshold = 0.02
        self.silence_duration = 1
        self.last_audio_level = 0
        self.silence_start = None
        self.accumulated_chunks = []
        self.is_recording_speech = False

    def add_audio(self, audio_data):
        self.buffer.append(audio_data)
        if self.is_recording_speech:
            self.accumulated_chunks.append(audio_data)

    def get_audio(self):
        if not self.accumulated_chunks:
            return None
        return np.concatenate(self.accumulated_chunks)

    def clear(self):
        self.buffer.clear()
        self.accumulated_chunks = []
        self.silence_start = None
        self.is_recording_speech = False

    def is_silence(self, audio_data):
        rms = np.sqrt(np.mean(np.square(audio_data)))
        # Return RMS for GUI updates
        # The actual processing decision is handled externally
        is_speech = rms > self.silence_threshold
        is_noise = self.noise_threshold < rms <= self.silence_threshold
        is_silent = rms <= self.noise_threshold

        print(f"[DEBUG] RMS: {rms:.6f}, Is Speech: {is_speech}, Is Noise: {is_noise}, Is Silent: {is_silent}")

        if is_speech and not self.is_recording_speech:
            self.is_recording_speech = True
            self.silence_start = None
            print("[DEBUG] Speech started")

        if not is_speech and self.is_recording_speech:
            if self.silence_start is None:
                self.silence_start = time.time()
                print("[DEBUG] Potential silence started")
        elif is_speech:
            self.silence_start = None

        should_process = False

        if self.is_recording_speech and self.silence_start is not None:
            silence_time = time.time() - self.silence_start
            if silence_time > self.silence_duration:
                should_process = True
                print(f"[DEBUG] Processing after {silence_time:.2f}s of silence")


        if should_process:
            self.is_recording_speech = False

        return rms, should_process

class AudioRecorder(threading.Thread):
    def __init__(self, text_queue: queue.Queue, status_queue: queue.Queue, audio_level_queue: queue.Queue, settings: dict):
        super().__init__()
        self.text_queue = text_queue
        self.status_queue = status_queue
        self.audio_level_queue = audio_level_queue
        self.settings = settings
        self.model = None
        self.daemon = True
        self.audio_buffer = AudioBuffer(max_size=3)
        self.processing = False
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    def load_model(self):
        try:
            if self.model is None:
                self.status_queue.put(("Loading Whisper model...", "yellow"))
                self.model = WhisperModel(
                    model_size_or_path="base",
                    device="cpu",
                    compute_type="int8",
                    num_workers=4
                )
                self.status_queue.put(("Status: Ready", "green"))
        except Exception as e:
            self.status_queue.put((f"Error loading model: {str(e)}", "red"))
            print(f"Model loading error details: {str(e)}")
            self.stop_event.set()

    def run(self):
        print("[DEBUG] Starting AudioRecorder thread")
        self.load_model()
        if self.model is None:
            print("[DEBUG] Model failed to load, exiting thread")
            return
        
        sample_rate = 16000
        block_size = 1024  # Smaller block size for better responsiveness

        def callback(indata, frames, time_info, status):
            if self.stop_event.is_set():
                raise sd.CallbackStop()
            audio_data = indata[:, 0].copy()
            self.audio_buffer.add_audio(audio_data)
            rms, should_process = self.audio_buffer.is_silence(audio_data)
            # Send RMS to GUI
            self.audio_level_queue.put(rms)
            if should_process and not self.processing:
                self.processing = True
                self.status_queue.put(("Processing speech...", "blue"))
                audio_data_combined = self.audio_buffer.get_audio()
                if audio_data_combined is not None:
                    print(f"[DEBUG] Audio data shape: {audio_data_combined.shape}")
                    print("[DEBUG] Transcribing audio...")
                    try:
                        segments, _ = self.model.transcribe(
                            audio_data_combined,
                            language=self.settings["language"],
                            beam_size=5,
                            vad_filter=True,
                            vad_parameters=dict(min_silence_duration_ms=500)
                        )
                        
                        text = " ".join([segment.text for segment in segments]).strip()
                        print(f"[DEBUG] Transcribed text: {text}")
                        if text:
                            print("[DEBUG] Sending text to queue")
                            self.text_queue.put(text)
                            self.status_queue.put(("Status: Recording", "green"))
                        else:
                            print("[DEBUG] No text transcribed")
                    except Exception as e:
                        print(f"[DEBUG] Transcription error: {str(e)}")
                        self.status_queue.put((f"Transcription error: {str(e)}", "red"))
                else:
                    print("[DEBUG] No audio data to process")
                
                self.audio_buffer.clear()
                self.processing = False

        try:
            with sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype='float32',
                blocksize=block_size,
                callback=callback
            ):
                print("[DEBUG] InputStream started")
                while not self.stop_event.is_set():
                    sd.sleep(100)  # Sleep briefly to reduce CPU usage
        except Exception as e:
            print(f"[DEBUG] Error in recording stream: {str(e)}")
            self.status_queue.put((f"Recording error: {str(e)}", "red"))

    def stop(self):
        """Signal the thread to stop"""
        self.stop_event.set()

class WhisperTyper(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.text_queue = queue.Queue()
        self.status_queue = queue.Queue()
        self.audio_level_queue = queue.Queue()
        self.keyboard_listener = None
        
        self.title("Whisper Typer")
        self.geometry("400x900")
        self.resizable(False, False)
        self.attributes('-topmost', True)
        
        self.recorder: Optional[AudioRecorder] = None
        
        self.load_settings()
        # Add default language if not present
        if "language" not in self.settings:
            self.settings["language"] = "en"
            self.settings["language_hotkey"] = "F8" 
            self.save_settings()
        self.setup_ui()
        self.setup_hotkey()
        self.setup_language_hotkey()
        self.check_queues()

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)

        container = ctk.CTkFrame(self)
        container.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        container.grid_columnconfigure(0, weight=1)
        
        title = ctk.CTkLabel(
            container,
            text="Whisper Typer",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=("#1E90FF", "#00BFFF")  # Blue shade
        )
        title.grid(row=0, column=0, pady=(20, 5))
        
        subtitle = ctk.CTkLabel(
            container,
            text="Speech-to-Text Anywhere",
            font=ctk.CTkFont(size=14),
            text_color=("gray70", "gray30")
        )
        subtitle.grid(row=1, column=0, pady=(0, 20))
        
        status_frame = ctk.CTkFrame(container, fg_color=("gray90", "gray20"))
        status_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 20))
        self.status_label = ctk.CTkLabel(
            status_frame, 
            text="Status: Initializing...",
            font=ctk.CTkFont(size=13)
        )
        self.status_label.grid(row=0, column=0, padx=10, pady=5)
        
        self.progress_label = ctk.CTkLabel(
            status_frame,
            text="○",
            font=ctk.CTkFont(size=20),
            text_color="gray"
        )
        self.progress_label.grid(row=0, column=1, padx=(0, 10), pady=5)
        
        self.toggle_button = ctk.CTkButton(
            container,
            text=f"🎤 Start Recording (or press {self.settings['hotkey']})",
            command=self.toggle_recording,
            height=45,
            font=ctk.CTkFont(size=15, weight="bold"),
            corner_radius=10
        )
        self.toggle_button.grid(row=3, column=0, pady=20, padx=20, sticky="ew")
        
        hotkey_frame = ctk.CTkFrame(container)
        hotkey_frame.grid(row=4, column=0, pady=10, padx=20, sticky="ew")
        hotkey_frame.grid_columnconfigure(1, weight=1)
        
        hotkey_label = ctk.CTkLabel(hotkey_frame, text="Hotkey:", font=ctk.CTkFont(size=13))
        hotkey_label.grid(row=0, column=0, padx=(10, 5), pady=5)
        
        self.hotkey_combo = ctk.CTkOptionMenu(
            hotkey_frame,
            values=["F9", "F10", "F11", "F12"],
            command=self.update_hotkey,
            width=120
        )
        self.hotkey_combo.grid(row=0, column=1, padx=5, pady=5)
        self.hotkey_combo.set(self.settings["hotkey"])
        
        language_frame = ctk.CTkFrame(container)
        language_frame.grid(row=5, column=0, pady=10, padx=20, sticky="ew")
        language_frame.grid_columnconfigure(1, weight=1)
        
        self.language_button = ctk.CTkButton(
            language_frame,
            text=f"🌐 Current: {'English' if self.settings['language'] == 'en' else 'Turkish'}",
            command=self.toggle_language,
            height=35,
            font=ctk.CTkFont(size=13)
        )
        self.language_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        language_hotkey_label = ctk.CTkLabel(
            language_frame,
            text=f"Toggle with {self.settings['language_hotkey']}",
            font=ctk.CTkFont(size=11)
        )
        language_hotkey_label.grid(row=1, column=0, padx=5, pady=(0,5))
        
        audio_settings_frame = ctk.CTkFrame(container)
        audio_settings_frame.grid(row=6, column=0, pady=10, padx=20, sticky="ew")
        
        sensitivity_label = ctk.CTkLabel(
            audio_settings_frame,
            text="Microphone Sensitivity:",
            font=ctk.CTkFont(size=13)
        )
        sensitivity_label.grid(row=0, column=0, padx=10, pady=(5,0), sticky="w")
        
        # Audio Level Indicator
        self.audio_level_frame = ctk.CTkProgressBar(
            audio_settings_frame,
            mode="determinate",
            height=15,
            width=200
        )
        self.audio_level_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=(5,5), sticky="ew")
        self.audio_level_frame.set(0)
        
        self.threshold_label = ctk.CTkLabel(
            audio_settings_frame,
            text="▼",
            font=ctk.CTkFont(size=12),
            text_color="yellow"
        )
        self.threshold_label.grid(row=2, column=0, columnspan=2, padx=10, sticky="w")
        
        self.sensitivity_slider = ctk.CTkSlider(
            audio_settings_frame,
            from_=0.005,
            to=0.05,
            number_of_steps=100,
            command=self.update_sensitivity
        )
        self.sensitivity_slider.grid(row=3, column=0, columnspan=2, padx=10, pady=(0,10), sticky="ew")
        self.sensitivity_slider.set(self.settings["sensitivity"])
        
        self.sensitivity_value_label = ctk.CTkLabel(
            audio_settings_frame,
            text=f"Current: {self.settings['sensitivity']:.3f}",
            font=ctk.CTkFont(size=11)
        )
        self.sensitivity_value_label.grid(row=4, column=0, columnspan=2, pady=(0,5))
        
        instructions = """
        📝 Instructions:
        
        1. Click the button or press hotkey to start recording
        2. Speak clearly into your microphone
        3. Text will appear where your cursor is
        4. Click again or press hotkey to stop
        
        💡 Tips:
        • Position your cursor where you want the text
        • Speak naturally and clearly
        • Use the hotkey for quick access
        """
        
        info_frame = ctk.CTkFrame(container)
        info_frame.grid(row=7, column=0, pady=20, padx=20, sticky="ew")
        
        info_label = ctk.CTkLabel(
            info_frame,
            text=instructions,
            justify="left",
            wraplength=300,
            font=ctk.CTkFont(size=13)
        )
        info_label.pack(padx=15, pady=15)

    def toggle_recording(self):
        if not self.recorder or not self.recorder.is_alive():
            print("[DEBUG] Starting recording")
            self.initialize_recorder()
            self.recorder.start()
            
            self.toggle_button.configure(
                text=f"⏹ Stop Recording (or press {self.settings['hotkey']})",
                fg_color="#E74C3C",
                hover_color="#C0392B"
            )
            self.update_status(("Status: Recording", "green"))
        else:
            print("[DEBUG] Stopping recording")
            self.recorder.stop()
            self.recorder.join()
            self.toggle_button.configure(
                text=f"🎤 Start Recording (or press {self.settings['hotkey']})",
                fg_color=["#1E90FF", "#0078D7"],
                hover_color=["#1871CD", "#005FB3"]
            )
            self.update_status(("Status: Ready", "gray"))
            self.recorder = None

    def load_settings(self):
        try:
            with open("settings.json", "r") as f:
                self.settings = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.settings = {"hotkey": "F9", "sensitivity": 0.05}
            self.save_settings()

    def save_settings(self):
        with open("settings.json", "w") as f:
            json.dump(self.settings, f, indent=4)

    def setup_hotkey(self):
        """Setup hotkey using pynput correctly for function keys"""
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        
        hotkey = self.settings["hotkey"].lower()
        key_mapping = {
            "f9": Key.f9,
            "f10": Key.f10,
            "f11": Key.f11,
            "f12": Key.f12
        }
        
        if hotkey not in key_mapping:
            print(f"[DEBUG] Unsupported hotkey: {hotkey}. Defaulting to F9.")
            hotkey = "f9"
            self.settings["hotkey"] = "F9"
            self.save_settings()
        
        def on_activate():
            self.toggle_recording()
        
        self.keyboard_listener = kb.GlobalHotKeys({
            f'<{hotkey}>': on_activate
        })
        self.keyboard_listener.start()

    def update_hotkey(self, new_hotkey):
        """Update hotkey with pynput"""
        self.settings["hotkey"] = new_hotkey
        self.save_settings()
        self.setup_hotkey()
        self.toggle_button.configure(
            text=f"🎤 Start Recording (or press {self.settings['hotkey']})"
        )

    def update_status(self, status_info):
        """Update the status label text and color"""
        if isinstance(status_info, tuple):
            status, color = status_info
        else:
            status, color = status_info, "white"
            
        self.status_label.configure(text=status)
        self.progress_label.configure(text_color=color)
        
        if "Processing" in status:
            self.progress_label.configure(text="◉", text_color="blue")
        elif "Recording" in status:
            self.progress_label.configure(text="⬤", text_color="green")
        else:
            self.progress_label.configure(text="○", text_color="gray")

    def type_text(self, text: str):
        """Type text using pyautogui with support for Turkish characters"""
        try:
            self.after(100)
            
            if self.settings["language"] == "tr":
                kb_controller = kb.Controller()
                for char in text:
                    kb_controller.type(char)
                    time.sleep(0.01)  
                kb_controller.type(" ")  
                time.sleep(0.01)
            else:
                pyautogui.write(text + " ", interval=0.01)
                
        except Exception as e:
            self.update_status((f"Typing error: {str(e)}", "red"))
            print(f"[DEBUG] Typing error: {str(e)}")

    def on_closing(self):
        """Clean up resources before closing"""
        print("[DEBUG] Closing application...")
        if self.recorder and self.recorder.is_alive():
            print("[DEBUG] Stopping recorder...")
            self.recorder.stop()
            self.recorder.join()
        
        if self.keyboard_listener:
            print("[DEBUG] Stopping keyboard listener...")
            self.keyboard_listener.stop()
        
        if hasattr(self, 'language_listener'):
            print("[DEBUG] Stopping language listener...")
            self.language_listener.stop()
        
        self.destroy() 

    def check_queues(self):
        """Check text, status, and audio level queues for updates"""
        try:
            while True:
                try:
                    text = self.text_queue.get_nowait()
                    print(f"[DEBUG] Got text from queue: {text}")
                    self.type_text(text)
                except queue.Empty:
                    break

            while True:
                try:
                    status = self.status_queue.get_nowait()
                    print(f"[DEBUG] Got status update: {status}")
                    self.update_status(status)
                except queue.Empty:
                    break

            while True:
                try:
                    rms = self.audio_level_queue.get_nowait()
                    print(f"[DEBUG] Got audio level: {rms}")
                    self.update_audio_level(rms)
                except queue.Empty:
                    break

        except Exception as e:
            print(f"[DEBUG] Queue error: {str(e)}")
        finally:
            # Schedule next check
            self.after(50, self.check_queues)

    def initialize_recorder(self):
        """Initialize the recorder instance"""
        if self.recorder is None or not self.recorder.is_alive():
            self.recorder = AudioRecorder(
                self.text_queue, 
                self.status_queue,
                self.audio_level_queue,
                self.settings
            )

    def update_sensitivity(self, value):
        """Update sensitivity value and save to settings"""
        self.settings["sensitivity"] = float(value)
        self.save_settings()
        self.sensitivity_value_label.configure(text=f"Current: {value:.3f}")
        # Update threshold indicator position
        relative_pos = (value - 0.005) / (0.05 - 0.005)  # Adjusted for new range
        self.threshold_label.grid_configure(padx=(10 + relative_pos * 180, 0))
        if self.recorder:
            self.recorder.audio_buffer.silence_threshold = value
            self.recorder.audio_buffer.noise_threshold = value * 0.4

    def update_audio_level(self, rms_value):
        """Update the audio level indicator"""
        # Normalize RMS value to 0-1 range for progress bar
        normalized_value = min(1.0, rms_value / self.settings["sensitivity"])
        self.audio_level_frame.set(normalized_value)
        
        # Update color based on threshold
        if rms_value > self.settings["sensitivity"]:
            self.audio_level_frame.configure(progress_color="green")
        elif rms_value > self.settings["sensitivity"] * 0.4:
            self.audio_level_frame.configure(progress_color="yellow")
        else:
            self.audio_level_frame.configure(progress_color="gray")

    def setup_language_hotkey(self):
        """Setup hotkey for language toggle"""
        def on_language_toggle():
            self.toggle_language()
        
        self.language_listener = kb.GlobalHotKeys({
            f'<{self.settings["language_hotkey"].lower()}>': on_language_toggle
        })
        self.language_listener.start()

    def toggle_language(self):
        """Toggle between English and Turkish"""
        self.settings["language"] = "tr" if self.settings["language"] == "en" else "en"
        self.save_settings()
        self.language_button.configure(
            text=f"🌐 Current: {'English' if self.settings['language'] == 'en' else 'Turkish'}"
        )
        self.update_status(("Language changed to " + 
                           ("English" if self.settings["language"] == "en" else "Turkish"), 
                           "blue"))

if __name__ == "__main__":
    try:
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        app = WhisperTyper()
        app.protocol("WM_DELETE_WINDOW", app.on_closing)
        app.mainloop()
    except Exception as e:
        print(f"Application error: {str(e)}")

