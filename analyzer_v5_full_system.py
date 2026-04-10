import cv2
from transformers import pipeline, VideoMAEImageProcessor, VideoMAEForVideoClassification, AutoImageProcessor, AutoModelForImageClassification, ASTFeatureExtractor, ASTForAudioClassification
from PIL import Image
import torch
import argparse
import os
import face_recognition
import numpy as np
from ultralytics import YOLO
import collections
from moviepy.editor import VideoFileClip
import librosa

class FinalJanusSystem:
    def __init__(self):
        # --- AI Model Placeholders ---
        self.captioner = None
        self.object_detector = None
        self.action_recognizer = None
        self.action_processor = None
        self.emotion_recognizer = None
        self.emotion_processor = None
        self.audio_classifier = None
        self.audio_processor = None
        
        # --- System Variables ---
        self.known_face_encodings = []
        self.known_face_names = []
        self.device = 0 if torch.cuda.is_available() else -1

    def load_ai_models(self):
        """Loads all five necessary AI models."""
        print("--- Loading All AI Models for Final Janus System ---")
        try:
            print("1/5: Loading Captioning model...")
            self.captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device=self.device)
            
            print("2/5: Loading Object Detection model...")
            self.object_detector = YOLO('yolov8n.pt')
            
            print("3/5: Loading Action Recognition model...")
            action_model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
            self.action_processor = VideoMAEImageProcessor.from_pretrained(action_model_name)
            self.action_recognizer = VideoMAEForVideoClassification.from_pretrained(action_model_name).to(self.device)
            
            print("4/5: Loading Facial Emotion model...")
            emotion_model_name = "michellejieli/emotion_recognition_finetuned"
            self.emotion_processor = AutoImageProcessor.from_pretrained(emotion_model_name)
            self.emotion_recognizer = AutoModelForImageClassification.from_pretrained(emotion_model_name).to(self.device)

            print("5/5: Loading Audio Classification model...")
            audio_model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
            self.audio_processor = ASTFeatureExtractor.from_pretrained(audio_model_name)
            self.audio_classifier = ASTForAudioClassification.from_pretrained(audio_model_name).to(self.device)

        except Exception as e:
            print(f"Fatal Error during model loading: {e}"); return False
            
        print("--- All models loaded successfully. Janus is fully operational. ---")
        return True

    def enroll_known_faces(self, known_faces_dir: str):
        """Learns faces from a directory structure."""
        print(f"\n--- Enrolling Known Faces from '{known_faces_dir}' ---")
        if not os.path.isdir(known_faces_dir):
            print(f"Warning: Known faces directory not found."); return
        for person_name in os.listdir(known_faces_dir):
            person_dir = os.path.join(known_faces_dir, person_name)
            if os.path.isdir(person_dir):
                for filename in os.listdir(person_dir):
                    image_path = os.path.join(person_dir, filename)
                    try:
                        known_image = face_recognition.load_image_file(image_path)
                        face_encoding = face_recognition.face_encodings(known_image)[0]
                        self.known_face_encodings.append(face_encoding)
                        self.known_face_names.append(person_name)
                    except Exception: pass
        if self.known_face_names:
            print(f"Enrollment complete. Known characters: {list(set(self.known_face_names))}")

    def analyze_video(self, video_path: str, output_file: str):
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at '{video_path}'"); return

        # --- Audio Extraction ---
        print("\n--- Extracting Audio from Video ---")
        try:
            video_clip = VideoFileClip(video_path)
            audio_path = "temp_audio.wav"
            video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le')
            audio_data, audio_sr = librosa.load(audio_path, sr=16000)
            print("Audio extracted successfully.")
        except Exception as e:
            print(f"Warning: Could not extract audio: {e}. Proceeding without audio analysis.")
            audio_data = None

        # --- Video Processing ---
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30
        
        print(f"\n--- Processing Video: {video_path} ({fps:.2f} FPS) ---")
        all_analyses = []
        frame_count = 0
        frame_buffer = collections.deque(maxlen=16)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_buffer.append(rgb_frame)

            if frame_count % int(fps) == 0:
                timestamp = frame_count // int(fps)
                print(f"\n  - Analyzing moment at timestamp ~{timestamp}s...")
                
                # --- 1. Character & Emotion Recognition ---
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                char_analysis = []
                for i, (top, right, bottom, left) in enumerate(face_locations):
                    face_image = rgb_frame[top:bottom, left:right]
                    face_encoding = face_encodings[i]
                    
                    # Recognize name
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown Person"
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]: name = self.known_face_names[best_match_index]
                    
                    # Recognize emotion
                    inputs = self.emotion_processor(images=Image.fromarray(face_image), return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        logits = self.emotion_recognizer(**inputs).logits
                    emotion = self.emotion_recognizer.config.id2label[logits.argmax().item()]
                    char_analysis.append(f"{name} ({emotion})")

                # --- 2. Object Detection ---
                results = self.object_detector(frame, verbose=False)
                names = self.object_detector.names
                detected_objects = [names[int(c)] for r in results for c in r.boxes.cls if names[int(c)] != 'person']

                # --- 3. Action Recognition ---
                detected_action = "N/A"
                if len(frame_buffer) == 16:
                    inputs = self.action_processor(list(frame_buffer), return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = self.action_recognizer(**inputs)
                    detected_action = self.action_recognizer.config.id2label[outputs.logits.argmax(-1).item()]

                # --- 4. Audio Analysis ---
                audio_mood = "N/A"
                if audio_data is not None:
                    start_sample = int(timestamp * audio_sr)
                    end_sample = int((timestamp + 1) * audio_sr)
                    audio_chunk = audio_data[start_sample:end_sample]
                    if len(audio_chunk) > 0:
                        inputs = self.audio_processor(audio_chunk, sampling_rate=16000, return_tensors="pt").to(self.device)
                        with torch.no_grad():
                            outputs = self.audio_classifier(**inputs)
                        # Get top 2 predictions
                        top_scores, top_indices = torch.topk(outputs.logits, 2)
                        audio_mood = ", ".join([self.audio_classifier.config.id2label[i.item()] for i in top_indices])

                # --- 5. Scene Captioning ---
                pil_image = Image.fromarray(rgb_frame)
                base_caption = self.captioner(pil_image)[0]['generated_text']
                
                # --- Synthesize Full Analysis ---
                analysis = f"Time: ~{timestamp}s\n"
                analysis += f"    Mood/Tone: {audio_mood}\n"
                analysis += f"    Action: {detected_action.replace('_', ' ')}\n"
                analysis += f"    Scene: {base_caption}\n"
                if char_analysis:
                    analysis += f"    Characters: {', '.join(char_analysis)}\n"
                if detected_objects:
                    object_counts = collections.Counter(detected_objects)
                    object_summary = ", ".join([f"{count} {obj}" if count == 1 else f"{count} {obj}s" for obj, count in object_counts.items()])
                    analysis += f"    Key Objects: {object_summary}\n"
                all_analyses.append(analysis)

                frame_count += 1
            
            cap.release()
            if os.path.exists("temp_audio.wav"): os.remove("temp_audio.wav")
            print("\nVideo processing complete.")

        except Exception as e:
            print(f"Fatal Error during video processing. Aborting. Error: {e}")
            return

        # --- Final Report Generation ---
        summary = "Janus System Final Analysis Report:\n\n"
        summary += "\n".join(all_analyses)
        with open(output_file, 'w') as f:
            f.write(summary)
        print(f"--- Analysis complete. Full report saved to '{output_file}' ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Janus System - Full AI Video Analyzer V5")
    parser.add_argument("video_file", type=str, help="Path to the video file to analyze.")
    parser.add_argument("--faces", type=str, default="known_faces", help="Directory for known faces.")
    parser.add_argument("--output", type=str, default="final_janus_report.txt", help="Path for the output report.")
    
    args = parser.parse_args()
    
    janus = FinalJanusSystem()
    if janus.load_ai_models():
        janus.enroll_known_faces(args.faces)
        janus.analyze_video(args.video_file, args.output)
