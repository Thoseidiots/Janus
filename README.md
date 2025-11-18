# Janus Vision System

The Janus Vision System is an advanced, multi-modal AI pipeline designed to analyze and understand video content in a human-like way. It processes video to identify characters, objects, actions, and emotional tone to generate a comprehensive, narrative analysis of the content.

## Features

This system is built in five main stages:
1.  **Character Recognition:** Identifies pre-enrolled characters in the video.
2.  **Object Detection:** Detects and lists key objects within the scene using YOLO.
3.  **Action Recognition:** Classifies complex human actions by analyzing video motion.
4.  **Emotional/Tonal Analysis:** Infers character emotion from facial expressions and scene mood from audio analysis.
5.  **Synthesis:** Combines all data streams into a rich, time-stamped report.

## Setup and Usage

### 1. Prerequisites
Before installation, you may need system-level dependencies for `dlib`:
- **macOS:** `brew install cmake dlib`
- **Ubuntu/Debian:** `sudo apt-get install build-essential cmake libdlib-dev`

### 2. Installation
Clone the repository and install the required Python packages:
\`\`\`bash
git clone https://github.com/Thoseidiots/Janus.git
cd Janus
pip install -r requirements.txt
\`\`\`

### 3. Running the Analyzer
a. **Enroll Faces:** Create a `known_faces` directory. Inside it, create sub-directories for each person you want to recognize (e.g., `known_faces/character_name/`). Place one or more images of that person in their directory.

b. **Execute:** Run the analyzer from your terminal, pointing it to a video file.
\`\`\`bash
python analyzer_v5_full_system.py path/to/your/video.mp4
\`\`\`
The final report will be saved as `final_janus_report.txt`.
