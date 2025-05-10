# 🚁 Drone Tracking with YOLOv8 + SORT

This project is focused on detecting and tracking drones in videos using a custom-trained YOLOv8 model and the SORT (Simple Online and Realtime Tracking) algorithm.

## 📌 Features

- Real-time drone detection using YOLOv8
- Object tracking with persistent IDs via SORT (Kalman Filter + Hungarian Algorithm)
- Works with video files or live webcam input
- Supports custom training with drone datasets

## 🚀 Installation

```bash
git clone https://github.com/Askhat1/Final-Project.git
cd Final-Project
pip install -r requirements.txt

```
▶️ How to Run

### Run on a Video File

Make sure you place your video(with name drone_video.mp4) file in the `input/` directory, then run:

```bash
python main.py
```
Run on Webcam
```python camera_live.py```
The result will be saved in output/result.mp4


## 👤 Author

This project was created by **Askhat S.**

📬 Contact: [@aswsss on Telegram](https://t.me/aswsss)  
🔗 GitHub: [github.com/Askhat1](https://github.com/Askhat1)
