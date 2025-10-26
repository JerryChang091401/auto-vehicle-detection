# auto-vehicle-detection

---
Auto Vehincle Detection is a lightweight real-time perception system that detects nearby objects and estimates their collision risk level \
🟩 GREEN (Safe) 🟨 YELLOW (Caution)  🟥 RED (High Risk) 
## File Structure
main.py # Control point for real-time demo 
perception.py # YOLOv8 detection  
risk.py # Risk estimation logic 
speed_det.py # Optimizes existing FPS 
videos/ # Sample input videos 
requirements.txt # Dependency list 
Demo Video:
https://drive.google.com/file/d/1KTz4o2aSbyAm528ALn4YBNzu4gGY1vYz/view?usp=sharing

##  Installation
```bash
# Create and activate a virtual environment 
python3 -m venv autodetect 
source autodetect/bin/activate        # or venv\Scripts\activate on Windows

#  Install dependencies
pip install -r requirements.txt

# YOLOv8 model will auto-download on first run (yolov8n.pt)

python3 main.py # Starts the process of control
