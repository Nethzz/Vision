# Vision



# Hand Gesture-Based Image Controller

This project is a hand gesture-controlled image viewer that allows users to navigate through images, zoom in and out, and detect specific gestures using OpenCV and MediaPipe. Additionally, it uploads zoomed images to Firebase Firestore.

## Features
- **Hand Gesture Detection**: Uses MediaPipe to track hand landmarks.
- **Swipe Detection**: Navigate through images by swiping left or right.
- **Zoom Control**: Adjust zoom level with a pinch gesture.
- **Circle Gesture Recognition**: Detect circular hand movements.
- **Firestore Integration**: Uploads zoomed images to Firebase Firestore.

## Technologies Used
- Python
- OpenCV
- MediaPipe
- NumPy
- Firebase Firestore

## Installation
### Prerequisites
Ensure you have Python installed (>=3.7) and install the necessary dependencies:
```sh
pip install opencv-python mediapipe numpy firebase-admin
```

### Firebase Setup
firebase console link - https://console.firebase.google.com/
1. Create a Firebase project and enable Firestore.
2. Generate a service account key and download the `vision-firebase.json` file.
3. Place `vision-firebase.json` in the project root.

## Usage
Run the script with:
```sh
python main.py
```
### Controls
- **Swipe Left/Right**: Change images.
- **Pinch Gesture**: Zoom in and out.
- **Circle Gesture**: Marks an "eye" position.
- **Press 'q'**: Exit the program.

## Project Structure
```
├── main.py               # Main script
├── vision-firebase.json  # Firebase credentials
├── radio/                # Image directory
└── README.md             # Project documentation
```

## Future Improvements
- Add more gestures for additional controls.
- Implement hand tracking optimizations.
- Improve UI/UX with overlays.

## License
This project is licensed under the MIT License.

## Author
Neethu Vasundharan Sheeja


