import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
from PIL import Image

# Configure AI model
genai.configure(api_key="GOOGLE_API_KEY")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not found or failed to open.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize hand detector
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

# Function to get hand information
def getHandInfo(img):
    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        print("Detected Fingers:", fingers)
        return fingers, lmList
    return None

# Function to draw on canvas based on hand gestures
def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:  # Index finger up
        current_pos = lmList[8][0:2]  # Tip of the index finger
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, tuple(prev_pos), tuple(current_pos), (255, 0, 255), 10)
    elif fingers == [1, 1, 1, 1, 1]:  # All fingers up
        canvas = np.zeros_like(canvas)  # Clear canvas
    return current_pos, canvas

# Function to send image data to AI for processing
def sendToAI(model, canvas, fingers):
    if fingers == [1, 0, 0, 0, 0]:  # Thumb up gesture
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve The Math Problem", pil_image])
        if response:
            return response.text
    return None

# Initialize variables
prev_pos = None
canvas = None

# Main loop to capture video frames and process hand gestures
while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to capture frame from webcam.")
        break

    img = cv2.flip(img, 1)  # Flip image for mirror effect

    # Initialize canvas on first run
    if canvas is None:
        canvas = np.zeros_like(img)

    # Get hand gesture information
    info = getHandInfo(img)
    output_text = None
    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas)
        output_text = sendToAI(model, canvas, fingers)

    # Combine the original frame with the drawn canvas
    combined_img = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    cv2.imshow("Hand Gesture Recognition", combined_img)

    # Display AI response if available
    if output_text:
        print("AI Response:", output_text)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
