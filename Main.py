import cv2
import pyautogui
import MouseControl as mc  # Importing the custom MouseControl module

# Capture the camera
cap = cv2.VideoCapture(0)  # Initialize video capture from the camera
# Initialize mouse control
origin = 0  # Flag to track if the mouse origin has been set
control = mc.mouseControl()  # Creating an instance of the mouseControl class from the MouseControl module
# Get screen size
screen_w, screen_h = pyautogui.size()  # Get the screen width and height using pyautogui
pyautogui.FAILSAFE = False  # Disable the fail-safe feature of pyautogui, which terminates the script when the mouse cursor reaches the top-left corner

while True:
    
    # Read frame from camera
    success, img = cap.read()  # Read a frame from the video capture
    # Resize and flip image
    img = cv2.resize(img, (1280, 720))  # Resize the image to 1280x720 pixels
    img = cv2.flip(img, 1)  # Flip the image horizontally
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video capture
    
    h, w, c = img.shape  # Get the height, width, and number of channels of the image
    
    # Find face and its landmarks
    img = control.findFace(img)  # Detect faces in the image using the mouseControl class
    lmList = control.findPosition(img, False)  # Find landmark positions without drawing them on the image
    control.drawFace(img, False, False)  # Draw the face mesh on the image (without tessellation and contours)
    
    # Define landmark lists for different facial features
    pList_1 = [133, 173, 157, 158, 159, 160, 161, 246, 33, 7, 163, 144, 145, 153, 154, 155]  # List of landmark indices for the right eye
    pList_2 = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]  # List of landmark indices for the left eye
    pList_3 = [19, 354, 461, 457, 440, 363, 281, 5, 51, 134, 220, 237, 241, 125]  # List of landmark indices for the nose
    pList_4 = [5, 275, 45, 1]  # List of landmark indices for the mouse origin
    
    """
    important landmarks
    
    145 - 159 right eye middle of the eyelid
    374 - 386 left eye middle of the eyelid
        4     nose tip (mouse origin) 
    """
    
    if len(lmList) != 0:  # Check if landmarks are detected
        
        control.drawFeatures(img, pList_1)  # Draw features related to the right eye on the image
        control.drawFeatures(img, pList_2)  # Draw features related to the left eye on the image
        #control.drawFeatures(img, pList_3)  # Draw features related to the nose on the image
        for p in pList_4:
            x1, y1 = lmList[4][1:]  # Coordinates of the nose tip (mouse origin)
            x2, y2 = lmList[p][1:]  # Coordinates of other specified landmarks                       
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1)  # Draw lines between the nose tip and other specified landmarks   
        distx_r, disty_r = control.pDist(145, 159)  # Calculate the distance between landmarks of the right eye
        distx_l, disty_l = control.pDist(374, 386)  # Calculate the distance between landmarks of the left eye
        if origin == 0:  # Check if the mouse origin has been set
            origin = 1  # Set the origin flag to 1
            x_origin, y_origin = lmList[4][1:]  # Set the mouse origin coordinates to the nose tip
            pyautogui.moveTo(x_origin, y_origin)  # Move the mouse cursor to the origin
            
        perimeter = 60  # Define the perimeter radius for mouse movement
        fast = 45  # Define the fast movement radius
        normal = 30  # Define the normal movement radius
        slow = 15  # Define the slow movement radius
        closed_eye = 5  # Define the threshold for closed eyes
        
        x_nose, y_nose = lmList[4][1:]  # Get the current coordinates of the nose tip 
        # Draw circles representing different movement speeds around the mouse origin
        cv2.circle(img, (x_origin, y_origin), perimeter, (255, 255, 255), 1)
        cv2.circle(img, (x_origin, y_origin), fast, (0, 0, 255), 1)
        cv2.circle(img, (x_origin, y_origin), normal, (0, 255, 0), 1)
        cv2.circle(img, (x_origin, y_origin), slow, (255, 0, 0), 1)
        cv2.line(img, (x_origin, y_origin), (x_nose, y_nose), (0, 255, 0), 1)  # Draw a line from the origin to the nose tip
                     
        distORG_x, distORG_y = x_nose - x_origin, y_nose - y_origin  # Calculate the distance between the nose tip and the origin
        # Calculate the amount of mouse movement based on different speeds
        moveX_fast, moveY_fast = distORG_x/2.5, distORG_y/2.5
        moveX_normal, moveY_normal = distORG_x/5, distORG_y/5
        moveX_slow, moveY_slow = distORG_x/10, distORG_y/10
        
        # Check if the nose tip is outside the movement perimeter
        if distORG_x > perimeter or distORG_x < -perimeter or distORG_y > perimeter or distORG_y < -perimeter:
            cv2.putText(img, "Out of the perimeter.", (380, 65), cv2.FONT_HERSHEY_PLAIN, 3,
                (0, 0, 255), 3)
            
        # Check the distance between the nose tip and the origin for different movement speeds
        elif distORG_x > slow or distORG_x < -slow or distORG_y > slow or distORG_y < -slow:            
            cv2.putText(img, str(int(distORG_x)), (590, 65), cv2.FONT_HERSHEY_PLAIN, 2,
                (0, 0, 255), 2)
            cv2.putText(img, str(int(distORG_y)), (690, 65), cv2.FONT_HERSHEY_PLAIN, 2,
                (0, 0, 255), 2)
            
            # Move the mouse cursor relative to the current position based on different speeds
            if distORG_x > fast or distORG_x < -fast or distORG_y > fast or distORG_y < -fast:
                pyautogui.moveRel(moveX_fast, moveY_fast)
                
            elif distORG_x > normal or distORG_x < -normal or distORG_y > normal or distORG_y < -normal:
                pyautogui.moveRel(moveX_normal, moveY_normal)
                
            elif distORG_x > slow or distORG_x < -slow or distORG_y > slow or distORG_y < -slow:
                pyautogui.moveRel(moveX_slow, moveY_slow)
        
        # Check if the eyes are closed and perform mouse click actions accordingly
        if disty_r < closed_eye and disty_l > closed_eye:            
            pyautogui.click(button='left')  # Perform a left mouse click
        elif disty_l < closed_eye and disty_r > closed_eye:            
            pyautogui.click(button='right')  # Perform a right mouse click
        elif disty_r < closed_eye and disty_l < closed_eye:
            origin = 0  # Reset the origin flag if both eyes are closed
            
    # Display the frame rate on the image
    cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5,
                (255, 0, 0), 5)
    
    # Display the processed image
    cv2.imshow("Mouse Control", img)
    if cv2.waitKey(5) & 0xFF == ord("q"):  # Exit the loop if 'q' is pressed
        break
    
cap.release()  # Release the video capture
cv2.destroyAllWindows()  # Close all OpenCV windows
