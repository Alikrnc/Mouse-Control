import cv2
import mediapipe as mp

class mouseControl():
    
    def __init__(self, mode=False, maxFace=1, refine=True, detectionCon=0.5, trackCon=0.5):
        """
        Initialize the mouseControl class with the specified parameters.

        Args:
            mode: If set to False, the solution treats the input images as a video stream.
            maxFace: Maximum number of faces
            refine: Whether to further refine the landmark coordinates around the eyes and lips, 
                    and output additional landmarks around the irises by applying the Attention Mesh Model.
            detectionCon: Minimum confidence value ([0.0, 1.0]) from the face detection model 
                          for the detection to be considered successful.
            trackCon: Minimum confidence value ([0.0, 1.0]) from the landmark-tracking model for 
                      the face landmarks to be considered tracked successfully, 
                      or otherwise face detection will be invoked automatically on the next input image.
        """
        
        self.mode = mode
        self.maxFace = maxFace
        self.refine = refine
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpStyle = mp.solutions.drawing_styles
        self.mpFaceMesh = mp.solutions.face_mesh
        
        # Initialize the face mesh model
        self.FaceMesh = self.mpFaceMesh.FaceMesh(self.mode, self.maxFace, self.refine,
                                                  self.detectionCon, self.trackCon)
        
    def findFace(self, img):        
        """
        Detect faces in the input image and return the image with landmarks drawn.

        Args:
            img: The input image.

        Returns:
            The image with landmarks drawn.
        """
        # Convert image to RGB format
        img.flags.writeable = False
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        
        # Process the face mesh detection
        self.results = self.FaceMesh.process(imgRGB)        
        img.flags.writeable = True
        return img
        
    def findPosition(self, img, draw=True):
        """
        Find the positions of facial landmarks in the input image.

        Args:
            img: The input image.
            draw: A boolean indicating whether to draw the landmarks on the image.

        Returns:
            List of landmark positions.
        """
        # Initialize landmark list
        self.lmList = []
        # Check if face landmarks are detected
        if self.results.multi_face_landmarks:
            # Iterate through each detected face
            for lmark in self.results.multi_face_landmarks:
                # Iterate through each landmark point
                for id, lm in enumerate(lmark.landmark):
                    h, w, c = img.shape
                    cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z)
                    self.lmList.append([id, cx, cy])
                    # Draw landmarks on the image if required
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList
                          
    def drawFace(self, img, drawTess=True, drawCont=True):       
        """
        Draw the face mesh on the input image.

        Args:
            img: The input image.
            drawTess: A boolean indicating whether to draw the face mesh tesselation.
            drawCont: A boolean indicating whether to draw the face mesh contours.

        Returns:
            None
        """
        # Draw face mesh on the image
        if self.results.multi_face_landmarks: 
            for lm in self.results.multi_face_landmarks:             
                if drawTess:
                    self.mpDraw.draw_landmarks(
                        img,
                        lm,
                        self.mpFaceMesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mpStyle
                        .get_default_face_mesh_tesselation_style())
                if drawCont:
                    self.mpDraw.draw_landmarks(
                        img,
                        lm,
                        self.mpFaceMesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mpStyle
                        .get_default_face_mesh_contours_style())
        return None
    
    def drawFeatures(self, img, pList, color=(0, 255, 0), size=1):
        """
        Draw specific facial features using provided landmark points.

        Args:
            img: The input image.
            pList: List of landmark points to draw.
            color: Color of the drawn points.
            size: Size of the drawn points.

        Returns:
            None
        """
        # Draw specific facial features using provided landmark points
        for p in pList:
            x, y = self.lmList[p][1:]                        
            cv2.circle(img, (x, y), size, color, cv2.FILLED)
        return None
    
    def pDist(self, p1, p2):
        """
        Calculate Euclidean distance between two landmark points.

        Args:
            p1: Index of the first landmark point.
            p2: Index of the second landmark point.

        Returns:
            Tuple containing the horizontal and vertical distances between the points.
        """
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]        
        distx, disty = x2 - x1, y2 - y1
        if distx < 0:
            distx = 0 - distx
        if disty < 0:
            disty = 0 - disty                
        return distx, disty
