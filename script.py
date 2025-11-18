import cv2
import mediapipe as mp
import numpy as np


# --- Load images ---
normal_img = cv2.imread("normal.png")
smile_img = cv2.imread("smile.png")
mouth_open_img = cv2.imread("mouthopen.png")
wide_smile_img = cv2.imread("widesmile.png")
frown_img = cv2.imread("frown.png")



display_size = (800, 800)
normal_img = cv2.resize(normal_img, display_size)
smile_img = cv2.resize(smile_img, display_size)
mouth_open_img = cv2.resize(mouth_open_img, display_size)
wide_smile_img = cv2.resize(wide_smile_img, display_size)
frown_img = cv2.resize(frown_img, display_size)

# --- Mediapipe setup ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Landmark indices
TIP_NOSE = 1
UPPER_LIP = 13
LOWER_LIP = 14
INNER_LEFT_EYE = 33
INNER_RIGHT_EYE = 263
LEFT_CORNER_MOUTH = 61
RIGHT_CORNER_MOUTH = 291
INNERMOST_RIGHT_EYEBROW = 285
INNERMOST_LEFT_EYEBROW = 55
MIDDLE_RIGHT_EYEBROW = 296
MIDDLE_LEFT_EYEBROW = 66

# --- detection params ---
# consider adding hysteresis (two metrics, range of possible smile indices)
SMILE_THRESHOLD = 0.65   # smile threshold (if above, smiling)
OPEN_THRESHOLD = 0.2 # mouth open threshold (if above, consider mouth open)
EYEBROW_FROWN_THRESHOLD = 0.64


# --- State ---
is_smiling = False
is_mouth_open = False
is_frowning_eyebrow = False

# --- Camera ---
cap = cv2.VideoCapture(0)

def draw_text(img, text, org=(10,30)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)


    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]
        lm = face.landmark

        # pixel coords
        p_left_eye = np.array([lm[INNER_LEFT_EYE].x * w, lm[INNER_LEFT_EYE].y * h])
        p_right_eye = np.array([lm[INNER_RIGHT_EYE].x * w, lm[INNER_RIGHT_EYE].y * h])
        p_left_corner_mouth = np.array([lm[LEFT_CORNER_MOUTH].x * w, lm[LEFT_CORNER_MOUTH].y * h])
        p_right_corner_mouth = np.array([lm[RIGHT_CORNER_MOUTH].x * w, lm[RIGHT_CORNER_MOUTH].y * h])
        p_upper_lip = np.array([lm[UPPER_LIP].x * w, lm[UPPER_LIP].y * h])
        p_lower_lip = np.array([lm[LOWER_LIP].x * w, lm[LOWER_LIP].y * h])
        p_innermost_right_eyebrow = np.array([lm[INNERMOST_RIGHT_EYEBROW].x * w, lm[INNERMOST_RIGHT_EYEBROW].y * h])
        p_innermost_left_eyebrow = np.array([lm[INNERMOST_LEFT_EYEBROW].x * w, lm[INNERMOST_LEFT_EYEBROW].y * h])
        p_tip_nose = np.array([lm[TIP_NOSE].x * w, lm[TIP_NOSE].y * h])
        p_middle_right_eyebrow = np.array([lm[MIDDLE_RIGHT_EYEBROW].x * w, lm[MIDDLE_RIGHT_EYEBROW].y * h])
        p_middle_left_eyebrow = np.array([lm[MIDDLE_LEFT_EYEBROW].x * w, lm[MIDDLE_LEFT_EYEBROW].y * h])
        p_middle_eyebrow_average = (p_middle_left_eyebrow + p_middle_right_eyebrow) / 2


        eye_dist = np.linalg.norm(p_right_eye - p_left_eye)
        mouth_width = np.linalg.norm(p_right_corner_mouth - p_left_corner_mouth)
        mouth_height = np.linalg.norm(p_lower_lip - p_upper_lip)
        middle_eyebrow_nose_distance = np.linalg.norm(p_middle_eyebrow_average - p_tip_nose)
        lcorner_mouth_tip_nose_distance = np.linalg.norm(p_tip_nose - p_left_corner_mouth)
        rcorner_mouth_tip_nose_distance = np.linalg.norm(p_tip_nose - p_right_corner_mouth)

        # avoid division by zero / tiny numbers
        if eye_dist < 1e-6:
            smile_metric = 0.0
            mouth_open_metric = 0.0
            middle_eyebrow_frown_metric = 0.0
            tip_nose_frown_metricL = 0.0
            tip_nose_frown_metricR = 0.0
        else:
            smile_metric = mouth_width / eye_dist
            mouth_open_metric = mouth_height / eye_dist
            middle_eyebrow_frown_metric = middle_eyebrow_nose_distance / eye_dist
            tip_nose_frown_metricL = lcorner_mouth_tip_nose_distance / eye_dist
            tip_nose_frown_metricR = rcorner_mouth_tip_nose_distance / eye_dist

        # draw debug points
        cv2.circle(frame, tuple(p_left_corner_mouth.astype(int)), 3, (0, 255, 255), -1)
        cv2.circle(frame, tuple(p_right_corner_mouth.astype(int)), 3, (0, 255, 255), -1)
        cv2.circle(frame, tuple(p_left_eye.astype(int)), 3, (255, 0, 255), -1)
        cv2.circle(frame, tuple(p_right_eye.astype(int)), 3, (255, 0, 255), -1)
        cv2.circle(frame, tuple(p_upper_lip.astype(int)), 3, (0, 255, 255), -1)
        cv2.circle(frame, tuple(p_lower_lip.astype(int)), 3, (0, 255, 255), -1)
        cv2.circle(frame, tuple(p_innermost_right_eyebrow.astype(int)), 3, (255, 0, 255), -1)
        cv2.circle(frame, tuple(p_innermost_left_eyebrow.astype(int)), 3, (255, 0, 255), -1)
        cv2.circle(frame, tuple(p_tip_nose.astype(int)), 3, (197, 203, 225), -1)
        cv2.circle(frame, tuple(p_middle_right_eyebrow.astype(int)), 3, (0, 255, 0), -1)
        cv2.circle(frame, tuple(p_middle_left_eyebrow.astype(int)), 3, (0, 255, 0), -1)

    # --- state assignment ---
    if smile_metric >= SMILE_THRESHOLD and mouth_open_metric < OPEN_THRESHOLD:
        is_smiling = True
        is_mouth_open = False
        is_frowning_eyebrow = False
    elif mouth_open_metric >= OPEN_THRESHOLD and smile_metric < SMILE_THRESHOLD:
        is_smiling = False
        is_mouth_open = True
        is_frowning_eyebrow = False
    elif middle_eyebrow_frown_metric <= EYEBROW_FROWN_THRESHOLD:
        is_smiling = False
        is_mouth_open = False
        is_frowning_eyebrow = True
    elif mouth_open_metric >= OPEN_THRESHOLD and smile_metric >= SMILE_THRESHOLD:
        is_smiling = True
        is_mouth_open = True
        is_frowning_eyebrow = False
    else:
        is_smiling = False
        is_mouth_open = False
        is_frowning_eyebrow = False



    # --- output logic ---

    if is_smiling and not is_mouth_open and not is_frowning_eyebrow:
        output_img = smile_img
    elif is_smiling and is_mouth_open and not is_frowning_eyebrow:
        output_img = wide_smile_img
    elif not is_smiling and is_mouth_open and not is_frowning_eyebrow:
        output_img = mouth_open_img
    elif not is_smiling and not is_frowning_eyebrow and not is_mouth_open:
        output_img = normal_img
    elif not is_smiling and is_frowning_eyebrow and not is_mouth_open:
        output_img = frown_img




    # --- UI overlays for debugging / tuning ---
    debug = frame.copy()
    draw_text(debug, f"smile_metric:{smile_metric:.3f}", (10, 30))
    draw_text(debug, f"mouth_open_metric:{mouth_open_metric}", (10, 60))
    draw_text(debug, f"middle_eyebrow_frown_metric:{middle_eyebrow_frown_metric}", (10, 90))

    cv2.imshow("Camera (debug)", debug)
    cv2.imshow("Output", output_img)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()