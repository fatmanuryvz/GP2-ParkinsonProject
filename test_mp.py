import mediapipe as mp
try:
    print(f"MediaPipe version: {mp.__version__}")
    print(f"MediaPipe solutions: {hasattr(mp, 'solutions')}")
    import mediapipe.python.solutions.hands as mp_hands
    print("Direct import of hands success")
except Exception as e:
    print(f"Error: {e}")
