try:
    import mediapipe as mp
    print(f"Vers: {mp.__version__}")
    from mediapipe.python.solutions import hands
    print("Hands imported!")
except Exception as e:
    print(f"Error: {e}")
