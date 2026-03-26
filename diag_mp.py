import mediapipe as mp
print("DEBUG: mediapipe attributes:")
print(dir(mp))
if hasattr(mp, 'solutions'):
    print("DEBUG: solutions FOUND")
else:
    print("DEBUG: solutions NOT FOUND")
