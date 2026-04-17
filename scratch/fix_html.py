
import os

filepath = r'c:\Users\Fatmanur\Desktop\Parkinsons_Disease_GP2\parkinson_detection_app.html'

with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# We want to identify the "garbage" block between the correctly closed renderAdvancedReport
# and the start of the Drawing/Upload section.

start_marker = -1
end_marker = -1

# Identify the good closure of renderAdvancedReport (approx line 2852)
# and the start of the next section (approx line 3001)

for i, line in enumerate(lines):
    # Looking for the specifically broken block start
    if 'const content = document.getElementById("reportContent");' in line and 2800 < i < 2900:
        if start_marker == -1:
            start_marker = i
    # Looking for the next section
    if '// ── Çizim Upload ──' in line:
        end_marker = i
        break

if start_marker != -1 and end_marker != -1:
    print(f"Removing lines {start_marker + 1} to {end_marker}")
    # Remove lines from start_marker to end_marker-1
    # We want to keep the drawing section start line.
    new_lines = lines[:start_marker] + lines[end_marker:]
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("Cleanup successful.")
else:
    print(f"Markers not found. Start: {start_marker}, End: {end_marker}")
