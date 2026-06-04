import json
import os

transcript_path = r"C:\Users\Fatmanur\.gemini\antigravity\brain\90559e59-97c3-4622-a4ea-b10bf846d583\.system_generated\logs\transcript.jsonl"
target_steps = [113, 114, 202, 203, 221, 222]

if os.path.exists(transcript_path):
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                d = json.loads(line)
                step = d.get("step_index")
                # Also capture if the step index is in the list
                if step in target_steps:
                    print(f"Found step {step}, type: {d.get('type')}")
                    # If there's content or tool_calls, write them out
                    if "tool_calls" in d and d["tool_calls"]:
                        for tc in d["tool_calls"]:
                            name = tc.get("name")
                            args = tc.get("args")
                            out_name = f"scratch/step_{step}_{name}.json"
                            with open(out_name, 'w', encoding='utf-8') as out_f:
                                json.dump(args, out_f, indent=2, ensure_ascii=False)
                            print(f"  Saved tool call args to {out_name}")
                    if "content" in d and d["content"]:
                        out_name = f"scratch/step_{step}_content.txt"
                        with open(out_name, 'w', encoding='utf-8') as out_f:
                            out_f.write(d["content"])
                        print(f"  Saved content to {out_name}")
            except Exception as e:
                print(f"Error parsing line {idx}: {e}")
else:
    print("Transcript not found")
