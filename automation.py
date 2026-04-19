import os

label_dir = "pens/labels/cap_inserted"

for file in os.listdir(label_dir):
    if file.endswith(".txt"):
        file_path = os.path.join(label_dir, file)
        
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                if parts[0] == '0':   # wrong class
                    parts[0] = '3'   # correct class for sword
            new_lines.append(" ".join(parts) + "\n")
        
        with open(file_path, "w") as f:
            f.writelines(new_lines)

print("✅ Labels fixed")