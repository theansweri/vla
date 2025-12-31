import os

file_path = r"c:\Meta\projects\SimCar-20250319\Assets\DashboardUI\Scripts\UIMiniMapFrame.cs"

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    inserted = False
    for line in lines:
        new_lines.append(line)
        if "//draw lines" in line and not inserted:
            # Check if the TODO already exists to avoid duplicates
            if len(lines) > lines.index(line) + 1 and "TODO" in lines[lines.index(line) + 1]:
                print("Comment already exists.")
                inserted = True
                continue
            
            indent = line[:line.find("//")]
            new_lines.append(f"{indent}// TODO: 之后需要恢复此段代码，用于在小地图上绘制路线 (Temporary commented out)\n")
            inserted = True

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print("Successfully added TODO comment.")

except Exception as e:
    print(f"Error: {e}")
