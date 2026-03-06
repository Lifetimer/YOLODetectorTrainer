from pathlib import Path
def modify_first_line_path(new_path: str) -> None:
    with open(Path(__file__).parent / 'data.yaml', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if lines:
        if lines[0].startswith("path:"):
            colon_index = lines[0].find(':')
            if colon_index != -1:
                lines[0] = f"{lines[0][:colon_index+1]} {new_path}\n"
    
    with open(Path(__file__).parent / 'data.yaml', 'w', encoding='utf-8') as f:
        f.writelines(lines)
        
modify_first_line_path(str(Path(__file__).parent))