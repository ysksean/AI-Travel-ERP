# Rebuild product_create.html with correct structure

# Read the file
with open(r'templates\product_create.html', 'r', encoding='utf-8') as f:
    all_lines = f.readlines()

# Extract correct script block (from line 612, 0-based index 611)
correct_script = ''.join(all_lines[611:])

# We need to find the HTML content block
# The HTML should end with {% endblock %} before the scripts block
# Let's check if there's any HTML content before line 612
html_candidate = ''.join(all_lines[:611])

# Check if HTML content exists (should contain {% extends or {% block content)
if '{% extends' in html_candidate or '{% block content' in html_candidate:
    print("Found HTML content in first 611 lines")
    # But we also have duplicate script block, so we need to remove it
    # Find where the first script block ends
    first_script_end = html_candidate.rfind('{% endblock %}')
    if first_script_end != -1:
        # Extract HTML part (before first script block)
        html_part = html_candidate[:first_script_end]
        # Find where HTML content block ends
        content_end = html_part.rfind('{% endblock %}')
        if content_end != -1:
            html_content = html_part[:content_end + len('{% endblock %}')]
            print(f"Found HTML content, length: {len(html_content)}")
            # Rebuild file
            output = html_content + '\n\n' + correct_script
            with open(r'templates\product_create.html', 'w', encoding='utf-8') as f:
                f.write(output)
            print("File rebuilt successfully!")
        else:
            print("Error: Could not find HTML content block end")
    else:
        print("Error: Could not find first script block end")
else:
    print("No HTML content found. Need to add it manually.")
    print("The file structure is broken. We need the original HTML content.")

