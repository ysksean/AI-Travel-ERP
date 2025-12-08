import os

file_path = r'c:\AIDC\travel\flask_web\templates\product_create.html'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Fix renderCalendar function
# The original code has a comment that swallows the variable declarations:
# cell.textContent = d; // Simplified logic for demo const startDate=null; ...

# We will replace the entire renderCalendar function with a clean version.

old_render_calendar_start = 'function renderCalendar() {'
old_render_calendar_end = '} window.changeMonth = function'

new_render_calendar = """function renderCalendar() {
        const currentDate = new Date();
        const year = currentDate.getFullYear();
        const month = currentDate.getMonth();
        const title = document.getElementById('calendarTitle');
        if (title) title.textContent = `${year}년 ${month + 1} 월`;

        const firstDay = new Date(year, month, 1);
        const lastDay = new Date(year, month + 1, 0);
        const grid = document.getElementById('calendarGrid');
        if (!grid) return;
        grid.innerHTML = '';

        for (let i = 0; i < firstDay.getDay(); i++) {
            grid.appendChild(document.createElement('div'));
        }

        // Mock selected range for demo
        const startDate = new Date(year, month, 1);
        const endDate = new Date(year, month, 5);

        for (let d = 1; d <= lastDay.getDate(); d++) {
            const date = new Date(year, month, d);
            const cell = document.createElement('div');
            cell.className = 'aspect-square flex items-center justify-center rounded-lg text-sm cursor-default transition-colors';
            cell.textContent = d;

            if (startDate && endDate) {
                if (date >= startDate && date <= endDate) {
                    cell.classList.add('bg-indigo-100', 'text-indigo-700', 'font-bold');
                    // Fix comparison for highlighting start/end
                    if (date.getDate() === startDate.getDate()) cell.classList.add('bg-indigo-600', 'text-white');
                    if (date.getDate() === endDate.getDate()) cell.classList.add('bg-indigo-600', 'text-white');
                }
            }
            grid.appendChild(cell);
        }
    }"""

# Locate the function in the content
# Since the original code is messy, we might need to find a unique start and end point.
# Start: function renderCalendar() {
# End: } window.changeMonth = function

start_idx = content.find(old_render_calendar_start)
end_idx = content.find(old_render_calendar_end)

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + new_render_calendar + content[end_idx:]
    print("Fixed renderCalendar function")
else:
    print("Could not find renderCalendar function boundaries")
    # Fallback: try to match a larger chunk if the end marker is ambiguous
    # But 'window.changeMonth' seems unique enough.

# 2. Add click handler for dateRangePicker to scroll to calendar
# We'll add this to the end of the script block
js_append_marker = '// Initialize Timeline'
js_scroll_logic = """
    const datePicker = document.getElementById('dateRangePicker');
    if (datePicker) {
        datePicker.addEventListener('click', function() {
            const calendarSection = document.getElementById('calendarGrid').closest('section');
            if (calendarSection) {
                calendarSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
                // Highlight the section briefly
                calendarSection.classList.add('ring-2', 'ring-indigo-500', 'transition-all', 'duration-500');
                setTimeout(() => calendarSection.classList.remove('ring-2', 'ring-indigo-500'), 1000);
            }
        });
    }
    """

if js_append_marker in content:
    content = content.replace(js_append_marker, js_scroll_logic + '\n    ' + js_append_marker)
    print("Added dateRangePicker scroll logic")
else:
    print("Could not find JS append marker")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
