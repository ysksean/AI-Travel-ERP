import os

file_path = r'c:\AIDC\travel\flask_web\templates\product_create.html'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Fix HTML: Add onclick to dropzone and onchange to input
# Target:
# <div class="border-2 border-dashed border-slate-200 rounded-xl p-10 text-center hover:border-indigo-500 hover:bg-indigo-50 transition-all cursor-pointer mb-8" id="heroDropzone">
# <input type="file" class="hidden" multiple accept="image/*" id="heroImageInput">

target_html_start = 'id="heroDropzone">'
replacement_html_start = 'id="heroDropzone" onclick="document.getElementById(\'heroImageInput\').click()">'

target_input = 'id="heroImageInput">'
replacement_input = 'id="heroImageInput" onchange="handleHeroImages(this)">'

if target_html_start in content:
    content = content.replace(target_html_start, replacement_html_start)
    print("Added onclick to heroDropzone")
else:
    print("heroDropzone ID not found for replacement")

if target_input in content:
    content = content.replace(target_input, replacement_input)
    print("Added onchange to heroImageInput")
else:
    print("heroImageInput ID not found for replacement")

# 2. Add JavaScript function handleHeroImages
# I'll append it to the end of the script block, before window.handleLandFile
js_marker = 'window.handleLandFile = function (input) {'
js_code = """
        window.handleHeroImages = function(input) {
            const grid = document.getElementById('heroImageGrid');
            const files = Array.from(input.files);
            
            if (files.length === 0) return;

            files.forEach(file => {
                if (!file.type.startsWith('image/')) return;

                const reader = new FileReader();
                reader.onload = function(e) {
                    const div = document.createElement('div');
                    div.className = 'relative group aspect-square rounded-xl overflow-hidden border border-slate-200';
                    div.innerHTML = `
                        <img src="${e.target.result}" class="w-full h-full object-cover">
                        <div class="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2">
                            <button type="button" onclick="this.closest('div.relative').remove()" class="p-2 bg-white/90 rounded-full text-red-500 hover:bg-white transition-colors">
                                <i data-lucide="trash-2" size="16"></i>
                            </button>
                        </div>
                        <div class="absolute top-2 left-2 px-2 py-1 bg-black/60 text-white text-xs rounded-md opacity-0 group-hover:opacity-100 transition-opacity">
                            대표
                        </div>
                    `;
                    grid.appendChild(div);
                    lucide.createIcons();
                };
                reader.readAsDataURL(file);
            });
            
            // Reset input so same files can be selected again if needed (though usually we append)
            input.value = '';
        }

        """

if js_marker in content:
    content = content.replace(js_marker, js_code + js_marker)
    print("Added handleHeroImages function")
else:
    print("JS marker not found")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
