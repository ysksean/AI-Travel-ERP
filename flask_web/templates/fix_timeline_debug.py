import os

file_path = r'c:\AIDC\travel\flask_web\templates\product_create.html'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove duplicate addEvent function
# The second one starts around line 1296.
# We'll look for the second occurrence of "window.addEvent = function (btn, type) {"
first_add_event = content.find("window.addEvent = function (btn, type) {")
if first_add_event != -1:
    second_add_event = content.find("window.addEvent = function (btn, type) {", first_add_event + 1)
    if second_add_event != -1:
        # Find the end of this function. It ends before "window.toggleMode" or similar.
        # Actually, looking at the file, the second addEvent is followed by toggleMode, handleSearch, selectItem, handleQuotationFile...
        # It seems the entire block from 1296 onwards is a duplicate of the block from 1117.
        
        # Let's check if we can identify the duplicate block.
        # The duplicate block seems to start with window.addEvent and go until...
        # In the previous view, we saw:
        # 1296: window.addEvent...
        # ...
        # 1416: window.handleQuotationFile...
        # 1448: window.handleQuotationFile... (Another duplicate!)
        
        # It seems I pasted a large chunk of code twice.
        # I should remove the second chunk.
        
        # Strategy: Keep the first occurrence of addEvent and everything up to the start of the second addEvent?
        # No, the second addEvent is likely at the end of the file.
        
        # Let's just remove the second definition of addEvent explicitly.
        # We need to find where it ends. It seems to end before window.toggleMode.
        pass

# Actually, let's just replace the entire script block with a clean version.
# This is safer than trying to patch it.
# I will reconstruct the script block from the known good parts.

script_start_marker = '{% block scripts %}'
script_end_marker = '{% endblock %}'

start_idx = content.find(script_start_marker)
end_idx = content.rfind(script_end_marker) # Use rfind to get the last one

if start_idx != -1 and end_idx != -1:
    # Extract the current script content
    current_script = content[start_idx:end_idx+len(script_end_marker)]
    
    # Define the clean script content
    # I will combine the logic from previous steps:
    # 1. Itinerary Controls (reset, regenerate)
    # 2. Sticky Bottom Bar (submit)
    # 3. Timeline & Event Logic (dummyDB, renderCalendar, addEvent, etc.)
    # 4. File Handlers (handleQuotationFile, handleHeroImages, handleLandFile)
    # 5. Initialization
    
    new_script = """{% block scripts %}
<script>
    console.log("Product Create Script Loaded");

    // --- 1. Itinerary Controls ---
    window.resetItinerary = function () {
        if (confirm('정말로 일정을 초기화하시겠습니까? 입력된 모든 일정이 삭제됩니다.')) {
            generateTimeline(); // Re-generates empty days
        }
    }

    window.regenerateItinerary = async function () {
        const btn = document.getElementById('regenerateBtn');
        const originalText = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = '<i data-lucide="loader-2" class="animate-spin" size="14"></i> 생성 중...';
        lucide.createIcons();

        await new Promise(r => setTimeout(r, 1000)); // Simulate AI delay

        generateTimeline(); // Reset first
        // Add some mock events
        const days = document.querySelectorAll('.day-section');
        if (days.length > 0) {
            // Add a mock flight to Day 1
            const day1Btn = days[0].querySelector('.add-event-btn');
            if (day1Btn) {
                addEvent(day1Btn, 'flight');
                addEvent(day1Btn, 'hotel');
            }
        }

        btn.disabled = false;
        btn.innerHTML = originalText;
        lucide.createIcons();
    }

    // --- 2. Sticky Bottom Bar & Submit ---
    window.submitForm = function (status) {
        document.getElementById('saveStatus').value = status;
        document.getElementById('productForm').dispatchEvent(new Event('submit'));
    }

    const productForm = document.getElementById('productForm');
    const statusModal = document.getElementById('statusModal');
    const modalContent = document.getElementById('modalContent');
    let isSuccess = false;

    if (productForm) {
        productForm.addEventListener('submit', async function (e) {
            e.preventDefault();

            const buttons = document.querySelectorAll('.fixed.bottom-0 button');
            buttons.forEach(b => b.disabled = true);

            // Simulate API call
            await new Promise(r => setTimeout(r, 1500));

            isSuccess = true; // Mock success
            showModal(isSuccess);

            buttons.forEach(b => b.disabled = false);
        });
    }

    window.showModal = function (success) {
        if (!statusModal || !modalContent) return;
        statusModal.classList.remove('hidden');
        if (success) {
            modalContent.innerHTML = `
                <div class="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <i data-lucide="check" class="text-green-600" size="24"></i></div>
                <h3 class="text-lg font-bold text-slate-800 mb-2">저장 완료</h3>
                <p class="text-slate-600 mb-6">상품 정보가 성공적으로 저장되었습니다.</p>
                <div class="flex gap-2">
                    <button onclick="handleModalAction('list')"
                        class="flex-1 py-3 bg-slate-100 text-slate-700 rounded-xl font-bold hover:bg-slate-200 transition-colors">목록으로</button>
                    <button onclick="handleModalAction('stay')"
                        class="flex-1 py-3 bg-indigo-600 text-white rounded-xl font-bold hover:bg-indigo-700 transition-colors shadow-lg shadow-indigo-200">계속
                        수정</button>
                </div>
            `;
        } else {
            modalContent.innerHTML = `
                <div class="w-12 h-12 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <i data-lucide="alert-circle" class="text-red-600" size="24"></i>
                </div>
                <h3 class="text-lg font-bold text-slate-800 mb-2">저장 실패</h3>
                <p class="text-slate-600 mb-6">오류가 발생했습니다. 다시 시도해주세요.</p>
                <button onclick="statusModal.classList.add('hidden')"
                    class="w-full py-3 bg-slate-100 text-slate-700 rounded-xl font-bold hover:bg-slate-200 transition-colors">닫기</button>
            `;
        }
        lucide.createIcons();
    }

    window.handleModalAction = function (action) {
        statusModal.classList.add('hidden');
        if (action === 'list') {
            window.location.href = '/products'; // Adjust URL as needed
        }
    }

    // --- 3. Timeline & Event Logic ---
    const dummyDB = {
        hotel: [
            { name: "하얏트 리젠시 다낭", image: "https://images.unsplash.com/photo-1566073771259-6a8506099945?auto=format&fit=crop&q=80&w=100", desc: "5성급 • 논누옥 비치", loc: "다낭, 베트남" },
            { name: "빈펄 리조트 & 스파", image: "https://images.unsplash.com/photo-1582719508461-905c673771fd?auto=format&fit=crop&q=80&w=100", desc: "5성급 • 프라이빗 풀빌라", loc: "호이안, 베트남" },
            { name: "노보텔 다낭 프리미어", image: "https://images.unsplash.com/photo-1566073771259-6a8506099945?auto=format&fit=crop&q=80&w=100", desc: "4성급 • 한강 뷰", loc: "다낭 시내" }
        ],
        golf: [
            { name: "다낭 CC", image: "https://images.unsplash.com/photo-1587174486073-ae5e5cff23aa?auto=format&fit=crop&q=80&w=100", desc: "18홀 • 듄스 코스", loc: "다낭" },
            { name: "몽고메리 링크스", image: "https://images.unsplash.com/photo-1535131749006-b7f58c99034b?auto=format&fit=crop&q=80&w=100", desc: "18홀 • 콜린 몽고메리 설계", loc: "꽝남" },
            { name: "바나힐 골프 클럽", image: "https://images.unsplash.com/photo-1605118386294-0365f3a13dd2?auto=format&fit=crop&q=80&w=100", desc: "18홀 • 산악형 코스", loc: "바나힐" }
        ],
        sightseeing: [
            { name: "바나힐 투어", image: "https://images.unsplash.com/photo-1559592413-7cec4d0cae2b?auto=format&fit=crop&q=80&w=100", desc: "테마파크 • 케이블카", loc: "다낭" },
            { name: "호이안 올드타운", image: "https://images.unsplash.com/photo-1552465011-b4e21bf6e79a?auto=format&fit=crop&q=80&w=100", desc: "유네스코 유산 • 야경", loc: "호이안" },
            { name: "미케 비치", image: "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&q=80&w=100", desc: "해변 • 서핑", loc: "다낭" }
        ]
    };

    function renderCalendar() {
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
                    if (date.getDate() === startDate.getDate()) cell.classList.add('bg-indigo-600', 'text-white');
                    if (date.getDate() === endDate.getDate()) cell.classList.add('bg-indigo-600', 'text-white');
                }
            }
            grid.appendChild(cell);
        }
    }
    
    window.changeMonth = function (delta) {
        renderCalendar();
    }

    const editorBlocks = document.getElementById('editorBlocks');
    const textBlockTemplate = document.getElementById('textBlockTemplate');
    const imageBlockTemplate = document.getElementById('imageBlockTemplate');

    if (editorBlocks) {
        new Sortable(editorBlocks, { animation: 150, handle: '.editor-block', ghostClass: 'bg-indigo-50' });
    }

    window.addTextBlock = function (initialContent = '') {
        const clone = textBlockTemplate.content.cloneNode(true);
        if (initialContent) {
            clone.querySelector('textarea').value = initialContent;
        }
        editorBlocks.appendChild(clone);
        lucide.createIcons();
    }

    window.addImageBlock = function () {
        const clone = imageBlockTemplate.content.cloneNode(true);
        new Sortable(clone.querySelector('.image-grid'), { animation: 150, ghostClass: 'bg-indigo-50' });
        editorBlocks.appendChild(clone);
        lucide.createIcons();
    }

    window.addImagesToBlock = function (input) {
        const grid = input.parentElement.nextElementSibling;
        handleFiles(input.files, grid);
        input.value = '';
    }

    window.removeBlock = function (btn) {
        btn.closest('.editor-block').remove();
    }

    window.generateAIContent = async function () {
        const btn = document.getElementById('aiContentBtn');
        const originalText = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = '<i data-lucide="loader-2" class="animate-spin"></i> 생성 중...';
        await new Promise(r => setTimeout(r, 1500));

        // 1. Basic Info
        const productName = document.querySelector('input[placeholder="고객에게 보여질 매력적인 상품명을 입력하세요"]');
        if (productName) productName.value = "[럭셔리] 다낭/호이안 3박 5일 프리미엄 골프 투어";

        const countryCity = document.querySelector('input[placeholder="예: 베트남 다낭"]');
        if (countryCity) countryCity.value = "베트남 다낭";

        const departureCity = document.getElementById('departureCity');
        if (departureCity) departureCity.value = "ICN/GMP";

        // 2. Hotel Info
        const hotelName = document.querySelector('input[placeholder="예: 다낭 메리어트 리조트"]');
        if (hotelName) hotelName.value = "하얏트 리젠시 다낭 리조트 앤 스파";

        // 4. Policies
        fillAllPolicies();

        // 5. Timeline
        const daysInput = document.getElementById('daysInput');
        if (daysInput.value < 3) { daysInput.value = 3; generateTimeline(); } else {
            generateTimeline();
        }
        
        const days = document.querySelectorAll('.day-section');
        if (days.length > 0) {
            // Day 1
            const day1Btn = days[0].querySelector('.add-event-btn');
            if (day1Btn) {
                addEvent(day1Btn, 'flight');
                addEvent(day1Btn, 'hotel');
            }

            // Day 2
            if (days.length > 1) {
                const day2Btn = days[1].querySelector('.add-event-btn');
                if (day2Btn) {
                    addEvent(day2Btn, 'golf');
                    addEvent(day2Btn, 'meal');
                    addEvent(day2Btn, 'sightseeing');
                }
            }
        }

        // 6. Editor Blocks
        const mockData = [
            "이 여행 상품은 베트남 다낭의 아름다운 해변과 역사적인 명소를 모두 즐길 수 있는 최고의 패키지입니다.",
            "첫째 날에는 다낭 국제공항에 도착하여 가이드 미팅 후 호텔로 이동합니다.",
            "둘째 날은 명문 골프장인 다낭 CC에서 18홀 라운딩을 즐깁니다.",
            "셋째 날은 호이안 올드타운을 방문하고 몽고메리 링크스에서 라운딩을 합니다.",
            "마지막 날에는 롯데마트에서 기념품을 쇼핑하고 공항으로 이동합니다."
        ];

        editorBlocks.innerHTML = '';
        mockData.forEach(text => addTextBlock(text));

        btn.disabled = false;
        btn.innerHTML = originalText;
        lucide.createIcons();
    }

    window.formatPrice = function (input) {
        let value = input.value.replace(/\D/g, '');
        if (value) {
            value = parseInt(value).toLocaleString();
        }
        input.value = value;
    }

    window.toggleChip = function (label) {
        const container = label.parentElement;
        container.querySelectorAll('label').forEach(l => {
            l.classList.remove('bg-white', 'text-slate-800', 'shadow-sm');
            l.classList.add('text-slate-500');
        });
        label.classList.add('bg-white', 'text-slate-800', 'shadow-sm');
        label.classList.remove('text-slate-500');
    }

    window.fillAllPolicies = function () {
        document.getElementById('safetyPolicy').value = "- 현지 기상 악화 시 일정이 변경될 수 있습니다.\\n- 개인 소지품 분실에 주의하시기 바랍니다.";
        document.getElementById('refundPolicy').value = "- 여행 개시 30일 전까지 통보 시: 계약금 환급\\n- 여행 개시 20일 전까지 통보 시: 여행요금의 10 % 배상";
    }

    window.handleCardImages = function (input) {
        const grid = input.closest('.grid-cols-1').querySelector('.card-image-grid');
        handleFiles(input.files, grid);
        input.value = '';
    }

    window.showTypeSelector = function (btn) {
        const selector = btn.nextElementSibling;
        selector.classList.remove('hidden');
        document.querySelectorAll('.type-selector').forEach(el => {
            if (el !== selector) el.classList.add('hidden');
        });
    }

    document.addEventListener('click', function (e) {
        if (!e.target.closest('.relative')) {
            document.querySelectorAll('.type-selector').forEach(el => el.classList.add('hidden'));
        }
    });

    window.addEvent = function (btn, type) {
        const selector = btn.closest('.type-selector');
        const daySection = btn.closest('.day-section');
        const eventList = daySection.querySelector('.event-list');

        let card;
        let icon;
        let contentArea;

        // Create basic card structure
        const div = document.createElement('div');
        div.innerHTML = `
            <div class="event-card group bg-white border border-slate-200 rounded-xl p-4 hover:border-indigo-300 transition-all shadow-sm cursor-move"
                data-category="${type}">
                <div class="flex items-start gap-3">
                    <div
                        class="w-8 h-8 rounded-lg bg-indigo-50 flex items-center justify-center text-lg event-icon">
                    </div>
                    <div class="flex-1 event-content"></div>
                    <button type="button" onclick="this.closest('.event-card').remove()"
                        class="text-slate-300 hover:text-red-500 transition-colors"><i data-lucide="trash-2"
                            size="16"></i></button>
                </div>
            </div>
            `;
        card = div.firstElementChild;
        const iconDiv = card.querySelector('.event-icon');
        contentArea = card.querySelector('.event-content');

        if (type === 'flight') {
            icon = '✈️';
            contentArea.appendChild(document.getElementById('flightContent').content.cloneNode(true));
        } else if (type === 'hotel') {
            icon = '🏨';
            contentArea.appendChild(document.getElementById('hotelContent').content.cloneNode(true));
            card.dataset.category = 'hotel';
            const hotelGrid = contentArea.querySelector('.card-image-grid');
            if (hotelGrid) {
                new Sortable(hotelGrid, { animation: 150, ghostClass: 'bg-indigo-50' });
            }
        } else if (type === 'golf') {
            icon = '⛳';
            contentArea.appendChild(document.getElementById('golfContent').content.cloneNode(true));
            card.dataset.category = 'golf';
            const golfGrid = contentArea.querySelector('.card-image-grid');
            if (golfGrid) {
                new Sortable(golfGrid, { animation: 150, ghostClass: 'bg-indigo-50' });
            }
        } else if (type === 'sightseeing') {
            icon = '📷';
            contentArea.appendChild(document.getElementById('hybridContent').content.cloneNode(true));
            card.dataset.category = 'sightseeing';
        } else if (type === 'meal') {
            icon = '🍽️';
            contentArea.innerHTML = `<input type="text"
                class="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:border-indigo-500 outline-none"
                placeholder="식사 장소/메뉴 입력">`;
        } else {
            icon = '📝';
            contentArea.innerHTML = `<input type="text"
                class="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:border-indigo-500 outline-none"
                placeholder="일정 내용 입력">`;
        }

        iconDiv.textContent = icon;
        eventList.appendChild(card);
        selector.classList.add('hidden');
        lucide.createIcons();
    }

    window.toggleMode = function (btn) {
        const container = btn.closest('.event-content');
        const searchMode = container.querySelector('.mode-search');
        const manualMode = container.querySelector('.mode-manual');

        if (searchMode.classList.contains('hidden')) {
            searchMode.classList.remove('hidden');
            manualMode.classList.add('hidden');
        } else {
            searchMode.classList.add('hidden');
            manualMode.classList.remove('hidden');
        }
    }

    window.handleSearch = function (input) {
        const query = input.value.toLowerCase();
        const container = input.closest('.mode-search');
        const resultsDiv = container.querySelector('.search-results');
        const card = input.closest('.event-card');
        const category = card.dataset.category || 'sightseeing'; // default

        if (query.length < 1) { resultsDiv.classList.add('hidden'); return; } const data = dummyDB[category] ||
            []; const filtered = data.filter(item => item.name.toLowerCase().includes(query));

        resultsDiv.innerHTML = filtered.map(item => `
                <div class="p-3 hover:bg-indigo-50 cursor-pointer flex gap-3 items-center transition-colors"
                    onclick='selectItem(this, ${JSON.stringify(item)})'>
                    <img src="${item.image}" class="w-10 h-10 rounded object-cover bg-slate-200" />
                    <div>
                        <div class="font-bold text-slate-800 text-sm">${item.name}</div>
                        <div class="text-xs text-slate-500">${item.desc}</div>
                    </div>
                </div>
                `).join('');

        resultsDiv.classList.remove('hidden');
    }

    window.selectItem = function (el, item) {
        const container = el.closest('.mode-search');
        const input = container.querySelector('.search-input');
        const resultsDiv = container.querySelector('.search-results');
        const preview = container.querySelector('.preview-card');

        input.value = item.name;
        resultsDiv.classList.add('hidden');

        preview.querySelector('img').src = item.image;
        preview.querySelector('.preview-name').textContent = item.name;
        preview.querySelector('.preview-desc').textContent = item.desc;
        preview.querySelector('.preview-loc span').textContent = item.loc;
        preview.classList.remove('hidden');
    }

    window.handleQuotationFile = async function(input) {
        const file = input.files[0];
        if (!file) return;
        
        const label = input.closest('label');
        const icon = label.querySelector('i');
        const span = label.querySelector('span');
        const originalIcon = icon.getAttribute('data-lucide');
        const originalText = span.textContent;
        
        icon.setAttribute('data-lucide', 'loader-2');
        icon.classList.add('animate-spin');
        span.textContent = '분석 중...';
        lucide.createIcons();
        
        await new Promise(r => setTimeout(r, 2000));
        
        // Removed generateAIContent() call as per revert request
        // await generateAIContent();
        
        icon.setAttribute('data-lucide', originalIcon);
        icon.classList.remove('animate-spin');
        span.textContent = originalText;
        lucide.createIcons();
        
        // Clear input
        input.value = '';
    }

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
        
        input.value = '';
    }

    window.handleLandFile = function (input) {
        const file = input.files[0];
        if (file) {
            const container = document.getElementById('landFileInfo');
            const name = document.getElementById('landFileName');
            name.textContent = file.name;
            container.classList.remove('hidden');
        }
    }

    window.clearLandFile = function () {
        document.getElementById('landFile').value = '';
        document.getElementById('landFileInfo').classList.add('hidden');
    }

    window.generateTimeline = function () {
        console.log("Generating Timeline...");
        const daysInput = document.getElementById('daysInput');
        const container = document.getElementById('timelineContainer');
        const days = parseInt(daysInput.value) || 3;

        console.log("Days:", days);
        console.log("Container:", container);

        if (!container) {
            console.error("Timeline container not found!");
            return;
        }

        container.innerHTML = '';

        for (let i = 1; i <= days; i++) {
            const daySection = document.createElement('div');
            daySection.className = 'day-section border border-slate-200 rounded-xl p-6 bg-slate-50';
            daySection.innerHTML = `
                    <div class="flex items-center justify-between mb-4">
                        <h4 class="font-bold text-slate-700 text-lg">Day ${i}</h4>
                        <div class="relative">
                            <button type="button" onclick="showTypeSelector(this)"
                                class="add-event-btn flex items-center gap-2 px-3 py-2 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 text-sm font-bold text-slate-600 transition-colors shadow-sm">
                                <i data-lucide="plus" size="16"></i> 일정 추가
                            </button>
                            <div class="type-selector hidden absolute right-0 top-full mt-2 w-48 bg-white border border-slate-200 rounded-xl shadow-xl z-20 p-2 grid grid-cols-2 gap-2">
                                <button type="button" onclick="addEvent(this, 'flight')" class="flex flex-col items-center justify-center p-3 hover:bg-indigo-50 rounded-lg text-slate-600 hover:text-indigo-600 transition-colors gap-1">
                                    <span class="text-xl">✈️</span><span class="text-xs font-bold">항공</span>
                                </button>
                                <button type="button" onclick="addEvent(this, 'hotel')" class="flex flex-col items-center justify-center p-3 hover:bg-indigo-50 rounded-lg text-slate-600 hover:text-indigo-600 transition-colors gap-1">
                                    <span class="text-xl">🏨</span><span class="text-xs font-bold">숙박</span>
                                </button>
                                <button type="button" onclick="addEvent(this, 'golf')" class="flex flex-col items-center justify-center p-3 hover:bg-indigo-50 rounded-lg text-slate-600 hover:text-indigo-600 transition-colors gap-1">
                                    <span class="text-xl">⛳</span><span class="text-xs font-bold">골프</span>
                                </button>
                                <button type="button" onclick="addEvent(this, 'sightseeing')" class="flex flex-col items-center justify-center p-3 hover:bg-indigo-50 rounded-lg text-slate-600 hover:text-indigo-600 transition-colors gap-1">
                                    <span class="text-xl">📷</span><span class="text-xs font-bold">관광</span>
                                </button>
                                <button type="button" onclick="addEvent(this, 'meal')" class="flex flex-col items-center justify-center p-3 hover:bg-indigo-50 rounded-lg text-slate-600 hover:text-indigo-600 transition-colors gap-1">
                                    <span class="text-xl">🍽️</span><span class="text-xs font-bold">식사</span>
                                </button>
                                <button type="button" onclick="addEvent(this, 'other')" class="flex flex-col items-center justify-center p-3 hover:bg-indigo-50 rounded-lg text-slate-600 hover:text-indigo-600 transition-colors gap-1">
                                    <span class="text-xl">📝</span><span class="text-xs font-bold">기타</span>
                                </button>
                            </div>
                        </div>
                    </div>
                    <div class="event-list space-y-3 min-h-[50px]" data-day="${i}"></div>
                `;
            container.appendChild(daySection);

            new Sortable(daySection.querySelector('.event-list'), {
                group: 'shared',
                animation: 150,
                ghostClass: 'bg-indigo-50',
                handle: '.event-card'
            });
        }
        lucide.createIcons();
    }

    const datePicker = document.getElementById('dateRangePicker');
    if (datePicker) {
        datePicker.addEventListener('click', function() {
            const calendarSection = document.getElementById('calendarGrid').closest('section');
            if (calendarSection) {
                calendarSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
                calendarSection.classList.add('ring-2', 'ring-indigo-500', 'transition-all', 'duration-500');
                setTimeout(() => calendarSection.classList.remove('ring-2', 'ring-indigo-500'), 1000);
            }
        });
    }

    // Initialize
    renderCalendar();
    generateTimeline();
</script>
{% endblock %}"""

    content = content[:start_idx] + new_script + content[end_idx+len(script_end_marker):]
    print("Replaced script block with clean version")
else:
    print("Could not find script block")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
