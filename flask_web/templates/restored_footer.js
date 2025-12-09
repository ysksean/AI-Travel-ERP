// 3. UI 유틸리티 (기존 기능 유지)
// ==========================================

// 타임라인 생성 및 일정 관리
const dummyDB = {
    hotel: [
        { name: "하얏트 리젠시 다낭", image: "https://images.unsplash.com/photo-1566073771259-6a8506099945?auto=format&fit=crop&q=80&w=100", desc: "5성급 • 논누옥 비치", loc: "다낭, 베트남" },
        { name: "빈펄 리조트 & 스파", image: "https://images.unsplash.com/photo-1582719508461-905c673771fd?auto=format&fit=crop&q=80&w=100", desc: "5성급 • 프라이빗 풀빌라", loc: "호이안, 베트남" },
    ],
    golf: [
        { name: "다낭 CC", image: "https://images.unsplash.com/photo-1587174486073-ae5e5cff23aa?auto=format&fit=crop&q=80&w=100", desc: "18홀 • 듄스 코스", loc: "다낭" },
    ],
    sightseeing: [
        { name: "바나힐 투어", image: "https://images.unsplash.com/photo-1559592413-7cec4d0cae2b?auto=format&fit=crop&q=80&w=100", desc: "테마파크 • 케이블카", loc: "다낭" },
    ]
};

window.resetItinerary = function () {
    if (confirm('정말로 일정을 초기화하시겠습니까?')) {
        generateTimeline();
    }
}

window.regenerateItinerary = async function () {
    const btn = document.getElementById('regenerateBtn');
    const originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i data-lucide="loader-2" class="animate-spin" size="14"></i> 생성 중...';
    if (window.lucide) lucide.createIcons();
    await new Promise(r => setTimeout(r, 1000));
    generateTimeline();
    btn.disabled = false;
    btn.innerHTML = originalText;
    if (window.lucide) lucide.createIcons();
}

window.generateTimeline = function () {
    const daysInput = document.getElementById('daysInput');
    const container = document.getElementById('timelineContainer');
    const days = parseInt(daysInput.value) || 3;
    if (!container) return;

    container.innerHTML = '';
    for (let i = 1; i <= days; i++) {
        const template = document.getElementById('daySectionTemplate').content.cloneNode(true);
        template.querySelector('.day-title').textContent = `Day ${i}`;
        template.querySelector('.event-list').dataset.day = i;

        // Sortable 초기화
        new Sortable(template.querySelector('.event-list'), {
            group: 'shared',
            animation: 150,
            ghostClass: 'bg-indigo-50',
            handle: '.event-card'
        });
        container.appendChild(template);
    }
    if (window.lucide) lucide.createIcons();
}

window.addEvent = function (btn, type) {
    const selector = btn.closest('.type-selector');
    const daySection = btn.closest('.day-section');
    const eventList = daySection.querySelector('.event-list');
    const div = document.createElement('div');

    let contentHTML = '';
    let icon = '';

    if (type === 'flight') {
        icon = '✈️';
        contentHTML = document.getElementById('flightContent').innerHTML;
    } else if (type === 'hotel') {
        icon = '🏨';
        contentHTML = document.getElementById('hotelContent').innerHTML;
    } else if (type === 'golf') {
        icon = '⛳';
        contentHTML = document.getElementById('golfContent').innerHTML;
    } else if (type === 'sightseeing') {
        icon = '📷';
        contentHTML = document.getElementById('hybridContent').innerHTML;
    } else if (type === 'meal') {
        icon = '🍽️';
        contentHTML = `<input type="text" class="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:border-indigo-500 outline-none" placeholder="식사 장소/메뉴 입력">`;
    } else {
        icon = '📝';
        contentHTML = `<input type="text" class="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:border-indigo-500 outline-none" placeholder="일정 내용 입력">`;
    }

    div.innerHTML = `
            <div class="event-card group bg-white border border-slate-200 rounded-xl p-4 hover:border-indigo-300 transition-all shadow-sm cursor-move" data-category="${type}">
                <div class="flex items-start gap-3">
                    <div class="w-8 h-8 rounded-lg bg-indigo-50 flex items-center justify-center text-lg event-icon">${icon}</div>
                    <div class="flex-1 event-content">${contentHTML}</div>
                    <button type="button" onclick="this.closest('.event-card').remove()" class="text-slate-300 hover:text-red-500 transition-colors"><i data-lucide="trash-2" size="16"></i></button>
                </div>
            </div>`;

    const card = div.firstElementChild;
    // 내부 이미지 그리드 Sortable 적용
    const imageGrid = card.querySelector('.card-image-grid');
    if (imageGrid) new Sortable(imageGrid, { animation: 150, ghostClass: 'bg-indigo-50' });

    eventList.appendChild(card);
    selector.classList.add('hidden');
    if (window.lucide) lucide.createIcons();
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

// 검색 및 모드 전환
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
    const category = card.dataset.category || 'sightseeing';

    if (query.length < 1) { resultsDiv.classList.add('hidden'); return; }

    const data = dummyDB[category] || [];
    const filtered = data.filter(item => item.name.toLowerCase().includes(query));

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

// 파일 처리 공통
window.handleLandFile = function (input) {
    // handleFileSelect에서 처리하므로 여기서는 UI만 보조하거나 비워둠
    // 하지만 기존 코드 호환성을 위해 남겨둠
}
window.clearLandFile = function () {
    document.getElementById('landFile').value = '';
    document.getElementById('landFileInfo').classList.add('hidden');
}
window.clearQuotationFile = function () {
    document.getElementById('topQuotationInput').value = '';
    document.getElementById('quotationFileInfo').classList.add('hidden');
}

// 대표 이미지 핸들러
window.handleHeroImages = function (input) {
    const grid = document.getElementById('heroImageGrid');
    const files = Array.from(input.files);
    if (files.length === 0) return;

    files.forEach(file => {
        if (!file.type.startsWith('image/')) return;
        const reader = new FileReader();
        reader.onload = function (e) {
            const div = document.createElement('div');
            div.className = 'relative group aspect-square rounded-xl overflow-hidden border border-slate-200';
            div.innerHTML = `
                    <img src="${e.target.result}" class="w-full h-full object-cover">
                    <div class="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2">
                        <button type="button" onclick="this.closest('div.relative').remove()" class="p-2 bg-white/90 rounded-full text-red-500 hover:bg-white transition-colors">
                            <i data-lucide="trash-2" size="16"></i>
                        </button>
                    </div>`;
            grid.appendChild(div);
            if (window.lucide) lucide.createIcons();
        };
        reader.readAsDataURL(file);
    });
    input.value = '';
}

// 카드 이미지 핸들러
window.handleCardImages = function (input) {
    const grid = input.closest('.grid-cols-1').querySelector('.card-image-grid');
    // (간단 구현) 실제로는 FileReader로 미리보기 구현 필요
    input.value = '';
}

// 기타 UI 함수
window.formatPrice = function (input) {
    let value = input.value.replace(/\D/g, '');
    if (value) input.value = parseInt(value).toLocaleString();
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
    document.getElementById('safetyPolicy').value = "- 현지 기상 악화 시 일정이 변경될 수 있습니다.\n- 개인 소지품 분실에 주의하시기 바랍니다.";
    document.getElementById('refundPolicy').value = "- 여행 개시 30일 전까지 통보 시: 계약금 환급\n- 여행 개시 20일 전까지 통보 시: 여행요금의 10 % 배상";
}

// 에디터 관련
const editorBlocks = document.getElementById('editorBlocks');
const textBlockTemplate = document.getElementById('textBlockTemplate');
const imageBlockTemplate = document.getElementById('imageBlockTemplate');
if (editorBlocks && typeof Sortable !== 'undefined') {
    new Sortable(editorBlocks, { animation: 150, handle: '.editor-block', ghostClass: 'bg-indigo-50' });
}

window.addTextBlock = function (content = '') {
    const clone = textBlockTemplate.content.cloneNode(true);
    if (content) clone.querySelector('textarea').value = content;
    editorBlocks.appendChild(clone);
    if (window.lucide) lucide.createIcons();
}
window.addImageBlock = function () {
    const clone = imageBlockTemplate.content.cloneNode(true);
    if (clone.querySelector('.image-grid')) {
        new Sortable(clone.querySelector('.image-grid'), { animation: 150, ghostClass: 'bg-indigo-50' });
    }
    editorBlocks.appendChild(clone);
    if (window.lucide) lucide.createIcons();
}
window.addImagesToBlock = function (input) { /* (기존 로직 유지) */ }
window.removeBlock = function (btn) { btn.closest('.editor-block').remove(); }

// [초기화]
renderCalendar();
generateTimeline();
if (window.lucide) lucide.createIcons();
</script >
    {% endblock %}
