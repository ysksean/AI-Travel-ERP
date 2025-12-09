        } catch (error) {
    console.error(error);
    alert("분석 중 오류가 발생했습니다: " + error.message);

    // 오버레이 숨김
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) overlay.classList.add('hidden');

    const infoDiv = document.getElementById('quotationFileInfo');
    const statusText = document.getElementById('quotationStatusText');
    const statusIcon = document.getElementById('quotationStatusIcon');

    if (infoDiv) {
        infoDiv.classList.remove('animate-pulse', 'bg-indigo-50', 'border-indigo-100');
        infoDiv.classList.add('bg-red-50', 'border-red-100');
        statusText.textContent = "오류 발생";
        statusText.className = "text-xs text-red-600 font-bold";
        statusIcon.setAttribute('data-lucide', 'alert-circle');
        statusIcon.classList.remove('animate-spin', 'text-indigo-600');
        statusIcon.classList.add('text-red-600');
        if (window.lucide) lucide.createIcons();
    }
}
    }

// Helper: Placeholder 텍스트로 input 찾아서 값 설정
function setValByPlaceholder(placeholderPart, value) {
    if (!value) return;
    const inputs = document.querySelectorAll('input, textarea');
    for (const el of inputs) {
        if (el.placeholder && el.placeholder.includes(placeholderPart)) {
            el.value = value;
            return;
        }
    }
}

// ==========================================
// 2. 달력 (일정 시각화)
// ==========================================
window.currentDate = new Date();
window.currentPriceData = [];

window.renderCalendar = function (baseDate = window.currentDate, priceData = window.currentPriceData) {
    const year = baseDate.getFullYear();
    const month = baseDate.getMonth();

    const title = document.getElementById('calendarTitle');
    if (title) title.textContent = `${year}년 ${month + 1}월`;

    const firstDay = new Date(year, month, 1);
    const lastDay = new Date(year, month + 1, 0);
    const grid = document.getElementById('calendarGrid');
    if (!grid) return;
    grid.innerHTML = '';

    // 빈 칸 채우기
    for (let i = 0; i < firstDay.getDay(); i++) {
        grid.appendChild(document.createElement('div'));
    }

    // 날짜 채우기
    for (let d = 1; d <= lastDay.getDate(); d++) {
        const dateStr = `${year}-${String(month + 1).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
        const cell = document.createElement('div');
        cell.className = 'aspect-square flex flex-col items-center justify-center rounded-lg text-sm cursor-pointer hover:bg-slate-50 transition-colors border border-transparent';

        const numSpan = document.createElement('span');
        numSpan.textContent = d;
        cell.appendChild(numSpan);

        // DB Price 매핑
        if (priceData && priceData.length > 0) {
            const info = priceData.find(p => p.departure_date === dateStr);
            if (info) {
                cell.classList.add('bg-indigo-50', 'border-indigo-100');
                cell.querySelector('span').classList.add('font-bold', 'text-indigo-700');

                // 출발 가능 표시 (점)
                const priceDot = document.createElement('div');
                priceDot.className = 'w-1.5 h-1.5 bg-indigo-500 rounded-full mt-1';
                cell.appendChild(priceDot);

                // 툴팁
                if (info.price_adult) {
                    cell.title = `${Number(info.price_adult).toLocaleString()}원`;
                }
            }
        }
        grid.appendChild(cell);
    }
}

window.changeMonth = function (delta) {
    window.currentDate.setMonth(window.currentDate.getMonth() + delta);
    renderCalendar(window.currentDate, window.currentPriceData);
}
