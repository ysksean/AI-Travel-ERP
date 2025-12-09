window.handleFileSelect = function (type, input) {
    try {
        const file = input.files[0];
        if (!file) return;

        console.log(`File selected: ${type}, ${file.name}`);

        if (type === 'product') {
            selectedProductFile = file;
            // UI 업데이트
            const infoDiv = document.getElementById('quotationFileInfo');
            const fileNameSpan = document.getElementById('quotationFileName');
            const statusText = document.getElementById('quotationStatusText');
            const statusIcon = document.getElementById('quotationStatusIcon');

            if (infoDiv) {
                infoDiv.classList.remove('hidden', 'bg-red-50', 'border-red-100', 'bg-green-50', 'border-green-100', 'animate-pulse');
                infoDiv.classList.add('bg-white', 'border-slate-200');
                fileNameSpan.textContent = file.name;
                statusText.textContent = "분석 대기 중";
                statusText.className = "text-xs text-slate-500 font-bold";
                statusIcon.setAttribute('data-lucide', 'file-text');
                statusIcon.classList.remove('animate-spin', 'text-indigo-600', 'text-green-600', 'text-red-600');
                statusIcon.classList.add('text-slate-400');
                if (window.lucide) lucide.createIcons();
            }
        } else if (type === 'price') {
            selectedPriceFile = file;
            // UI 업데이트
            const infoDiv = document.getElementById('landFileInfo');
            const fileNameSpan = document.getElementById('landFileName');
            if (infoDiv) {
                infoDiv.classList.remove('hidden');
                fileNameSpan.textContent = file.name;
            }
        }
    } catch (e) {
        console.error("File select error:", e);
        alert("파일 선택 중 오류가 발생했습니다: " + e.message);
    }
}
