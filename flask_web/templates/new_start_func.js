window.startAIAnalysis = async function () {
    console.log("startAIAnalysis called");
    if (!selectedProductFile) {
        alert("상품 견적서 파일을 먼저 선택해주세요.");
        return;
    }

    // UI: 로딩 오버레이 표시
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) overlay.classList.remove('hidden');

    // 카드 UI 업데이트 (보조)
    const infoDiv = document.getElementById('quotationFileInfo');
    const statusText = document.getElementById('quotationStatusText');
    const statusIcon = document.getElementById('quotationStatusIcon');

    if (infoDiv) {
        infoDiv.classList.remove('bg-white', 'border-slate-200');
        infoDiv.classList.add('bg-indigo-50', 'border-indigo-100', 'animate-pulse');
        statusText.textContent = "AI 분석 중...";
        statusText.className = "text-xs text-indigo-600 font-bold";
        statusIcon.setAttribute('data-lucide', 'loader-2');
        statusIcon.classList.add('animate-spin', 'text-indigo-600');
        statusIcon.classList.remove('text-slate-400');
        if (window.lucide) lucide.createIcons();
    }

    try {
        const formData = new FormData();
        formData.append('product_file', selectedProductFile);
        if (selectedPriceFile) {
            formData.append('price_file', selectedPriceFile);
        }

        console.log("Sending request to /product/analyze");
        const response = await fetch('/product/analyze', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        console.log("Response:", result);

        if (!response.ok) throw new Error(result.message || "Server Error");

        if (result.status === 'success' || result.data) {
            const data = result.data || result;
            console.log("✅ Mapped DB Data:", data);

            // === [DB Schema <-> HTML Mapping Start] ===
            // 1. 기본 및 위치 정보
            if (data.product_info?.product_name) setValByPlaceholder("고객에게 보여질 매력적인 상품명을 입력하세요", data.product_info.product_name);

            const country = data.location_info?.country || '';
            const city = data.location_info?.city || '';
            if (country || city) setValByPlaceholder("예: 베트남 다낭", `${country} ${city}`.trim());

            if (data.location_info?.departure_port) {
                const portSelect = document.getElementById('departureCity');
                if (portSelect) portSelect.value = data.location_info.departure_port;
            }

            if (data.basic_info?.product_type) {
                const pType = data.basic_info.product_type.includes("국내") ? "domestic" : "overseas";
                const radio = document.querySelector(`input[name="product_type"][value="${pType}"]`);
                if (radio) {
                    radio.checked = true;
                    toggleChip(radio.closest('label'));
                }
            }

            // 2. 가격 및 일정
            if (data.price_info && data.price_info.length > 0) {
                const firstPrice = data.price_info[0];
                if (firstPrice.night_count) document.getElementById('nightsInput').value = firstPrice.night_count;
                if (firstPrice.day_count) document.getElementById('daysInput').value = firstPrice.day_count;

                const priceInputs = document.querySelectorAll('.price-input');
                if (priceInputs.length >= 2 && firstPrice.price_adult) {
                    const price = parseInt(firstPrice.price_adult);
                    priceInputs[0].value = price.toLocaleString();
                    priceInputs[1].value = Math.floor(price * 0.9).toLocaleString();
                }

                window.currentPriceData = data.price_info;

                // 달력 자동 이동 (첫 출발일 기준)
                if (firstPrice.departure_date) {
                    const [y, m, d] = firstPrice.departure_date.split('-').map(Number);
                    window.currentDate = new Date(y, m - 1, 1);
                }
                renderCalendar(window.currentDate, data.price_info);
            }

            // 3. 호텔 정보
            if (data.hotels && data.hotels.length > 0) {
                const hotel = data.hotels[0];
                setValByPlaceholder("예: 다낭 메리어트 리조트", hotel.name_kr || '');
                setValByPlaceholder("호텔 특징 및 수영장, 레스토랑 등 부대시설 입력", hotel.description || '');
                if (hotel.meta_info) {
                    setValByPlaceholder("15:00 / 11:00", hotel.meta_info.check_in_out);
                    setValByPlaceholder("웹사이트/전화번호", hotel.meta_info.website);
                }
                document.getElementById('hotelInfoSection').classList.remove('hidden');
            }

            // 4. 골프장 정보
            if (data.golf_courses && data.golf_courses.length > 0) {
                const golf = data.golf_courses[0];
                const golfNameInputs = document.querySelectorAll('input');
                for (let inp of golfNameInputs) {
                    if (inp.closest('#golfInfoSection')) {
                        if (inp.placeholder.includes("골프장명")) inp.value = golf.name_kr || '';
                        if (inp.placeholder.includes("운영 정보")) inp.value = golf.operation_info || '';
                    }
                }
                document.getElementById('golfInfoSection').classList.remove('hidden');
            }

            // 5. 상세 조건
            if (data.details) {
                if (data.details.inclusions) setValByPlaceholder("- 왕복 항공권", data.details.inclusions.join('\n'));
                if (data.details.exclusions) setValByPlaceholder("- 가이드/기사 경비", data.details.exclusions.join('\n'));

                const notes = [];
                if (data.details.others) notes.push(data.details.others);
                if (data.details.special_notes) notes.push(data.details.special_notes.join(', '));
                if (notes.length > 0) setValByPlaceholder("기타 특이사항", notes.join("\n"));
            }

            // 6. 항공 정보
            if (data.flight_info) {
                const fullFlightStr = `${data.flight_info.airline || ''} ${data.flight_info.flight_number || ''} ${data.flight_info.departure_time || ''} 출발`.trim();
                setValByPlaceholder("예: 대한항공 KE463", fullFlightStr);
            }

            // 7. AI 콘텐츠
            if (data.ai_content?.body_text) {
                const editor = document.getElementById('editorBlocks');
                if (editor) {
                    editor.innerHTML = '';
                    const paragraphs = data.ai_content.body_text.split('\n');
                    paragraphs.forEach(p => {
                        if (p.trim()) addTextBlock(p.trim());
                    });
                }
            }

            // 성공 알림 UI
            if (infoDiv) {
                infoDiv.classList.remove('animate-pulse', 'bg-indigo-50', 'border-indigo-100');
                infoDiv.classList.add('bg-green-50', 'border-green-100');
                statusText.textContent = "분석 완료!";
                statusText.className = "text-xs text-green-600 font-bold";
                statusIcon.setAttribute('data-lucide', 'check-circle');
                statusIcon.classList.remove('animate-spin', 'text-indigo-600');
                statusIcon.classList.add('text-green-600');
                if (window.lucide) lucide.createIcons();
            }

            // 오버레이 숨김
            if (overlay) overlay.classList.add('hidden');

            // alert("AI 분석이 완료되었습니다!");

        } else {
            throw new Error("데이터 분석 결과가 비어있습니다.");
        }

    } catch (error) {
        console.error(error);
        alert("분석 중 오류가 발생했습니다: " + error.message);

        // 오버레이 숨김
        if (overlay) overlay.classList.add('hidden');

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
