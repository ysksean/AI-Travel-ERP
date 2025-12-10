document.addEventListener('DOMContentLoaded', function () {
    const chatToggle = document.getElementById('chatToggle');
    const chatSidebar = document.getElementById('chatSidebar');
    const closeChat = document.getElementById('closeChat');
    const chatRoomList = document.querySelectorAll('.chat-room-item');
    const chatMessagesArea = document.getElementById('chatMessages');
    const currentChatUser = document.getElementById('currentChatUser');
    const currentChatStatus = document.getElementById('currentChatStatus');

    // Toggle Sidebar
    if (chatToggle && chatSidebar) {
        chatToggle.addEventListener('click', () => {
            chatSidebar.classList.toggle('translate-x-full');
        });
    }

    // Close Sidebar
    if (closeChat && chatSidebar) {
        closeChat.addEventListener('click', () => {
            chatSidebar.classList.add('translate-x-full');
        });
    }

    // Mock Chat Data (Pre-written for demo)
    const mockMessages = {
        '홍길동': [
            { sender: 'user', text: '안녕하세요, 가족 여행 패키지 문의드립니다.', time: '오전 10:30' },
            { sender: 'admin', text: '네, 안녕하세요! 여행 인원과 원하시는 날짜가 어떻게 되시나요?', time: '오전 10:32' },
            { sender: 'user', text: '성인 4명이고, 12월 15일 출발 원합니다.', time: '오전 10:33' }
        ],
        '김철수': [
            { sender: 'user', text: '예약 취소 수수료가 어떻게 되나요?', time: '오후 2:15' },
            { sender: 'admin', text: '출발 7일 전까지는 전액 환불 가능합니다.', time: '오후 2:20' }
        ],
        '이영희': [
            { sender: 'user', text: '입금 확인 부탁드립니다.', time: '어제' }
        ]
    };

    // Chat Switching Logic
    chatRoomList.forEach(item => {
        item.addEventListener('click', function () {
            // 1. UI Active State Reset
            chatRoomList.forEach(r => {
                r.classList.remove('bg-indigo-50/60', 'border-indigo-500');
                r.classList.add('border-transparent');
            });

            // 2. Set Active State
            this.classList.remove('border-transparent');
            this.classList.add('bg-indigo-50/60', 'border-indigo-500');

            // 3. Get User Info
            const userName = this.querySelector('.room-name').textContent;

            // 4. Update Header
            if (currentChatUser) currentChatUser.textContent = userName;
            // (Optional) Randomize status
            if (currentChatStatus) {
                const isOnline = Math.random() > 0.5;
                currentChatStatus.innerHTML = isOnline
                    ? `<span class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span> 온라인`
                    : `<span class="w-2 h-2 bg-gray-400 rounded-full"></span> 오프라인`;
                currentChatStatus.className = `text-[11px] ${isOnline ? 'text-green-600' : 'text-gray-500'} flex items-center gap-1.5 font-medium`;
            }

            // 5. Render Messages
            if (chatMessagesArea) {
                chatMessagesArea.innerHTML = ''; // Clear previous

                // Add Date Divider
                const dateDiv = document.createElement('div');
                dateDiv.className = 'flex justify-center my-4';
                dateDiv.innerHTML = `<span class="bg-gray-100 px-4 py-1.5 rounded-full text-[11px] text-gray-500 font-medium shadow-sm">오늘 2023년 12월 10일</span>`;
                chatMessagesArea.appendChild(dateDiv);

                const messages = mockMessages[userName] || [];

                messages.forEach(msg => {
                    const isUser = msg.sender === 'user';
                    const msgWrapper = document.createElement('div');
                    msgWrapper.className = `flex w-full ${isUser ? 'justify-start' : 'justify-end'} mb-4 animate-fade-in-up`;

                    const innerHTML = isUser ? `
                        <div class="flex gap-3 max-w-[85%]">
                            <div class="w-8 h-8 rounded-full bg-slate-200 flex-shrink-0 flex items-center justify-center text-slate-500 text-xs shadow-sm">
                                <i class="fas fa-user"></i>
                            </div>
                            <div class="flex flex-col gap-1">
                                <div class="bg-white border border-gray-100 p-3 rounded-2xl rounded-tl-none shadow-sm text-sm text-gray-800 leading-relaxed">
                                    ${msg.text}
                                </div>
                                <span class="text-[10px] text-gray-400 ml-1">${msg.time}</span>
                            </div>
                        </div>
                    ` : `
                        <div class="flex flex-col gap-1 items-end max-w-[85%]">
                            <div class="bg-indigo-600 p-3 rounded-2xl rounded-tr-none shadow-md text-sm text-white leading-relaxed">
                                ${msg.text}
                            </div>
                            <span class="text-[10px] text-gray-400 mr-1 flex items-center gap-1">
                                ${msg.time} <i class="fas fa-check-double text-indigo-400 text-[8px]"></i>
                            </span>
                        </div>
                    `;

                    msgWrapper.innerHTML = innerHTML;
                    chatMessagesArea.appendChild(msgWrapper);
                });

                // Scroll to bottom
                chatMessagesArea.scrollTop = chatMessagesArea.scrollHeight;
            }
        });
    });

    // Send Message Logic
    const chatInput = chatSidebar.querySelector('textarea');
    // Find the send button (it's the one with the paper-plane icon, usually the last button in that container)
    const sendBtn = chatSidebar.querySelector('button.bg-indigo-600');

    function sendMessage() {
        const message = chatInput.value.trim();
        if (!message) return;

        // 1. Append User Message
        const userMsgHTML = `
            <div class="flex w-full justify-end mb-4 animate-fade-in-up">
                <div class="flex flex-col gap-1 items-end max-w-[85%]">
                    <div class="bg-indigo-600 p-3 rounded-2xl rounded-tr-none shadow-md text-sm text-white leading-relaxed">
                        ${message}
                    </div>
                    <span class="text-[10px] text-gray-400 mr-1 flex items-center gap-1">
                        지금 <i class="fas fa-check text-indigo-400 text-[8px]"></i>
                    </span>
                </div>
            </div>
        `;
        chatMessagesArea.insertAdjacentHTML('beforeend', userMsgHTML);
        chatInput.value = '';
        chatMessagesArea.scrollTop = chatMessagesArea.scrollHeight;

        // 2. Mock Response (Timeout)
        setTimeout(() => {
            const aiMsgHTML = `
                <div class="flex w-full justify-start mb-4 animate-fade-in-up">
                    <div class="flex gap-3 max-w-[85%]">
                        <div class="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500 flex-shrink-0 flex items-center justify-center text-white text-xs shadow-sm">
                            <i class="fas fa-robot"></i>
                        </div>
                        <div class="flex flex-col gap-1">
                            <div class="bg-white border border-gray-100 p-3 rounded-2xl rounded-tl-none shadow-sm text-sm text-gray-800 leading-relaxed">
                                (자동 응답) 죄송합니다. 상담원이 잠시 자리를 비웠습니다. 잠시만 기다려주시면 답변 드리겠습니다.
                            </div>
                            <span class="text-[10px] text-gray-400 ml-1">지금</span>
                        </div>
                    </div>
                </div>
            `;
            chatMessagesArea.insertAdjacentHTML('beforeend', aiMsgHTML);
            chatMessagesArea.scrollTop = chatMessagesArea.scrollHeight;
        }, 1500);
    }

    if (sendBtn) {
        sendBtn.addEventListener('click', sendMessage);
    }

    if (chatInput) {
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }

});
