import React, { useState } from 'react';
import {
    User,
    Phone,
    Mail,
    Calendar,
    MapPin,
    CreditCard,
    MessageSquare,
    Send,
    FileText,
    Camera,
    AlertCircle,
    Save,
    ArrowLeft,
    Plus,
    Trash2
} from 'lucide-react';
import { Link, useParams } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import PageContainer from '../components/PageContainer';

const ReservationDetailPage = () => {
    const { id } = useParams();
    const [status, setStatus] = useState('예약확정');
    const [penalty, setPenalty] = useState('');
    const [messages, setMessages] = useState([
        { id: 1, author: '시스템', text: '웹사이트를 통해 예약이 생성되었습니다', time: '2025-10-01 10:00', type: 'system' },
        { id: 2, author: '김철수 (직원)', text: '고객이 기내식 채식 요청하셨습니다.', time: '2025-10-01 14:30', type: 'staff' },
    ]);
    const [newMessage, setNewMessage] = useState('');

    // Rooming List State
    const [rooms, setRooms] = useState([
        { id: 1, type: 'Twin', guests: ['홍길동', '김철수'] },
    ]);

    const handleStatusChange = (e) => {
        setStatus(e.target.value);
        if (e.target.value !== '취소됨') {
            setPenalty('');
        }
    };

    const handleSendMessage = (e) => {
        e.preventDefault();
        if (!newMessage.trim()) return;
        setMessages([...messages, {
            id: Date.now(),
            author: '관리자',
            text: newMessage,
            time: new Date().toLocaleString(),
            type: 'staff'
        }]);
        setNewMessage('');
    };

    const addRoom = () => {
        setRooms([...rooms, { id: Date.now(), type: 'Twin', guests: [''] }]);
    };

    const updateGuest = (roomId, guestIndex, value) => {
        setRooms(rooms.map(room => {
            if (room.id === roomId) {
                const newGuests = [...room.guests];
                newGuests[guestIndex] = value;
                return { ...room, guests: newGuests };
            }
            return room;
        }));
    };

    const removeRoom = (roomId) => {
        setRooms(rooms.filter(room => room.id !== roomId));
    };

    return (
        <PageContainer className="max-w-6xl mx-auto space-y-6 pb-20">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <Link to="/reservations" className="p-2 hover:bg-slate-100 rounded-full text-slate-500 transition-colors">
                        <ArrowLeft size={24} />
                    </Link>
                    <div>
                        <div className="flex items-center gap-3">
                            <h1 className="text-2xl font-bold text-slate-900">예약 {id || 'R20251005-001'}</h1>
                            <span className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${status === '예약확정' ? 'bg-green-100 text-green-800' :
                                    status === '취소됨' ? 'bg-red-100 text-red-800' :
                                        'bg-yellow-100 text-yellow-800'
                                }`}>
                                {status}
                            </span>
                        </div>
                        <p className="text-slate-500 text-sm">일본 프리미엄 골프 투어 (3박4일)</p>
                    </div>
                </div>
                <button className="px-6 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-medium shadow-sm shadow-primary-200 flex items-center gap-2">
                    <Save size={18} />
                    변경사항 저장
                </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left Column: Customer & Booking Info */}
                <div className="lg:col-span-2 space-y-6">

                    {/* Status & Penalty Section */}
                    <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                        <h2 className="text-lg font-bold text-slate-900 mb-4">예약 상태</h2>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-slate-700">현재 상태</label>
                                <select
                                    value={status}
                                    onChange={handleStatusChange}
                                    className="w-full px-3 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-primary-500"
                                >
                                    <option value="입금대기">입금대기</option>
                                    <option value="예약확정">예약확정</option>
                                    <option value="취소됨">취소됨</option>
                                </select>
                            </div>

                            {/* Cancellation Penalty Input - Animated */}
                            <AnimatePresence>
                                {status === '취소됨' && (
                                    <motion.div
                                        initial={{ opacity: 0, height: 0 }}
                                        animate={{ opacity: 1, height: 'auto' }}
                                        exit={{ opacity: 0, height: 0 }}
                                        className="overflow-hidden"
                                    >
                                        <div className="space-y-2">
                                            <label className="text-sm font-medium text-red-600 flex items-center gap-1">
                                                <AlertCircle size={14} />
                                                취소 위약금
                                            </label>
                                            <div className="relative">
                                                <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400">₩</span>
                                                <input
                                                    type="number"
                                                    value={penalty}
                                                    onChange={(e) => setPenalty(e.target.value)}
                                                    placeholder="위약금 금액 입력"
                                                    className="w-full pl-8 pr-4 py-2 border-2 border-red-100 bg-red-50 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-red-500 text-red-900 font-medium placeholder-red-300"
                                                />
                                            </div>
                                            <p className="text-xs text-red-500">수동으로 계산된 위약금을 입력하세요.</p>
                                        </div>
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </div>
                    </div>

                    {/* Customer Info */}
                    <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                        <div className="flex items-center justify-between mb-4">
                            <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                                <User size={20} className="text-slate-400" />
                                고객 정보
                            </h2>
                            <button className="text-sm text-primary-600 font-medium hover:underline flex items-center gap-1">
                                <Camera size={16} /> 여권 OCR 스캔
                            </button>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-slate-700">이름 (한글)</label>
                                <input type="text" defaultValue="홍길동" className="w-full px-3 py-2 border border-slate-200 rounded-lg" />
                            </div>
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-slate-700">이름 (영문)</label>
                                <input type="text" defaultValue="HONG GILDONG" className="w-full px-3 py-2 border border-slate-200 rounded-lg" />
                            </div>
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-slate-700">전화번호</label>
                                <div className="relative">
                                    <Phone className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={16} />
                                    <input type="text" defaultValue="010-1234-5678" className="w-full pl-9 pr-3 py-2 border border-slate-200 rounded-lg" />
                                </div>
                            </div>
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-slate-700">여권번호</label>
                                <input type="text" defaultValue="M12345678" className="w-full px-3 py-2 border border-slate-200 rounded-lg font-mono" />
                            </div>
                        </div>
                    </div>

                    {/* Rooming List */}
                    <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                        <div className="flex items-center justify-between mb-4">
                            <h2 className="text-lg font-bold text-slate-900">룸 배정 (Rooming List)</h2>
                            <button onClick={addRoom} className="text-sm text-primary-600 font-medium hover:underline flex items-center gap-1">
                                <Plus size={16} /> 룸 추가
                            </button>
                        </div>

                        <div className="space-y-4">
                            {rooms.map((room, idx) => (
                                <div key={room.id} className="p-4 border border-slate-200 rounded-lg bg-slate-50 relative group">
                                    <button
                                        onClick={() => removeRoom(room.id)}
                                        className="absolute top-2 right-2 text-slate-400 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity"
                                    >
                                        <Trash2 size={16} />
                                    </button>
                                    <div className="flex items-center gap-4 mb-3">
                                        <span className="font-bold text-slate-700">Room {idx + 1}</span>
                                        <select
                                            className="text-sm border border-slate-200 rounded px-2 py-1 bg-white"
                                            defaultValue={room.type}
                                        >
                                            <option value="Single">Single</option>
                                            <option value="Twin">Twin</option>
                                            <option value="Double">Double</option>
                                            <option value="Triple">Triple</option>
                                        </select>
                                    </div>
                                    <div className="space-y-2">
                                        {room.guests.map((guest, guestIdx) => (
                                            <input
                                                key={guestIdx}
                                                type="text"
                                                placeholder={`투숙객 ${guestIdx + 1}`}
                                                value={guest}
                                                onChange={(e) => updateGuest(room.id, guestIdx, e.target.value)}
                                                className="w-full px-3 py-2 border border-slate-200 rounded bg-white text-sm"
                                            />
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Right Column: CRM Chat */}
                <div className="space-y-6">
                    <div className="bg-white rounded-xl border border-slate-200 shadow-sm flex flex-col h-[600px] sticky top-6">
                        <div className="p-4 border-b border-slate-200 bg-slate-50 rounded-t-xl">
                            <h3 className="font-bold text-slate-900 flex items-center gap-2">
                                <MessageSquare size={18} className="text-slate-500" />
                                상담 메모
                            </h3>
                        </div>

                        <div className="flex-1 overflow-y-auto p-4 space-y-4">
                            {messages.map((msg) => (
                                <div key={msg.id} className={`flex flex-col ${msg.type === 'system' ? 'items-center' : 'items-start'}`}>
                                    {msg.type === 'system' ? (
                                        <span className="text-xs bg-slate-100 text-slate-500 px-2 py-1 rounded-full">{msg.text}</span>
                                    ) : (
                                        <div className="bg-blue-50 p-3 rounded-lg rounded-tl-none max-w-[90%]">
                                            <div className="flex items-center justify-between gap-2 mb-1">
                                                <span className="text-xs font-bold text-blue-700">{msg.author}</span>
                                                <span className="text-[10px] text-blue-400">{msg.time}</span>
                                            </div>
                                            <p className="text-sm text-slate-700">{msg.text}</p>
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>

                        <div className="p-4 border-t border-slate-200">
                            <form onSubmit={handleSendMessage} className="flex gap-2">
                                <input
                                    type="text"
                                    value={newMessage}
                                    onChange={(e) => setNewMessage(e.target.value)}
                                    placeholder="메모 입력..."
                                    className="flex-1 px-3 py-2 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                                />
                                <button type="submit" className="p-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors">
                                    <Send size={18} />
                                </button>
                            </form>
                        </div>
                    </div>
                </div>
            </div>
        </PageContainer>
    );
};

export default ReservationDetailPage;
