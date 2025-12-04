import React, { useState } from 'react';
import { Search, Filter, Calendar, MessageSquare, MoreHorizontal, CreditCard } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import PageContainer from '../components/PageContainer';

const ReservationListPage = () => {
    const navigate = useNavigate();
    const [reservations] = useState([
        { id: 'RES-2024-001', customer: '김철수', phone: '010-1234-5678', product: '일본 오사카 3박 4일', date: '2024-04-15', pax: 2, status: '확정' },
        { id: 'RES-2024-002', customer: '이영희', phone: '010-9876-5432', product: '베트남 다낭 4박 6일', date: '2024-05-01', pax: 4, status: '대기' },
        { id: 'RES-2024-003', customer: '박지민', phone: '010-5555-4444', product: '유럽 서유럽 4국 9일', date: '2024-06-10', pax: 1, status: '취소' },
    ]);

    const getStatusColor = (status) => {
        switch (status) {
            case '확정': return 'bg-green-100 text-green-700';
            case '대기': return 'bg-yellow-100 text-yellow-700';
            case '취소': return 'bg-red-100 text-red-700';
            default: return 'bg-slate-100 text-slate-700';
        }
    };

    return (
        <PageContainer>
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h1 className="text-2xl font-bold text-slate-900">예약 관리</h1>
                    <p className="text-slate-500 text-sm mt-1">전체 예약 현황을 조회하고 관리하세요.</p>
                </div>
                <button className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-medium shadow-sm shadow-primary-200">
                    + 예약 등록
                </button>
            </div>

            {/* Global Search & Filter */}
            <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex flex-col sm:flex-row gap-4 items-center justify-between mb-6">
                <div className="relative w-full sm:w-96">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                    <input
                        type="text"
                        placeholder="예약번호, 고객명, 연락처 검색..."
                        className="w-full pl-10 pr-4 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                    />
                </div>
                <button className="p-2 border border-slate-200 rounded-lg hover:bg-slate-50 text-slate-600">
                    <Filter size={18} />
                </button>
            </div>

            {/* Reservation List */}
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                <div className="overflow-x-auto">
                    <table className="w-full text-sm text-left">
                        <thead className="bg-slate-50 text-slate-500 font-medium border-b border-slate-200">
                            <tr>
                                <th className="px-6 py-4">상태</th>
                                <th className="px-6 py-4">예약번호</th>
                                <th className="px-6 py-4">예약자</th>
                                <th className="px-6 py-4">상품명</th>
                                <th className="px-6 py-4">출발일</th>
                                <th className="px-6 py-4">인원</th>
                                <th className="px-6 py-4 text-right">관리</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                            {reservations.map((res) => (
                                <tr
                                    key={res.id}
                                    onClick={() => navigate(`/reservations/${res.id}`)}
                                    className="hover:bg-slate-50 transition-colors cursor-pointer group"
                                >
                                    <td className="px-6 py-4">
                                        <div className="flex items-center gap-2">
                                            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(res.status)}`}>
                                                {res.status}
                                            </span>
                                            {res.status === '대기' && (
                                                <button
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        alert(`[알림톡 발송]\n${res.customer} 고객님께 결제 요청 메시지가 발송되었습니다.`);
                                                    }}
                                                    className="inline-flex items-center gap-1 px-2 py-0.5 bg-primary-50 text-primary-700 text-xs font-medium rounded hover:bg-primary-100 transition-colors"
                                                >
                                                    <CreditCard size={12} />
                                                    결제요청
                                                </button>
                                            )}
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 font-mono text-slate-500">{res.id}</td>
                                    <td className="px-6 py-4">
                                        <div className="flex flex-col">
                                            <span className="font-medium text-slate-900">{res.customer}</span>
                                            <span className="text-xs text-slate-400">{res.phone}</span>
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 text-slate-600 max-w-xs truncate">{res.product}</td>
                                    <td className="px-6 py-4 text-slate-500">
                                        <div className="flex items-center gap-1.5">
                                            <Calendar size={14} />
                                            {res.date}
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 text-slate-600">{res.pax}명</td>
                                    <td className="px-6 py-4 text-right" onClick={(e) => e.stopPropagation()}>
                                        <div className="flex items-center justify-end gap-2">
                                            <button className="p-1.5 text-slate-400 hover:text-primary-600 hover:bg-primary-50 rounded transition-colors" title="상담 메모">
                                                <MessageSquare size={18} />
                                            </button>
                                            <button className="p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded transition-colors">
                                                <MoreHorizontal size={18} />
                                            </button>
                                        </div>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </PageContainer>
    );
};

export default ReservationListPage;
