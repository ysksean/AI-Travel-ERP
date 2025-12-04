import React, { useState } from 'react';
import { CreditCard, Search, Filter, Plus, Download, FileText, CheckCircle, AlertCircle } from 'lucide-react';
import PageContainer from '../components/PageContainer';

const FinancePage = () => {
    const [transactions, setTransactions] = useState([
        { id: 'T001', date: '2025-10-01', sender: '홍길동', amount: 500000, type: '입금', status: '확인완료', related: 'R20251005-001' },
        { id: 'T002', date: '2025-10-02', sender: '김철수', amount: 1200000, type: '입금', status: '미확인', related: '-' },
        { id: 'T003', date: '2025-10-03', sender: '하나투어', amount: -3500000, type: '출금', status: '지급완료', related: '랜드사 송금' },
    ]);

    return (
        <PageContainer className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-slate-900">정산 관리 (Finance)</h1>
                    <p className="text-slate-500 text-sm mt-1">입출금 내역 및 미수금을 관리합니다.</p>
                </div>
                <div className="flex gap-3">
                    <button className="px-4 py-2 text-slate-600 font-medium hover:bg-slate-100 rounded-lg transition-colors flex items-center gap-2">
                        <Download size={18} />
                        엑셀 다운로드
                    </button>
                    <button className="px-5 py-2.5 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-medium shadow-sm flex items-center gap-2">
                        <Plus size={18} />
                        내역 수기 등록
                    </button>
                </div>
            </div>

            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                    <p className="text-slate-500 text-sm">이번 달 입금</p>
                    <h3 className="text-2xl font-bold text-blue-600 mt-2">₩15,400,000</h3>
                </div>
                <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                    <p className="text-slate-500 text-sm">이번 달 출금</p>
                    <h3 className="text-2xl font-bold text-red-600 mt-2">₩8,200,000</h3>
                </div>
                <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                    <p className="text-slate-500 text-sm">현재 미수금</p>
                    <h3 className="text-2xl font-bold text-amber-600 mt-2">₩2,500,000</h3>
                </div>
            </div>

            {/* Transaction List */}
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                <div className="p-4 border-b border-slate-200 flex items-center gap-4">
                    <div className="relative flex-1">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                        <input
                            type="text"
                            placeholder="입금자명, 금액 검색..."
                            className="w-full pl-10 pr-4 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                        />
                    </div>
                    <select className="px-3 py-2 border border-slate-200 rounded-lg text-sm">
                        <option>전체 내역</option>
                        <option>입금</option>
                        <option>출금</option>
                    </select>
                </div>

                <table className="w-full text-sm">
                    <thead className="bg-slate-50 border-b border-slate-200">
                        <tr>
                            <th className="px-6 py-4 text-left font-medium text-slate-500">날짜</th>
                            <th className="px-6 py-4 text-left font-medium text-slate-500">구분</th>
                            <th className="px-6 py-4 text-left font-medium text-slate-500">내용/입금자</th>
                            <th className="px-6 py-4 text-right font-medium text-slate-500">금액</th>
                            <th className="px-6 py-4 text-left font-medium text-slate-500">관련 예약</th>
                            <th className="px-6 py-4 text-left font-medium text-slate-500">상태</th>
                            <th className="px-6 py-4 text-right font-medium text-slate-500">증빙</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                        {transactions.map((t) => (
                            <tr key={t.id} className="hover:bg-slate-50 transition-colors">
                                <td className="px-6 py-4 text-slate-500">{t.date}</td>
                                <td className="px-6 py-4">
                                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${t.type === '입금' ? 'bg-blue-100 text-blue-700' : 'bg-red-100 text-red-700'
                                        }`}>
                                        {t.type}
                                    </span>
                                </td>
                                <td className="px-6 py-4 font-medium text-slate-900">{t.sender}</td>
                                <td className={`px-6 py-4 text-right font-mono font-medium ${t.type === '입금' ? 'text-blue-600' : 'text-red-600'
                                    }`}>
                                    {t.amount > 0 ? '+' : ''}{t.amount.toLocaleString()}
                                </td>
                                <td className="px-6 py-4 text-slate-500 text-xs">{t.related}</td>
                                <td className="px-6 py-4">
                                    {t.status === '확인완료' || t.status === '지급완료' ? (
                                        <div className="flex items-center gap-1 text-green-600 text-xs font-medium">
                                            <CheckCircle size={14} /> {t.status}
                                        </div>
                                    ) : (
                                        <div className="flex items-center gap-1 text-amber-600 text-xs font-medium">
                                            <AlertCircle size={14} /> {t.status}
                                        </div>
                                    )}
                                </td>
                                <td className="px-6 py-4 text-right">
                                    <button className="text-slate-400 hover:text-slate-600">
                                        <FileText size={18} />
                                    </button>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </PageContainer>
    );
};

export default FinancePage;
