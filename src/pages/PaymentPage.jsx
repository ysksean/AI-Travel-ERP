import React from 'react';
import { CreditCard, CheckCircle, Clock, Search } from 'lucide-react';
import PageContainer from '../components/PageContainer';

const PaymentPage = () => {
    const payments = [
        { id: 'PAY-2024-001', customer: '김철수', amount: '2,500,000', method: '신용카드', date: '2024-03-15', status: '완료' },
        { id: 'PAY-2024-002', customer: '이영희', amount: '1,800,000', method: '계좌이체', date: '2024-03-14', status: '대기중' },
        { id: 'PAY-2024-003', customer: '박지민', amount: '3,200,000', method: '신용카드', date: '2024-03-14', status: '완료' },
    ];

    const getStatusColor = (status) => {
        switch (status) {
            case '완료': return 'bg-green-100 text-green-700';
            case '대기중': return 'bg-yellow-100 text-yellow-700';
            case '실패': return 'bg-red-100 text-red-700';
            default: return 'bg-slate-100 text-slate-700';
        }
    };

    const getStatusIcon = (status) => {
        switch (status) {
            case '완료': return <CheckCircle size={14} className="text-green-600" />;
            case '대기중': return <Clock size={14} className="text-yellow-600" />;
            default: return null;
        }
    };

    return (
        <PageContainer>
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h1 className="text-2xl font-bold text-slate-900">결제 관리</h1>
                    <p className="text-slate-500 text-sm mt-1">결제 내역을 확인하고 관리하세요.</p>
                </div>
                <button className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-medium flex items-center gap-2 shadow-sm shadow-primary-200">
                    <CreditCard size={18} />
                    결제 요청
                </button>
            </div>

            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-slate-500 text-sm">총 결제금액</p>
                            <h3 className="text-2xl font-bold text-slate-900 mt-2">₩7,500,000</h3>
                        </div>
                        <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                            <CreditCard className="text-blue-600" size={24} />
                        </div>
                    </div>
                </div>
                <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-slate-500 text-sm">완료된 결제</p>
                            <h3 className="text-2xl font-bold text-green-600 mt-2">1건</h3>
                        </div>
                        <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
                            <CheckCircle className="text-green-600" size={24} />
                        </div>
                    </div>
                </div>
                <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-slate-500 text-sm">대기중</p>
                            <h3 className="text-2xl font-bold text-yellow-600 mt-2">1건</h3>
                        </div>
                        <div className="w-12 h-12 bg-yellow-100 rounded-lg flex items-center justify-center">
                            <Clock className="text-yellow-600" size={24} />
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex items-center gap-4 mb-6">
                <div className="relative flex-1">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                    <input
                        type="text"
                        placeholder="결제번호, 고객명 검색..."
                        className="w-full pl-10 pr-4 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                    />
                </div>
                <select className="px-3 py-2 border border-slate-200 rounded-lg text-sm">
                    <option>전체 상태</option>
                    <option>완료</option>
                    <option>대기중</option>
                    <option>실패</option>
                </select>
            </div>

            <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                <table className="w-full text-sm">
                    <thead className="bg-slate-50 border-b border-slate-200">
                        <tr>
                            <th className="px-6 py-4 text-left font-medium text-slate-500">결제번호</th>
                            <th className="px-6 py-4 text-left font-medium text-slate-500">고객명</th>
                            <th className="px-6 py-4 text-left font-medium text-slate-500">결제금액</th>
                            <th className="px-6 py-4 text-left font-medium text-slate-500">결제수단</th>
                            <th className="px-6 py-4 text-left font-medium text-slate-500">결제일</th>
                            <th className="px-6 py-4 text-left font-medium text-slate-500">상태</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                        {payments.map((p) => (
                            <tr key={p.id} className="hover:bg-slate-50 transition-colors">
                                <td className="px-6 py-4 font-mono text-slate-500">{p.id}</td>
                                <td className="px-6 py-4 font-medium text-slate-900">{p.customer}</td>
                                <td className="px-6 py-4 font-semibold text-slate-900">₩{p.amount}</td>
                                <td className="px-6 py-4 text-slate-600">{p.method}</td>
                                <td className="px-6 py-4 text-slate-500">{p.date}</td>
                                <td className="px-6 py-4">
                                    <div className="flex items-center gap-2">
                                        {getStatusIcon(p.status)}
                                        <span className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(p.status)}`}>
                                            {p.status}
                                        </span>
                                    </div>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </PageContainer>
    );
};

export default PaymentPage;
