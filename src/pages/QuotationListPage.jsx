import React, { useState } from 'react';
import { Plus, Search, Filter, MoreHorizontal } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import PageContainer from '../components/PageContainer';

const QuotationListPage = () => {
    const navigate = useNavigate();
    const [quotations] = useState([
        { id: 'Q20240315-01', customer: '김철수', product: '다낭 3박 4일 패키지', amount: '1,250,000', date: '2024-03-15', status: '발송완료' },
        { id: 'Q20240314-05', customer: '이영희', product: '방콕/파타야 5일', amount: '890,000', date: '2024-03-14', status: '작성중' },
        { id: 'Q20240314-02', customer: '박지민', product: '오사카 자유여행', amount: '650,000', date: '2024-03-14', status: '발송완료' },
        { id: 'Q20240313-08', customer: '최민수', product: '세부 제이파크 4일', amount: '1,100,000', date: '2024-03-13', status: '작성중' },
        { id: 'Q20240312-03', customer: '정수진', product: '나트랑 빈펄 5일', amount: '1,450,000', date: '2024-03-12', status: '발송완료' },
    ]);

    return (
        <PageContainer>
            <div className="space-y-6">
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-bold text-slate-900">견적 관리</h1>
                        <p className="text-slate-500 text-sm">고객에게 발송된 견적서를 관리합니다.</p>
                    </div>
                    <Link
                        to="/quotations/new"
                        className="inline-flex items-center gap-2 bg-primary-600 hover:bg-primary-700 text-white px-5 py-2.5 rounded-lg font-medium transition-colors shadow-sm"
                    >
                        <Plus size={18} />
                        견적서 작성
                    </Link>
                </div>

                <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex items-center gap-4">
                    <div className="relative flex-1">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                        <input
                            type="text"
                            placeholder="견적번호, 고객명 검색..."
                            className="w-full pl-10 pr-4 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                        />
                    </div>
                    <button className="p-2 border border-slate-200 rounded-lg hover:bg-slate-50">
                        <Filter size={18} />
                    </button>
                </div>

                <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                    <table className="w-full text-sm">
                        <thead className="bg-slate-50 border-b border-slate-200">
                            <tr>
                                <th className="px-6 py-4 text-left font-medium text-slate-500">견적번호</th>
                                <th className="px-6 py-4 text-left font-medium text-slate-500">고객명</th>
                                <th className="px-6 py-4 text-left font-medium text-slate-500">상품명</th>
                                <th className="px-6 py-4 text-left font-medium text-slate-500">견적금액</th>
                                <th className="px-6 py-4 text-left font-medium text-slate-500">작성일</th>
                                <th className="px-6 py-4 text-left font-medium text-slate-500">상태</th>
                                <th className="px-6 py-4 text-right font-medium text-slate-500">관리</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                            {quotations.map((q) => (
                                <tr
                                    key={q.id}
                                    onClick={() => navigate(`/quotations/${q.id}`)}
                                    className="hover:bg-slate-50 transition-colors cursor-pointer"
                                >
                                    <td className="px-6 py-4 font-mono text-slate-500">{q.id}</td>
                                    <td className="px-6 py-4 font-medium text-slate-900">{q.customer}</td>
                                    <td className="px-6 py-4 text-slate-600">{q.product}</td>
                                    <td className="px-6 py-4 font-semibold text-slate-900">₩{q.amount}</td>
                                    <td className="px-6 py-4 text-slate-500">{q.date}</td>
                                    <td className="px-6 py-4">
                                        <span className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${q.status === '발송완료' ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
                                            }`}>
                                            {q.status}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 text-right" onClick={(e) => e.stopPropagation()}>
                                        <button className="p-1.5 hover:bg-slate-200 rounded text-slate-400 hover:text-slate-600">
                                            <MoreHorizontal size={18} />
                                        </button>
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

export default QuotationListPage;
