import React, { useState } from 'react';
import { ArrowLeft, Save, Printer, Mail, Calculator, Calendar } from 'lucide-react';
import { Link, useParams } from 'react-router-dom';
import PageContainer from '../components/PageContainer';

const QuotationDetailPage = () => {
    const { id } = useParams();

    // Mock Data
    const quotation = {
        id: id || 'Q20240315-01',
        customer: '김철수',
        phone: '010-1234-5678',
        email: 'kim@example.com',
        product: '다낭 3박 4일 패키지',
        date: '2024-03-15',
        validUntil: '2024-03-22',
        status: '발송완료',
        items: [
            { id: 1, name: '성인 기본 상품가', quantity: 2, price: 1200000, total: 2400000 },
            { id: 2, name: '유류할증료', quantity: 2, price: 150000, total: 300000 },
            { id: 3, name: '가이드 경비', quantity: 2, price: 50000, total: 100000 },
        ]
    };

    const calculateTotal = () => {
        return quotation.items.reduce((sum, item) => sum + item.total, 0);
    };

    return (
        <PageContainer>
            <div className="space-y-6">
                {/* Header */}
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <Link to="/quotations" className="p-2 hover:bg-slate-100 rounded-full text-slate-500 transition-colors">
                            <ArrowLeft size={24} />
                        </Link>
                        <div>
                            <div className="flex items-center gap-3">
                                <h1 className="text-2xl font-bold text-slate-900">견적서 {quotation.id}</h1>
                                <span className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${quotation.status === '발송완료' ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'}`}>
                                    {quotation.status}
                                </span>
                            </div>
                            <p className="text-slate-500 text-sm">작성일: {quotation.date} | 유효기간: {quotation.validUntil}</p>
                        </div>
                    </div>
                    <div className="flex gap-3">
                        <button className="px-4 py-2 text-slate-600 font-medium hover:bg-slate-100 rounded-lg transition-colors flex items-center gap-2">
                            <Printer size={18} />
                            인쇄
                        </button>
                        <button className="px-4 py-2 text-slate-600 font-medium hover:bg-slate-100 rounded-lg transition-colors flex items-center gap-2">
                            <Mail size={18} />
                            재발송
                        </button>
                        <button className="px-6 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-medium shadow-sm flex items-center gap-2">
                            <Save size={18} />
                            수정
                        </button>
                    </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Left Column: Quotation Details */}
                    <div className="lg:col-span-2 space-y-6">
                        {/* Customer & Product Info */}
                        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                            <h2 className="text-lg font-bold text-slate-900 mb-4">기본 정보</h2>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <div className="space-y-1">
                                    <label className="text-xs font-medium text-slate-500">고객명</label>
                                    <p className="text-slate-900 font-medium">{quotation.customer}</p>
                                </div>
                                <div className="space-y-1">
                                    <label className="text-xs font-medium text-slate-500">연락처</label>
                                    <p className="text-slate-900">{quotation.phone}</p>
                                </div>
                                <div className="space-y-1">
                                    <label className="text-xs font-medium text-slate-500">이메일</label>
                                    <p className="text-slate-900">{quotation.email}</p>
                                </div>
                                <div className="space-y-1">
                                    <label className="text-xs font-medium text-slate-500">상품명</label>
                                    <p className="text-slate-900 font-medium">{quotation.product}</p>
                                </div>
                            </div>
                        </div>

                        {/* Items Table */}
                        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                            <h2 className="text-lg font-bold text-slate-900 mb-4">견적 상세</h2>
                            <div className="overflow-x-auto">
                                <table className="w-full text-sm">
                                    <thead className="bg-slate-50 border-b border-slate-200">
                                        <tr>
                                            <th className="px-4 py-3 text-left font-medium text-slate-500">항목명</th>
                                            <th className="px-4 py-3 text-center font-medium text-slate-500">수량</th>
                                            <th className="px-4 py-3 text-right font-medium text-slate-500">단가</th>
                                            <th className="px-4 py-3 text-right font-medium text-slate-500">금액</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-slate-100">
                                        {quotation.items.map((item) => (
                                            <tr key={item.id}>
                                                <td className="px-4 py-3 text-slate-900">{item.name}</td>
                                                <td className="px-4 py-3 text-center text-slate-600">{item.quantity}</td>
                                                <td className="px-4 py-3 text-right text-slate-600">₩{item.price.toLocaleString()}</td>
                                                <td className="px-4 py-3 text-right font-medium text-slate-900">₩{item.total.toLocaleString()}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                    <tfoot className="bg-slate-50 border-t border-slate-200">
                                        <tr>
                                            <td colSpan="3" className="px-4 py-3 text-right font-bold text-slate-900">합계</td>
                                            <td className="px-4 py-3 text-right font-bold text-primary-600">₩{calculateTotal().toLocaleString()}</td>
                                        </tr>
                                    </tfoot>
                                </table>
                            </div>
                        </div>
                    </div>

                    {/* Right Column: Summary & Actions */}
                    <div className="space-y-6">
                        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                            <h3 className="font-bold text-slate-900 mb-4 flex items-center gap-2">
                                <Calculator size={18} />
                                견적 요약
                            </h3>
                            <div className="space-y-3">
                                <div className="flex justify-between text-sm">
                                    <span className="text-slate-600">소계</span>
                                    <span className="font-medium">₩{calculateTotal().toLocaleString()}</span>
                                </div>
                                <div className="flex justify-between text-sm">
                                    <span className="text-slate-600">할인</span>
                                    <span className="font-medium text-red-600">- ₩0</span>
                                </div>
                                <div className="pt-3 border-t border-slate-200">
                                    <div className="flex justify-between">
                                        <span className="font-bold text-slate-900">총 견적금액</span>
                                        <span className="text-xl font-bold text-primary-600">₩{calculateTotal().toLocaleString()}</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div className="bg-blue-50 rounded-xl border border-blue-100 p-6">
                            <h3 className="font-bold text-blue-900 mb-2">안내사항</h3>
                            <ul className="list-disc list-inside text-sm text-blue-800 space-y-1">
                                <li>상기 견적은 항공 좌석 상황에 따라 변동될 수 있습니다.</li>
                                <li>유효기간 내에 예약 확정 시 요금이 보장됩니다.</li>
                                <li>예약 시 여권 사본이 필요합니다.</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </PageContainer>
    );
};

export default QuotationDetailPage;
