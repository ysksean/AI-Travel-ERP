import React, { useState } from 'react';
import { Save, Plus, Trash2, Calculator, Plane } from 'lucide-react';
import PageContainer from '../components/PageContainer';

const QuotationCreatePage = () => {
    const [items, setItems] = useState([
        { id: 1, name: '성인 기본 상품가', quantity: 2, price: 1200000, total: 2400000 },
        { id: 2, name: '유류할증료', quantity: 2, price: 150000, total: 300000 },
    ]);

    const addItem = () => {
        const newItem = {
            id: Date.now(),
            name: '',
            quantity: 1,
            price: 0,
            total: 0
        };
        setItems([...items, newItem]);
    };

    const removeItem = (id) => {
        setItems(items.filter(item => item.id !== id));
    };

    const calculateTotal = () => {
        return items.reduce((sum, item) => sum + item.total, 0);
    };

    return (
        <PageContainer>
            <div className="space-y-6">
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-bold text-slate-900">견적서 작성</h1>
                        <p className="text-slate-500 text-sm">고객에게 발송할 견적서를 작성합니다.</p>
                    </div>
                    <div className="flex gap-3">
                        <button className="px-4 py-2 text-slate-600 font-medium hover:bg-slate-100 rounded-lg transition-colors">
                            미리보기
                        </button>
                        <button className="px-6 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-medium shadow-sm flex items-center gap-2">
                            <Save size={18} />
                            저장 및 발송
                        </button>
                    </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <div className="lg:col-span-2 space-y-6">
                        {/* Customer Info */}
                        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                            <h2 className="text-lg font-bold text-slate-900 mb-4">고객 정보</h2>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">고객명</label>
                                    <input type="text" className="w-full px-3 py-2 border border-slate-200 rounded-lg" placeholder="홍길동" />
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">연락처</label>
                                    <input type="text" className="w-full px-3 py-2 border border-slate-200 rounded-lg" placeholder="010-1234-5678" />
                                </div>
                                <div className="space-y-2 md:col-span-2">
                                    <label className="text-sm font-medium text-slate-700">이메일</label>
                                    <input type="email" className="w-full px-3 py-2 border border-slate-200 rounded-lg" placeholder="customer@example.com" />
                                </div>
                            </div>
                        </div>

                        {/* Product Selection */}
                        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                            <h2 className="text-lg font-bold text-slate-900 mb-4">상품 선택</h2>
                            <select className="w-full px-3 py-2 border border-slate-200 rounded-lg">
                                <option>일본 프리미엄 골프 투어</option>
                                <option>규슈 온천 여행</option>
                                <option>다낭 가족여행</option>
                            </select>
                        </div>

                        {/* Flight Info */}
                        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                            <h2 className="text-lg font-bold text-slate-900 mb-4 flex items-center gap-2">
                                <Plane size={20} className="text-slate-400" />
                                항공 정보
                            </h2>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">항공사</label>
                                    <input type="text" className="w-full px-3 py-2 border border-slate-200 rounded-lg" placeholder="예: 대한항공" />
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">편명</label>
                                    <input type="text" className="w-full px-3 py-2 border border-slate-200 rounded-lg" placeholder="예: KE123" />
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">출발</label>
                                    <input type="datetime-local" className="w-full px-3 py-2 border border-slate-200 rounded-lg" />
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">도착</label>
                                    <input type="datetime-local" className="w-full px-3 py-2 border border-slate-200 rounded-lg" />
                                </div>
                            </div>
                        </div>

                        {/* Items Table */}
                        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                            <div className="flex items-center justify-between mb-4">
                                <h2 className="text-lg font-bold text-slate-900">견적 항목</h2>
                                <button onClick={addItem} className="text-sm text-primary-600 font-medium hover:underline flex items-center gap-1">
                                    <Plus size={16} /> 항목 추가
                                </button>
                            </div>

                            <div className="overflow-x-auto">
                                <table className="w-full text-sm">
                                    <thead className="bg-slate-50 border-b border-slate-200">
                                        <tr>
                                            <th className="px-4 py-3 text-left font-medium text-slate-500">항목명</th>
                                            <th className="px-4 py-3 text-left font-medium text-slate-500">수량</th>
                                            <th className="px-4 py-3 text-left font-medium text-slate-500">단가</th>
                                            <th className="px-4 py-3 text-left font-medium text-slate-500">금액</th>
                                            <th className="px-4 py-3 text-right font-medium text-slate-500">삭제</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-slate-100">
                                        {items.map((item) => (
                                            <tr key={item.id}>
                                                <td className="px-4 py-3">
                                                    <input type="text" defaultValue={item.name} className="w-full px-2 py-1 border border-slate-200 rounded" />
                                                </td>
                                                <td className="px-4 py-3">
                                                    <input type="number" defaultValue={item.quantity} className="w-20 px-2 py-1 border border-slate-200 rounded" />
                                                </td>
                                                <td className="px-4 py-3">
                                                    <input type="number" defaultValue={item.price} className="w-28 px-2 py-1 border border-slate-200 rounded" />
                                                </td>
                                                <td className="px-4 py-3 font-semibold">₩{item.total.toLocaleString()}</td>
                                                <td className="px-4 py-3 text-right">
                                                    <button onClick={() => removeItem(item.id)} className="text-red-500 hover:text-red-700">
                                                        <Trash2 size={16} />
                                                    </button>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>

                    {/* Right Column: Summary */}
                    <div className="space-y-6">
                        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm sticky top-6">
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
                                    <input type="number" className="w-24 px-2 py-1 border border-slate-200 rounded text-right" defaultValue={0} />
                                </div>
                                <div className="pt-3 border-t border-slate-200">
                                    <div className="flex justify-between">
                                        <span className="font-bold text-slate-900">총 금액</span>
                                        <span className="text-xl font-bold text-primary-600">₩{calculateTotal().toLocaleString()}</span>
                                    </div>
                                </div>
                            </div>

                            <div className="mt-6 space-y-3">
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">유효기간</label>
                                    <input type="date" className="w-full px-3 py-2 border border-slate-200 rounded-lg text-sm" />
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">비고</label>
                                    <textarea className="w-full px-3 py-2 border border-slate-200 rounded-lg text-sm" rows={3} placeholder="추가 안내사항"></textarea>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </PageContainer>
    );
};

export default QuotationCreatePage;
