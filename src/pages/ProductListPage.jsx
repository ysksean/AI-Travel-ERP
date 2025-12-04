import React, { useState } from 'react';
import { Plus, Search, Filter, MoreHorizontal, Calendar } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import PageContainer from '../components/PageContainer';

const ProductListPage = () => {
    const navigate = useNavigate();
    const [products] = useState([
        { id: 'P20240315-01', name: '다낭 3박 4일 패키지', status: '판매중', region: '동남아', period: '2024.03.15 ~ 2024.03.18', manager: '김철수' },
        { id: 'P20240314-05', name: '오사카 2박 3일 자유여행', status: '판매중', region: '일본', period: '2024.03.20 ~ 2024.03.22', manager: '이영희' },
        { id: 'P20240314-02', name: '방콕/파타야 5일', status: '마감임박', region: '동남아', period: '2024.04.01 ~ 2024.04.05', manager: '박지민' },
        { id: 'P20240313-08', name: '세부 제이파크 4일', status: '판매중', region: '동남아', period: '2024.03.25 ~ 2024.03.28', manager: '최민수' },
    ]);

    return (
        <PageContainer>
            <div className="space-y-6">
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-bold text-slate-900">상품 관리</h1>
                        <p className="text-slate-500 text-sm mt-1">여행 상품 및 일정표를 관리합니다.</p>
                    </div>
                    <Link
                        to="/products/new"
                        className="inline-flex items-center justify-center gap-2 bg-primary-600 hover:bg-primary-700 text-white px-5 py-2.5 rounded-lg font-medium transition-colors shadow-sm shadow-primary-200"
                    >
                        <Plus size={18} />
                        상품 등록
                    </Link>
                </div>

                {/* Filters & Search */}
                <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex flex-col sm:flex-row gap-4 items-center justify-between">
                    <div className="flex items-center gap-2 w-full sm:w-auto">
                        <div className="relative flex-1 sm:w-80">
                            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                            <input
                                type="text"
                                placeholder="상품코드 또는 상품명 검색..."
                                className="w-full pl-10 pr-4 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 transition-all"
                            />
                        </div>
                        <button className="p-2 border border-slate-200 rounded-lg hover:bg-slate-50 text-slate-600">
                            <Filter size={18} />
                        </button>
                    </div>

                    <div className="flex items-center gap-3 text-sm">
                        <select className="bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500">
                            <option>전체 담당자</option>
                            <option>김철수</option>
                            <option>이영희</option>
                        </select>
                        <select className="bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500">
                            <option>전체 지역</option>
                            <option>일본</option>
                            <option>동남아</option>
                        </select>
                    </div>
                </div>

                {/* Product Table */}
                <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm text-left">
                            <thead className="bg-slate-50 text-slate-500 font-medium border-b border-slate-200">
                                <tr>
                                    <th className="px-6 py-4 w-10">
                                        <input type="checkbox" className="rounded border-slate-300 text-primary-600 focus:ring-primary-500" />
                                    </th>
                                    <th className="px-6 py-4">상태</th>
                                    <th className="px-6 py-4">상품코드</th>
                                    <th className="px-6 py-4">상품명</th>
                                    <th className="px-6 py-4">행사기간</th>
                                    <th className="px-6 py-4">담당자</th>
                                    <th className="px-6 py-4 text-right">관리</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-100">
                                {products.map((product) => (
                                    <tr
                                        key={product.id}
                                        onClick={() => navigate(`/products/${product.id}`)}
                                        className="hover:bg-slate-50 transition-colors group cursor-pointer"
                                    >
                                        <td className="px-6 py-4" onClick={(e) => e.stopPropagation()}>
                                            <input type="checkbox" className="rounded border-slate-300 text-primary-600 focus:ring-primary-500" />
                                        </td>
                                        <td className="px-6 py-4">
                                            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${product.status === '판매중'
                                                ? 'bg-blue-50 text-blue-700'
                                                : 'bg-slate-100 text-slate-600'
                                                }`}>
                                                {product.status}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 font-mono text-slate-500">{product.id}</td>
                                        <td className="px-6 py-4 font-medium text-slate-900">
                                            <div className="flex items-center gap-2">
                                                {product.name}
                                                {product.region === '일본' && <span className="text-xs bg-red-50 text-red-600 px-1.5 rounded">JP</span>}
                                            </div>
                                        </td>
                                        <td className="px-6 py-4 text-slate-500">
                                            <div className="flex items-center gap-1.5">
                                                <Calendar size={14} />
                                                {product.period}
                                            </div>
                                        </td>
                                        <td className="px-6 py-4">
                                            <div className="flex items-center gap-2">
                                                <div className="w-6 h-6 rounded-full bg-slate-200 flex items-center justify-center text-xs font-bold text-slate-600">
                                                    {product.manager[0]}
                                                </div>
                                                <span className="text-slate-600">{product.manager}</span>
                                            </div>
                                        </td>
                                        <td className="px-6 py-4 text-right" onClick={(e) => e.stopPropagation()}>
                                            <button className="p-1.5 hover:bg-slate-200 rounded text-slate-400 hover:text-slate-600 transition-colors">
                                                <MoreHorizontal size={18} />
                                            </button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>

                    {/* Pagination */}
                    <div className="px-6 py-4 border-t border-slate-100 flex items-center justify-between text-sm text-slate-500">
                        <span>총 284개 중 1-4 표시</span>
                        <div className="flex gap-1">
                            <button className="px-3 py-1 border border-slate-200 rounded hover:bg-slate-50 disabled:opacity-50">이전</button>
                            <button className="px-3 py-1 bg-primary-50 text-primary-700 border border-primary-100 rounded font-medium">1</button>
                            <button className="px-3 py-1 border border-slate-200 rounded hover:bg-slate-50">2</button>
                            <button className="px-3 py-1 border border-slate-200 rounded hover:bg-slate-50">다음</button>
                        </div>
                    </div>
                </div>
            </div>
        </PageContainer>
    );
};

export default ProductListPage;
