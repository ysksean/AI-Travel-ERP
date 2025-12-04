import React, { useState } from 'react';
import { Plus, Search, Filter, MoreHorizontal, MapPin, Camera } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import PageContainer from '../components/PageContainer';

const AttractionListPage = () => {
    const navigate = useNavigate();
    const [attractions] = useState([
        { id: 'A001', name: '바나힐 테마파크', region: '다낭', type: '테마파크', time: '4시간' },
        { id: 'A002', name: '호이안 올드타운', region: '호이안', type: '문화유산', time: '3시간' },
        { id: 'A003', name: '오사카 성', region: '오사카', type: '유적지', time: '2시간' },
    ]);

    return (
        <PageContainer>
            <div className="space-y-6">
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-bold text-slate-900">관광지 관리</h1>
                        <p className="text-slate-500 text-sm">등록된 관광지 정보를 관리합니다.</p>
                    </div>
                    <Link
                        to="/attractions/new"
                        className="inline-flex items-center gap-2 bg-primary-600 hover:bg-primary-700 text-white px-5 py-2.5 rounded-lg font-medium transition-colors shadow-sm"
                    >
                        <Plus size={18} />
                        관광지 등록
                    </Link>
                </div>

                <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex items-center gap-4">
                    <div className="relative flex-1">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                        <input
                            type="text"
                            placeholder="관광지명, 지역 검색..."
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
                                <th className="px-6 py-4 text-left font-medium text-slate-500">관광지명</th>
                                <th className="px-6 py-4 text-left font-medium text-slate-500">지역</th>
                                <th className="px-6 py-4 text-left font-medium text-slate-500">유형</th>
                                <th className="px-6 py-4 text-left font-medium text-slate-500">소요시간</th>
                                <th className="px-6 py-4 text-right font-medium text-slate-500">관리</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                            {attractions.map((attraction) => (
                                <tr
                                    key={attraction.id}
                                    className="hover:bg-slate-50 transition-colors cursor-pointer"
                                    onClick={() => navigate(`/attractions/${attraction.id}`)}
                                >
                                    <td className="px-6 py-4 font-medium text-slate-900">{attraction.name}</td>
                                    <td className="px-6 py-4 text-slate-600">
                                        <div className="flex items-center gap-1">
                                            <MapPin size={14} />
                                            {attraction.region}
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 text-slate-600">
                                        <div className="flex items-center gap-1">
                                            <Camera size={14} />
                                            {attraction.type}
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 text-slate-600">{attraction.time}</td>
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

export default AttractionListPage;
