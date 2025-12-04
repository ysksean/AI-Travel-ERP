import React, { useState } from 'react';
import { Plus, Search, Filter, MoreHorizontal, MapPin, Star } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import PageContainer from '../components/PageContainer';

const HotelListPage = () => {
    const navigate = useNavigate();
    const [hotels] = useState([
        { id: 'H001', name: '노보텔 다낭 프리미어 한 리버', region: '다낭', rating: 5, rooms: 323, manager: '김철수' },
        { id: 'H002', name: '신라 호텔 서울', region: '서울', rating: 5, rooms: 464, manager: '이영희' },
        { id: 'H003', name: '힐튼 오사카', region: '오사카', rating: 4, rooms: 527, manager: '박지민' },
    ]);

    return (
        <PageContainer>
            <div className="space-y-6">
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-bold text-slate-900">호텔 관리</h1>
                        <p className="text-slate-500 text-sm">등록된 호텔 정보를 관리합니다.</p>
                    </div>
                    <Link
                        to="/hotels/new"
                        className="inline-flex items-center gap-2 bg-primary-600 hover:bg-primary-700 text-white px-5 py-2.5 rounded-lg font-medium transition-colors shadow-sm"
                    >
                        <Plus size={18} />
                        호텔 등록
                    </Link>
                </div>

                <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex items-center gap-4">
                    <div className="relative flex-1">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                        <input
                            type="text"
                            placeholder="호텔명, 지역 검색..."
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
                                <th className="px-6 py-4 text-left font-medium text-slate-500">호텔명</th>
                                <th className="px-6 py-4 text-left font-medium text-slate-500">지역</th>
                                <th className="px-6 py-4 text-left font-medium text-slate-500">등급</th>
                                <th className="px-6 py-4 text-left font-medium text-slate-500">객실수</th>
                                <th className="px-6 py-4 text-left font-medium text-slate-500">담당자</th>
                                <th className="px-6 py-4 text-right font-medium text-slate-500">관리</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                            {hotels.map((hotel) => (
                                <tr
                                    key={hotel.id}
                                    className="hover:bg-slate-50 transition-colors cursor-pointer"
                                    onClick={() => navigate(`/hotels/${hotel.id}`)}
                                >
                                    <td className="px-6 py-4 font-medium text-slate-900">{hotel.name}</td>
                                    <td className="px-6 py-4 text-slate-600">
                                        <div className="flex items-center gap-1">
                                            <MapPin size={14} />
                                            {hotel.region}
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 text-slate-600">
                                        <div className="flex items-center gap-1 text-yellow-500">
                                            <Star size={14} fill="currentColor" />
                                            <span className="text-slate-600">{hotel.rating}성급</span>
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 text-slate-600">{hotel.rooms}실</td>
                                    <td className="px-6 py-4 text-slate-600">{hotel.manager}</td>
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

export default HotelListPage;
