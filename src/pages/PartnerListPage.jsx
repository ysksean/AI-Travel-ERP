import React, { useState } from 'react';
import { Plus, Search, Filter, MoreHorizontal, MapPin, Phone, Mail, Globe } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import PageContainer from '../components/PageContainer';

const PartnerListPage = () => {
    const navigate = useNavigate();
    const [partners] = useState([
        { id: 'P001', name: '다낭 투어', region: '다낭', type: '랜드사', contact: '김현지', phone: '070-1234-5678', email: 'danang@tour.com', status: 'active' },
        { id: 'P002', name: '오사카 여행사', region: '오사카', type: '랜드사', contact: '박준형', phone: '070-9876-5432', email: 'osaka@travel.com', status: 'active' },
        { id: 'P003', name: '방콕 가이드', region: '방콕', type: '가이드', contact: '이민수', phone: '010-1111-2222', email: 'bangkok@guide.com', status: 'inactive' },
    ]);

    return (
        <PageContainer>
            <div className="space-y-6">
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-bold text-slate-900">파트너 관리</h1>
                        <p className="text-slate-500 text-sm">랜드사 및 협력업체 정보를 관리합니다.</p>
                    </div>
                    <Link
                        to="/partners/new"
                        className="inline-flex items-center gap-2 bg-primary-600 hover:bg-primary-700 text-white px-5 py-2.5 rounded-lg font-medium transition-colors shadow-sm"
                    >
                        <Plus size={18} />
                        파트너 등록
                    </Link>
                </div>

                <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex items-center gap-4">
                    <div className="relative flex-1">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                        <input
                            type="text"
                            placeholder="업체명, 지역, 담당자 검색..."
                            className="w-full pl-10 pr-4 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                        />
                    </div>
                    <select className="px-3 py-2 border border-slate-200 rounded-lg text-sm">
                        <option>전체 유형</option>
                        <option>랜드사</option>
                        <option>가이드</option>
                        <option>운송업체</option>
                    </select>
                    <button className="p-2 border border-slate-200 rounded-lg hover:bg-slate-50">
                        <Filter size={18} />
                    </button>
                </div>

                <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                    <table className="w-full text-sm">
                        <thead className="bg-slate-50 border-b border-slate-200">
                            <tr>
                                <th className="px-6 py-4 text-left font-medium text-slate-500">업체명</th>
                                <th className="px-6 py-4 text-left font-medium text-slate-500">지역</th>
                                <th className="px-6 py-4 text-left font-medium text-slate-500">유형</th>
                                <th className="px-6 py-4 text-left font-medium text-slate-500">담당자</th>
                                <th className="px-6 py-4 text-left font-medium text-slate-500">연락처</th>
                                <th className="px-6 py-4 text-left font-medium text-slate-500">상태</th>
                                <th className="px-6 py-4 text-right font-medium text-slate-500">관리</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                            {partners.map((partner) => (
                                <tr
                                    key={partner.id}
                                    className="hover:bg-slate-50 transition-colors cursor-pointer"
                                    onClick={() => navigate(`/partners/${partner.id}`)}
                                >
                                    <td className="px-6 py-4 font-medium text-slate-900">{partner.name}</td>
                                    <td className="px-6 py-4 text-slate-600">
                                        <div className="flex items-center gap-1">
                                            <MapPin size={14} />
                                            {partner.region}
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 text-slate-600">
                                        <span className="px-2 py-0.5 bg-slate-100 rounded text-xs font-medium text-slate-600">
                                            {partner.type}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 text-slate-600">{partner.contact}</td>
                                    <td className="px-6 py-4 text-slate-600">
                                        <div className="flex flex-col gap-1 text-xs">
                                            <div className="flex items-center gap-1">
                                                <Phone size={12} /> {partner.phone}
                                            </div>
                                            <div className="flex items-center gap-1">
                                                <Mail size={12} /> {partner.email}
                                            </div>
                                        </div>
                                    </td>
                                    <td className="px-6 py-4">
                                        {partner.status === 'active' ? (
                                            <span className="text-green-600 font-medium text-xs">활동중</span>
                                        ) : (
                                            <span className="text-slate-400 font-medium text-xs">중지</span>
                                        )}
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

export default PartnerListPage;
