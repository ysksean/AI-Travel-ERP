import React, { useState } from 'react';
import { Plane, Search, MapPin, Plus } from 'lucide-react';
import PageContainer from '../components/PageContainer';

const FlightPage = () => {
    const [flights] = useState([
        { id: 'KE081', airline: 'Korean Air', route: 'ICN - JFK', departure: '2024-04-01 10:00', arrival: '2024-04-01 11:30', pax: 2, status: '예약완료' },
        { id: 'OZ102', airline: 'Asiana Airlines', route: 'ICN - NRT', departure: '2024-04-05 09:00', arrival: '2024-04-05 11:20', pax: 4, status: '발권대기' },
        { id: 'DL158', airline: 'Delta Air Lines', route: 'ICN - DTW', departure: '2024-04-10 10:25', arrival: '2024-04-10 10:00', pax: 1, status: '취소됨' },
        { id: 'KE703', airline: 'Korean Air', route: 'ICN - NRT', departure: '2024-04-12 10:10', arrival: '2024-04-12 12:30', pax: 3, status: '예약완료' },
    ]);

    return (
        <PageContainer>
            <div className="space-y-6">
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-bold text-slate-900">항공권 관리</h1>
                        <p className="text-slate-500 text-sm mt-1">항공권 예약 및 발권 상태를 관리합니다.</p>
                    </div>
                    <button className="inline-flex items-center gap-2 bg-primary-600 hover:bg-primary-700 text-white px-5 py-2.5 rounded-lg font-medium transition-colors shadow-sm shadow-primary-200">
                        <Plus size={18} />
                        항공권 검색
                    </button>
                </div>

                {/* AI Search Section */}
                <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border border-blue-200 p-6 shadow-sm">
                    <div className="flex items-start gap-4">
                        <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center flex-shrink-0">
                            <Plane className="text-blue-600" size={24} />
                        </div>
                        <div className="flex-1">
                            <h3 className="font-bold text-slate-900 mb-2">AI 항공권 검색</h3>
                            <p className="text-sm text-slate-600 mb-4">출발지, 도착지, 날짜를 입력하면 AI가 최적의 항공권을 찾아드립니다.</p>
                            <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                                <input type="text" placeholder="출발지 (예: ICN)" className="px-3 py-2 bg-white border border-slate-200 rounded-lg text-sm" />
                                <input type="text" placeholder="도착지 (예: NRT)" className="px-3 py-2 bg-white border border-slate-200 rounded-lg text-sm" />
                                <input type="date" className="px-3 py-2 bg-white border border-slate-200 rounded-lg text-sm" />
                                <button className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg font-medium text-sm">
                                    검색
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Filter Bar */}
                <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex items-center gap-4">
                    <div className="relative flex-1">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                        <input
                            type="text"
                            placeholder="항공편명, 항공사 검색..."
                            className="w-full pl-10 pr-4 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                        />
                    </div>
                    <select className="px-3 py-2 border border-slate-200 rounded-lg text-sm">
                        <option>전체 상태</option>
                        <option>예약완료</option>
                        <option>발권대기</option>
                        <option>취소됨</option>
                    </select>
                </div>

                {/* Flight List */}
                <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                    <table className="w-full text-sm">
                        <thead className="bg-slate-50 border-b border-slate-200">
                            <tr>
                                <th className="px-6 py-4 text-left font-medium text-slate-500">항공편</th>
                                <th className="px-6 py-4 text-left font-medium text-slate-500">항공사</th>
                                <th className="px-6 py-4 text-left font-medium text-slate-500">노선</th>
                                <th className="px-6 py-4 text-left font-medium text-slate-500">출발</th>
                                <th className="px-6 py-4 text-left font-medium text-slate-500">도착</th>
                                <th className="px-6 py-4 text-left font-medium text-slate-500">인원</th>
                                <th className="px-6 py-4 text-left font-medium text-slate-500">상태</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                            {flights.map((flight) => (
                                <tr key={flight.id} className="hover:bg-slate-50 transition-colors">
                                    <td className="px-6 py-4 font-mono text-slate-500">{flight.id}</td>
                                    <td className="px-6 py-4 font-medium text-slate-900">{flight.airline}</td>
                                    <td className="px-6 py-4">
                                        <div className="flex items-center gap-2 text-slate-600">
                                            <MapPin size={14} />
                                            {flight.route}
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 text-slate-600">{flight.departure}</td>
                                    <td className="px-6 py-4 text-slate-600">{flight.arrival}</td>
                                    <td className="px-6 py-4 text-slate-600">{flight.pax}명</td>
                                    <td className="px-6 py-4">
                                        <span className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${flight.status === '예약완료' ? 'bg-green-100 text-green-800' :
                                            flight.status === '발권대기' ? 'bg-yellow-100 text-yellow-800' :
                                                'bg-red-100 text-red-800'
                                            }`}>
                                            {flight.status}
                                        </span>
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

export default FlightPage;
