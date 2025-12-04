import React, { useState } from 'react';
import { Plus, Search, Filter, Users, Calendar, Phone, Mail, Shield, AlertTriangle } from 'lucide-react';
import PageContainer from '../components/PageContainer';

const CustomerPage = () => {
    const [activeTab, setActiveTab] = useState('list');
    const [customers] = useState([
        { id: 'C24031501', name: '김철수', phone: '010-1234-5678', email: 'kim@example.com', trips: 5, lastTrip: '2024-01-15', grade: 'VIP', status: 'active' },
        { id: 'C24031405', name: '이영희', phone: '010-9876-5432', email: 'lee@example.com', trips: 2, lastTrip: '2023-12-20', grade: 'Gold', status: 'active' },
        { id: 'C24031402', name: '박지민', phone: '010-5555-4444', email: 'park@example.com', trips: 8, lastTrip: '2024-02-10', grade: 'VIP', status: 'active' },
        { id: 'C24031308', name: '최민수', phone: '010-1111-2222', email: 'choi@example.com', trips: 1, lastTrip: '2024-03-01', grade: 'Silver', status: 'active' },
        { id: 'C24031203', name: '정수진', phone: '010-3333-7777', email: 'jung@example.com', trips: 3, lastTrip: '2023-11-05', grade: 'Gold', status: 'active' },
        { id: 'C24031199', name: '홍길동', phone: '010-9999-8888', email: 'hong@example.com', trips: 0, lastTrip: '-', grade: 'Silver', status: 'blacklist', reason: '노쇼 3회 누적' },
    ]);

    const [grades] = useState([
        { name: 'VIP', criteria: '연 5회 이상 또는 1,000만원 이상', benefits: '전용 상담사, 5% 할인' },
        { name: 'Gold', criteria: '연 3회 이상 또는 500만원 이상', benefits: '3% 할인' },
        { name: 'Silver', criteria: '신규 가입', benefits: '기본 혜택' },
    ]);

    return (
        <PageContainer>
            <div className="space-y-6">
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-bold text-slate-900">고객 관리</h1>
                        <p className="text-slate-500 text-sm mt-1">고객 정보 및 여행 이력을 관리합니다.</p>
                    </div>
                    <button className="inline-flex items-center gap-2 bg-primary-600 hover:bg-primary-700 text-white px-5 py-2.5 rounded-lg font-medium transition-colors shadow-sm shadow-primary-200">
                        <Plus size={18} />
                        고객 등록
                    </button>
                </div>

                {/* Tabs */}
                <div className="flex border-b border-slate-200">
                    <button
                        onClick={() => setActiveTab('list')}
                        className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${activeTab === 'list' ? 'border-primary-600 text-primary-600' : 'border-transparent text-slate-500 hover:text-slate-700'}`}
                    >
                        고객 목록
                    </button>
                    <button
                        onClick={() => setActiveTab('grades')}
                        className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${activeTab === 'grades' ? 'border-primary-600 text-primary-600' : 'border-transparent text-slate-500 hover:text-slate-700'}`}
                    >
                        등급 관리
                    </button>
                    <button
                        onClick={() => setActiveTab('blacklist')}
                        className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${activeTab === 'blacklist' ? 'border-primary-600 text-primary-600' : 'border-transparent text-slate-500 hover:text-slate-700'}`}
                    >
                        블랙리스트
                    </button>
                </div>

                {/* Customer List Content */}
                {activeTab === 'list' && (
                    <>
                        {/* Stats */}
                        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                            <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-slate-500 text-sm">전체 고객</p>
                                        <h3 className="text-2xl font-bold text-slate-900 mt-2">284명</h3>
                                    </div>
                                    <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                                        <Users className="text-blue-600" size={24} />
                                    </div>
                                </div>
                            </div>
                            <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-slate-500 text-sm">VIP 고객</p>
                                        <h3 className="text-2xl font-bold text-amber-600 mt-2">42명</h3>
                                    </div>
                                    <div className="w-12 h-12 bg-amber-100 rounded-lg flex items-center justify-center">
                                        <Users className="text-amber-600" size={24} />
                                    </div>
                                </div>
                            </div>
                            <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-slate-500 text-sm">이번 달 신규</p>
                                        <h3 className="text-2xl font-bold text-green-600 mt-2">12명</h3>
                                    </div>
                                    <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
                                        <Plus className="text-green-600" size={24} />
                                    </div>
                                </div>
                            </div>
                            <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-slate-500 text-sm">재방문율</p>
                                        <h3 className="text-2xl font-bold text-purple-600 mt-2">68%</h3>
                                    </div>
                                    <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center">
                                        <Calendar className="text-purple-600" size={24} />
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
                                    placeholder="이름, 전화번호, 이메일 검색..."
                                    className="w-full pl-10 pr-4 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                                />
                            </div>
                            <select className="px-3 py-2 border border-slate-200 rounded-lg text-sm">
                                <option>전체 고객</option>
                                <option>VIP 고객</option>
                                <option>일반 고객</option>
                            </select>
                            <button className="p-2 border border-slate-200 rounded-lg hover:bg-slate-50">
                                <Filter size={18} />
                            </button>
                        </div>

                        {/* Customer List */}
                        <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                            <table className="w-full text-sm">
                                <thead className="bg-slate-50 border-b border-slate-200">
                                    <tr>
                                        <th className="px-6 py-4 text-left font-medium text-slate-500">고객번호</th>
                                        <th className="px-6 py-4 text-left font-medium text-slate-500">이름</th>
                                        <th className="px-6 py-4 text-left font-medium text-slate-500">연락처</th>
                                        <th className="px-6 py-4 text-left font-medium text-slate-500">이메일</th>
                                        <th className="px-6 py-4 text-left font-medium text-slate-500">여행 횟수</th>
                                        <th className="px-6 py-4 text-left font-medium text-slate-500">최근 여행</th>
                                        <th className="px-6 py-4 text-left font-medium text-slate-500">등급</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-100">
                                    {customers.filter(c => c.status === 'active').map((customer) => (
                                        <tr key={customer.id} className="hover:bg-slate-50 transition-colors cursor-pointer">
                                            <td className="px-6 py-4 font-mono text-slate-500">{customer.id}</td>
                                            <td className="px-6 py-4 font-medium text-slate-900">{customer.name}</td>
                                            <td className="px-6 py-4">
                                                <div className="flex items-center gap-2 text-slate-600">
                                                    <Phone size={14} />
                                                    {customer.phone}
                                                </div>
                                            </td>
                                            <td className="px-6 py-4">
                                                <div className="flex items-center gap-2 text-slate-600">
                                                    <Mail size={14} />
                                                    {customer.email}
                                                </div>
                                            </td>
                                            <td className="px-6 py-4 text-slate-600">{customer.trips}회</td>
                                            <td className="px-6 py-4 text-slate-500">{customer.lastTrip}</td>
                                            <td className="px-6 py-4">
                                                <span className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${customer.grade === 'VIP' ? 'bg-amber-100 text-amber-800' :
                                                    customer.grade === 'Gold' ? 'bg-yellow-100 text-yellow-800' :
                                                        'bg-slate-100 text-slate-600'
                                                    }`}>
                                                    {customer.grade}
                                                </span>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </>
                )}

                {/* Grades Content */}
                {activeTab === 'grades' && (
                    <div className="space-y-6">
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                            {grades.map((grade) => (
                                <div key={grade.name} className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-4">
                                    <div className="flex items-center justify-between">
                                        <h3 className="text-lg font-bold text-slate-900">{grade.name}</h3>
                                        <Shield className={`
                                            ${grade.name === 'VIP' ? 'text-amber-500' :
                                                grade.name === 'Gold' ? 'text-yellow-500' : 'text-slate-400'}
                                        `} size={24} />
                                    </div>
                                    <div className="space-y-2">
                                        <p className="text-sm font-medium text-slate-700">등급 기준</p>
                                        <p className="text-sm text-slate-600">{grade.criteria}</p>
                                    </div>
                                    <div className="space-y-2">
                                        <p className="text-sm font-medium text-slate-700">혜택</p>
                                        <p className="text-sm text-slate-600">{grade.benefits}</p>
                                    </div>
                                    <button className="w-full py-2 border border-slate-200 rounded-lg text-sm font-medium hover:bg-slate-50">
                                        수정
                                    </button>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Blacklist Content */}
                {activeTab === 'blacklist' && (
                    <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                        <div className="p-4 border-b border-slate-200 flex justify-between items-center">
                            <h3 className="font-bold text-slate-900 flex items-center gap-2">
                                <AlertTriangle className="text-red-500" size={20} />
                                블랙리스트 관리
                            </h3>
                            <button className="px-4 py-2 bg-red-50 text-red-600 rounded-lg text-sm font-medium hover:bg-red-100">
                                블랙리스트 등록
                            </button>
                        </div>
                        <table className="w-full text-sm">
                            <thead className="bg-slate-50 border-b border-slate-200">
                                <tr>
                                    <th className="px-6 py-4 text-left font-medium text-slate-500">고객명</th>
                                    <th className="px-6 py-4 text-left font-medium text-slate-500">연락처</th>
                                    <th className="px-6 py-4 text-left font-medium text-slate-500">등록 사유</th>
                                    <th className="px-6 py-4 text-left font-medium text-slate-500">등록일</th>
                                    <th className="px-6 py-4 text-right font-medium text-slate-500">관리</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-100">
                                {customers.filter(c => c.status === 'blacklist').map((customer) => (
                                    <tr key={customer.id} className="hover:bg-slate-50">
                                        <td className="px-6 py-4 font-medium text-slate-900">{customer.name}</td>
                                        <td className="px-6 py-4 text-slate-600">{customer.phone}</td>
                                        <td className="px-6 py-4 text-red-600 font-medium">{customer.reason}</td>
                                        <td className="px-6 py-4 text-slate-500">2024-03-20</td>
                                        <td className="px-6 py-4 text-right">
                                            <button className="text-slate-400 hover:text-slate-600">해제</button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </PageContainer>
    );
};

export default CustomerPage;
