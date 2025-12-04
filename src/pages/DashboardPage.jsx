import React from 'react';
import {
    TrendingUp,
    Users,
    Package,
    CreditCard,
    Calendar,
    ArrowUpRight,
    ArrowDownRight,
    Plus,
    FileText,
    DollarSign
} from 'lucide-react';
import { Link } from 'react-router-dom';
import PageContainer from '../components/PageContainer';

const StatCard = ({ title, value, change, trend, icon: Icon, color }) => {
    const isPositive = trend === 'up';
    return (
        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm hover:shadow-md transition-shadow">
            <div className="flex items-start justify-between">
                <div>
                    <p className="text-slate-500 text-sm font-medium">{title}</p>
                    <h3 className="text-2xl font-bold text-slate-900 mt-2">{value}</h3>
                    <div className={`flex items-center gap-1 mt-2 text-sm font-medium ${isPositive ? 'text-green-600' : 'text-red-600'
                        }`}>
                        {isPositive ? <ArrowUpRight size={16} /> : <ArrowDownRight size={16} />}
                        <span>{change}</span>
                    </div>
                </div>
                <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${color}`}>
                    <Icon size={24} className="text-white" />
                </div>
            </div>
        </div>
    );
};

const QuickActionButton = ({ label, icon: Icon, to, color }) => {
    return (
        <Link
            to={to}
            className={`flex flex-col items-center justify-center p-6 rounded-xl border-2 border-dashed ${color} hover:scale-105 transition-transform`}
        >
            <div className="w-12 h-12 rounded-full bg-white flex items-center justify-center mb-3 shadow-sm">
                <Icon size={24} className="text-slate-700" />
            </div>
            <span className="text-sm font-medium text-slate-700">{label}</span>
        </Link>
    );
};

const SimpleBarChart = () => {
    // Mock data for 6 months
    const data = [
        { label: '5월', value: 45 },
        { label: '6월', value: 60 },
        { label: '7월', value: 75 },
        { label: '8월', value: 90 },
        { label: '9월', value: 65 },
        { label: '10월', value: 80 },
    ];
    const max = 100;

    return (
        <div className="h-64 flex items-end justify-between gap-4 px-4 pb-2">
            {data.map((item, idx) => (
                <div key={idx} className="flex flex-col items-center gap-2 flex-1 group">
                    <div className="w-full bg-slate-100 rounded-t-lg relative h-48 flex items-end overflow-hidden">
                        <div
                            className="w-full bg-primary-500 rounded-t-lg transition-all duration-1000 ease-out group-hover:bg-primary-600"
                            style={{ height: `${(item.value / max) * 100}%` }}
                        ></div>
                        {/* Tooltip */}
                        <div className="absolute -top-8 left-1/2 -translate-x-1/2 bg-slate-800 text-white text-xs py-1 px-2 rounded opacity-0 group-hover:opacity-100 transition-opacity">
                            {item.value}건
                        </div>
                    </div>
                    <span className="text-xs text-slate-500 font-medium">{item.label}</span>
                </div>
            ))}
        </div>
    );
};

const DashboardPage = () => {
    const stats = [
        { title: '서비스 정보', value: '305,126 EP', change: '0 EP', trend: 'up', icon: TrendingUp, color: 'bg-blue-500' },
        { title: '온라인 예약 현황', value: '12 건', change: '이번달', trend: 'up', icon: Calendar, color: 'bg-green-500' },
        { title: '출발 예약 현황', value: '5 건', change: '이번주', trend: 'up', icon: Users, color: 'bg-purple-500' },
    ];

    const reservationStats = [
        { label: '입금대기 (상담완)', count: 3 },
        { label: '공동구매 (상담완)', count: 1 },
        { label: '결제대기 (수수료)', count: 5 },
    ];

    const documentStats = [
        { label: '자동결제서', count: 12 },
        { label: '결제대기서', count: 4 },
        { label: '정산결과안내서', count: 8 },
        { label: '정산서', count: 2 },
    ];

    return (
        <PageContainer className="space-y-6">
            {/* Header */}
            <div>
                <h1 className="text-2xl font-bold text-slate-900">Dashboard</h1>
                <p className="text-slate-500 text-sm mt-1">전체 현황을 한눈에 확인하세요.</p>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {stats.map((stat, idx) => (
                    <StatCard key={idx} {...stat} />
                ))}
            </div>

            {/* Quick Actions */}
            <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                <h2 className="text-lg font-bold text-slate-900 mb-4">빠른 작업</h2>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <QuickActionButton label="패키지 등록" icon={Package} to="/products/new" color="border-blue-200 hover:border-blue-400 hover:bg-blue-50" />
                    <QuickActionButton label="견적서 작성" icon={FileText} to="/quotations/new" color="border-green-200 hover:border-green-400 hover:bg-green-50" />
                    <QuickActionButton label="예약 등록" icon={Calendar} to="/reservations" color="border-purple-200 hover:border-purple-400 hover:bg-purple-50" />
                    <QuickActionButton label="정산 관리" icon={CreditCard} to="/finance" color="border-orange-200 hover:border-orange-400 hover:bg-orange-50" />
                </div>
            </div>

            {/* Two Column Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Reservation Status */}
                <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                    <div className="flex items-center justify-between mb-4">
                        <h2 className="text-lg font-bold text-slate-900">상담/결제 현황</h2>
                        <button className="text-sm text-primary-600 hover:underline">세부 보기</button>
                    </div>
                    <div className="space-y-3">
                        {reservationStats.map((item, idx) => (
                            <div key={idx} className="flex items-center justify-between p-3 bg-slate-50 rounded-lg hover:bg-slate-100 transition-colors cursor-pointer">
                                <span className="text-sm text-slate-700">{item.label}</span>
                                <span className="text-lg font-bold text-slate-900">{item.count}<span className="text-sm text-slate-500 ml-1">건</span></span>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Document Status */}
                <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                    <div className="flex items-center justify-between mb-4">
                        <h2 className="text-lg font-bold text-slate-900">전문문서 발행현황</h2>
                        <button className="text-sm text-primary-600 hover:underline">세부 보기</button>
                    </div>
                    <div className="space-y-3">
                        {documentStats.map((item, idx) => (
                            <div key={idx} className="flex items-center justify-between p-3 bg-slate-50 rounded-lg hover:bg-slate-100 transition-colors cursor-pointer">
                                <span className="text-sm text-slate-700">{item.label}</span>
                                <span className="text-lg font-bold text-slate-900">{item.count}<span className="text-sm text-slate-500 ml-1">건</span></span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Sales Chart */}
            <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                <div className="flex items-center justify-between mb-6">
                    <h2 className="text-lg font-bold text-slate-900">매출 통계 (최근 6개월)</h2>
                    <select className="text-sm border border-slate-200 rounded-lg px-3 py-1.5">
                        <option>이번 달</option>
                        <option>지난 달</option>
                        <option>올해</option>
                    </select>
                </div>
                <SimpleBarChart />
            </div>
        </PageContainer>
    );
};

export default DashboardPage;
