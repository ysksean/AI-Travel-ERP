import React, { useState } from 'react';
import { ArrowLeft, Calendar, MapPin, Users, Clock, Save, Plane, Hotel, Bus } from 'lucide-react';
import { Link, useParams } from 'react-router-dom';
import PageContainer from '../components/PageContainer';

const ProductDetailPage = () => {
    const { id } = useParams();
    const [activeTab, setActiveTab] = useState('itinerary');

    // Mock Data
    const product = {
        id: id || 'P20240315-01',
        name: '다낭 3박 4일 패키지',
        status: '판매중',
        region: '동남아',
        period: '2024.03.15 ~ 2024.03.18',
        manager: '김철수',
        price: 1250000,
        minPax: 4,
        maxPax: 20,
        description: '다낭의 아름다운 해변과 호이안의 고즈넉한 풍경을 즐기는 프리미엄 패키지입니다.',
        flights: [
            { type: 'departure', airline: 'Korean Air', flightNo: 'KE463', time: '18:40 - 21:20' },
            { type: 'return', airline: 'Korean Air', flightNo: 'KE464', time: '22:50 - 05:30(+1)' }
        ]
    };

    const itinerary = [
        { day: 1, title: '인천 출발 / 다낭 도착', activities: ['인천공항 미팅', '다낭 국제공항 도착', '가이드 미팅 후 호텔 이동'] },
        { day: 2, title: '다낭 시내 관광', activities: ['호텔 조식', '다낭 대성당', '한시장', '마블마운틴', '호이안 야간 투어'] },
        { day: 3, title: '바나힐 투어', activities: ['호텔 조식', '바나힐 테마파크', '골든브릿지', '프랑스 마을', '선짜반도 영흥사'] },
        { day: 4, title: '다낭 출발 / 인천 도착', activities: ['호텔 조식 및 체크아웃', '롯데마트 쇼핑', '공항 이동', '인천 도착'] },
    ];

    return (
        <PageContainer>
            <div className="space-y-6">
                {/* Header */}
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <Link to="/products" className="p-2 hover:bg-slate-100 rounded-full text-slate-500 transition-colors">
                            <ArrowLeft size={24} />
                        </Link>
                        <div>
                            <div className="flex items-center gap-3">
                                <h1 className="text-2xl font-bold text-slate-900">{product.name}</h1>
                                <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                                    {product.status}
                                </span>
                            </div>
                            <p className="text-slate-500 text-sm">{product.id} | {product.region}</p>
                        </div>
                    </div>
                    <div className="flex gap-3">
                        <button className="px-4 py-2 text-slate-600 font-medium hover:bg-slate-100 rounded-lg transition-colors">
                            복사
                        </button>
                        <button className="px-6 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-medium shadow-sm flex items-center gap-2">
                            <Save size={18} />
                            저장
                        </button>
                    </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Left Column: Product Details */}
                    <div className="lg:col-span-2 space-y-6">
                        {/* Basic Info */}
                        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                            <h2 className="text-lg font-bold text-slate-900 mb-4">기본 정보</h2>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">상품명</label>
                                    <input type="text" defaultValue={product.name} className="w-full px-3 py-2 border border-slate-200 rounded-lg" />
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">행사 기간</label>
                                    <div className="flex items-center gap-2">
                                        <Calendar size={18} className="text-slate-400" />
                                        <input type="text" defaultValue={product.period} className="w-full px-3 py-2 border border-slate-200 rounded-lg" />
                                    </div>
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">상품가</label>
                                    <input type="text" defaultValue={product.price.toLocaleString()} className="w-full px-3 py-2 border border-slate-200 rounded-lg text-right" />
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">담당자</label>
                                    <select className="w-full px-3 py-2 border border-slate-200 rounded-lg">
                                        <option>{product.manager}</option>
                                        <option>이영희</option>
                                        <option>박지민</option>
                                    </select>
                                </div>
                            </div>
                        </div>

                        {/* Itinerary */}
                        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                            <h2 className="text-lg font-bold text-slate-900 mb-4">일정표</h2>
                            <div className="space-y-6">
                                {itinerary.map((day) => (
                                    <div key={day.day} className="flex gap-4">
                                        <div className="flex flex-col items-center">
                                            <div className="w-8 h-8 rounded-full bg-primary-100 text-primary-700 flex items-center justify-center font-bold text-sm">
                                                {day.day}
                                            </div>
                                            <div className="w-0.5 h-full bg-slate-100 my-2"></div>
                                        </div>
                                        <div className="flex-1 pb-6">
                                            <h3 className="font-bold text-slate-900 mb-2">{day.title}</h3>
                                            <div className="bg-slate-50 rounded-lg p-4 space-y-2">
                                                {day.activities.map((activity, idx) => (
                                                    <div key={idx} className="flex items-center gap-2 text-sm text-slate-600">
                                                        <div className="w-1.5 h-1.5 rounded-full bg-slate-400"></div>
                                                        {activity}
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Right Column: Flight & Hotel */}
                    <div className="space-y-6">
                        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                            <h2 className="text-lg font-bold text-slate-900 mb-4 flex items-center gap-2">
                                <Plane size={20} className="text-slate-400" />
                                항공 정보
                            </h2>
                            <div className="space-y-4">
                                {product.flights.map((flight, idx) => (
                                    <div key={idx} className="p-3 border border-slate-200 rounded-lg bg-slate-50">
                                        <div className="flex justify-between items-center mb-2">
                                            <span className="text-xs font-bold text-slate-500">{flight.type === 'departure' ? '출국' : '귀국'}</span>
                                            <span className="text-xs font-bold text-slate-900">{flight.airline}</span>
                                        </div>
                                        <div className="flex justify-between items-center">
                                            <span className="font-mono text-slate-900">{flight.flightNo}</span>
                                            <span className="text-sm text-slate-600">{flight.time}</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                            <h2 className="text-lg font-bold text-slate-900 mb-4 flex items-center gap-2">
                                <Hotel size={20} className="text-slate-400" />
                                호텔 정보
                            </h2>
                            <div className="p-3 border border-slate-200 rounded-lg bg-slate-50">
                                <h3 className="font-bold text-slate-900">노보텔 다낭 프리미어 한 리버</h3>
                                <p className="text-sm text-slate-500 mt-1">Novotel Danang Premier Han River</p>
                                <div className="mt-3 flex gap-2">
                                    <span className="px-2 py-1 bg-white rounded text-xs border border-slate-200">5성급</span>
                                    <span className="px-2 py-1 bg-white rounded text-xs border border-slate-200">조식포함</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </PageContainer>
    );
};

export default ProductDetailPage;
