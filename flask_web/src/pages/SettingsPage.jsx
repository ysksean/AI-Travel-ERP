import React, { useState } from 'react';
import { User, Bell, Lock, Monitor, HardDrive, Save } from 'lucide-react';
import PageContainer from '../components/PageContainer';

const SettingsPage = () => {
    const [activeTab, setActiveTab] = useState('profile');

    const tabs = [
        { id: 'profile', label: '프로필', icon: User },
        { id: 'notifications', label: '알림', icon: Bell },
        { id: 'security', label: '보안', icon: Lock },
        { id: 'appearance', label: '화면', icon: Monitor },
        { id: 'system', label: '시스템', icon: HardDrive },
    ];

    return (
        <PageContainer>
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                {/* Sidebar */}
                <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm h-fit">
                    <nav className="space-y-1">
                        {tabs.map((tab) => {
                            const Icon = tab.icon;
                            return (
                                <button
                                    key={tab.id}
                                    onClick={() => setActiveTab(tab.id)}
                                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-colors ${activeTab === tab.id
                                        ? 'bg-primary-50 text-primary-700'
                                        : 'text-slate-600 hover:bg-slate-50'
                                        }`}
                                >
                                    <Icon size={18} />
                                    {tab.label}
                                </button>
                            );
                        })}
                    </nav>
                </div>

                {/* Content */}
                <div className="lg:col-span-3 bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                    {activeTab === 'profile' && (
                        <div className="space-y-6">
                            <h2 className="text-lg font-bold text-slate-900">프로필 설정</h2>
                            <div className="flex items-center gap-6">
                                <div className="w-24 h-24 rounded-full bg-slate-200 flex items-center justify-center text-slate-500 font-bold text-2xl">
                                    AD
                                </div>
                                <button className="px-4 py-2 bg-slate-100 hover:bg-slate-200 rounded-lg text-sm font-medium">
                                    사진 변경
                                </button>
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">이름</label>
                                    <input type="text" defaultValue="관리자" className="w-full px-3 py-2 border border-slate-200 rounded-lg" />
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">이메일</label>
                                    <input type="email" defaultValue="admin@travel.com" className="w-full px-3 py-2 border border-slate-200 rounded-lg" />
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">전화번호</label>
                                    <input type="text" defaultValue="010-0000-0000" className="w-full px-3 py-2 border border-slate-200 rounded-lg" />
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">부서</label>
                                    <select className="w-full px-3 py-2 border border-slate-200 rounded-lg">
                                        <option>영업팀</option>
                                        <option>수배팀</option>
                                        <option>관리팀</option>
                                    </select>
                                </div>
                            </div>
                            <button className="px-6 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-medium flex items-center gap-2">
                                <Save size={18} />
                                저장
                            </button>
                        </div>
                    )}

                    {activeTab === 'notifications' && (
                        <div className="space-y-6">
                            <h2 className="text-lg font-bold text-slate-900">알림 설정</h2>
                            <div className="space-y-4">
                                {[
                                    { label: '새로운 예약 알림', desc: '웹사이트를 통해 예약이 들어오면 알림을 받습니다.' },
                                    { label: '결제 알림', desc: '고객 결제가 완료되면 알림을 받습니다.' },
                                    { label: '출발 D-Day 알림', desc: '출발 7일 전, 3일 전에 알림을 받습니다.' },
                                    { label: '이메일 알림', desc: '중요한 알림을 이메일로도 받습니다.' },
                                ].map((item, idx) => (
                                    <div key={idx} className="flex items-center justify-between p-4 bg-slate-50 rounded-lg">
                                        <div>
                                            <p className="font-medium text-slate-900">{item.label}</p>
                                            <p className="text-sm text-slate-500 mt-1">{item.desc}</p>
                                        </div>
                                        <div className="w-11 h-6 bg-primary-600 rounded-full relative cursor-pointer">
                                            <div className="absolute right-1 top-1 w-4 h-4 bg-white rounded-full shadow-sm"></div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {activeTab === 'security' && (
                        <div className="space-y-6">
                            <h2 className="text-lg font-bold text-slate-900">보안 설정</h2>
                            <div className="space-y-4">
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">현재 비밀번호</label>
                                    <input type="password" className="w-full px-3 py-2 border border-slate-200 rounded-lg" />
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">새 비밀번호</label>
                                    <input type="password" className="w-full px-3 py-2 border border-slate-200 rounded-lg" />
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">비밀번호 확인</label>
                                    <input type="password" className="w-full px-3 py-2 border border-slate-200 rounded-lg" />
                                </div>
                                <button className="px-6 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-medium">
                                    비밀번호 변경
                                </button>
                            </div>
                        </div>
                    )}

                    {activeTab === 'appearance' && (
                        <div className="space-y-6">
                            <h2 className="text-lg font-bold text-slate-900">화면 설정</h2>
                            <div className="space-y-4">
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">테마</label>
                                    <div className="grid grid-cols-3 gap-4">
                                        {['라이트', '다크', '자동'].map((theme) => (
                                            <div key={theme} className="p-4 border-2 border-primary-500 rounded-lg cursor-pointer hover:bg-slate-50">
                                                <p className="text-center font-medium">{theme}</p>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">언어</label>
                                    <select className="w-full px-3 py-2 border border-slate-200 rounded-lg">
                                        <option>한국어</option>
                                        <option>English</option>
                                        <option>日本語</option>
                                    </select>
                                </div>
                            </div>
                        </div>
                    )}

                    {activeTab === 'system' && (
                        <div className="space-y-6">
                            <h2 className="text-lg font-bold text-slate-900">시스템 설정</h2>
                            <div className="space-y-4">
                                <div className="p-4 bg-slate-50 rounded-lg">
                                    <div className="flex justify-between items-center">
                                        <div>
                                            <p className="font-medium text-slate-900">버전</p>
                                            <p className="text-sm text-slate-500 mt-1">TravelERP v1.0.0</p>
                                        </div>
                                    </div>
                                </div>
                                <div className="p-4 bg-slate-50 rounded-lg">
                                    <div className="flex justify-between items-center">
                                        <div>
                                            <p className="font-medium text-slate-900">데이터 백업</p>
                                            <p className="text-sm text-slate-500 mt-1">마지막 백업: 2025-11-27</p>
                                        </div>
                                        <button className="px-4 py-2 bg-slate-100 hover:bg-slate-200 rounded-lg text-sm font-medium">
                                            백업 실행
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </PageContainer>
    );
};

export default SettingsPage;
