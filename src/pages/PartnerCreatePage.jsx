import React from 'react';
import { ArrowLeft, Save, MapPin, Phone, Mail, Globe, User, Building } from 'lucide-react';
import { Link } from 'react-router-dom';
import PageContainer from '../components/PageContainer';

const PartnerCreatePage = () => {
    return (
        <PageContainer>
            <div className="max-w-3xl mx-auto space-y-6">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <Link to="/partners" className="p-2 hover:bg-slate-100 rounded-full text-slate-500 transition-colors">
                            <ArrowLeft size={24} />
                        </Link>
                        <h1 className="text-2xl font-bold text-slate-900">파트너 등록</h1>
                    </div>
                    <button className="px-6 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-medium shadow-sm flex items-center gap-2">
                        <Save size={18} />
                        저장
                    </button>
                </div>

                <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-6">
                    <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                        <Building size={20} className="text-slate-400" />
                        업체 정보
                    </h2>
                    <div className="grid grid-cols-1 gap-6">
                        <div className="space-y-2">
                            <label className="text-sm font-medium text-slate-700">업체명</label>
                            <input type="text" className="w-full px-3 py-2 border border-slate-200 rounded-lg" placeholder="업체 이름을 입력하세요" />
                        </div>

                        <div className="grid grid-cols-2 gap-6">
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-slate-700">지역</label>
                                <div className="relative">
                                    <MapPin className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                                    <select className="w-full pl-10 pr-4 py-2 border border-slate-200 rounded-lg appearance-none bg-white">
                                        <option>지역 선택</option>
                                        <option>다낭</option>
                                        <option>나트랑</option>
                                        <option>푸꾸옥</option>
                                        <option>방콕</option>
                                        <option>오사카</option>
                                        <option>도쿄</option>
                                    </select>
                                </div>
                            </div>
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-slate-700">유형</label>
                                <select className="w-full px-3 py-2 border border-slate-200 rounded-lg bg-white">
                                    <option>랜드사</option>
                                    <option>가이드</option>
                                    <option>운송업체</option>
                                    <option>기타</option>
                                </select>
                            </div>
                        </div>

                        <div className="space-y-2">
                            <label className="text-sm font-medium text-slate-700">주소</label>
                            <input type="text" className="w-full px-3 py-2 border border-slate-200 rounded-lg" placeholder="상세 주소를 입력하세요" />
                        </div>

                        <div className="space-y-2">
                            <label className="text-sm font-medium text-slate-700">웹사이트</label>
                            <div className="relative">
                                <Globe className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                                <input type="text" className="w-full pl-10 pr-4 py-2 border border-slate-200 rounded-lg" placeholder="https://" />
                            </div>
                        </div>
                    </div>

                    <div className="border-t border-slate-100 pt-6">
                        <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2 mb-6">
                            <User size={20} className="text-slate-400" />
                            담당자 정보
                        </h2>
                        <div className="grid grid-cols-1 gap-6">
                            <div className="grid grid-cols-2 gap-6">
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">담당자명</label>
                                    <input type="text" className="w-full px-3 py-2 border border-slate-200 rounded-lg" placeholder="이름" />
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">직급</label>
                                    <input type="text" className="w-full px-3 py-2 border border-slate-200 rounded-lg" placeholder="직급" />
                                </div>
                            </div>

                            <div className="grid grid-cols-2 gap-6">
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">연락처</label>
                                    <div className="relative">
                                        <Phone className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                                        <input type="text" className="w-full pl-10 pr-4 py-2 border border-slate-200 rounded-lg" placeholder="전화번호" />
                                    </div>
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">이메일</label>
                                    <div className="relative">
                                        <Mail className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                                        <input type="email" className="w-full pl-10 pr-4 py-2 border border-slate-200 rounded-lg" placeholder="example@email.com" />
                                    </div>
                                </div>
                            </div>

                            <div className="space-y-2">
                                <label className="text-sm font-medium text-slate-700">메모</label>
                                <textarea className="w-full px-3 py-2 border border-slate-200 rounded-lg h-24 resize-none" placeholder="참고사항을 입력하세요"></textarea>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </PageContainer>
    );
};

export default PartnerCreatePage;
