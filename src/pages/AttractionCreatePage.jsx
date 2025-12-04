import React from 'react';
import { ArrowLeft, Save, MapPin, Clock, Tag, Image as ImageIcon } from 'lucide-react';
import { Link } from 'react-router-dom';
import PageContainer from '../components/PageContainer';

const AttractionCreatePage = () => {
    return (
        <PageContainer>
            <div className="max-w-3xl mx-auto space-y-6">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <Link to="/attractions" className="p-2 hover:bg-slate-100 rounded-full text-slate-500 transition-colors">
                            <ArrowLeft size={24} />
                        </Link>
                        <h1 className="text-2xl font-bold text-slate-900">관광지 등록</h1>
                    </div>
                    <button className="px-6 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-medium shadow-sm flex items-center gap-2">
                        <Save size={18} />
                        저장
                    </button>
                </div>

                <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-6">
                    <h2 className="text-lg font-bold text-slate-900">기본 정보</h2>
                    <div className="grid grid-cols-1 gap-6">
                        <div className="space-y-2">
                            <label className="text-sm font-medium text-slate-700">관광지명</label>
                            <input type="text" className="w-full px-3 py-2 border border-slate-200 rounded-lg" placeholder="관광지 이름을 입력하세요" />
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
                                <div className="relative">
                                    <Tag className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                                    <select className="w-full pl-10 pr-4 py-2 border border-slate-200 rounded-lg appearance-none bg-white">
                                        <option>유형 선택</option>
                                        <option>테마파크</option>
                                        <option>유적지</option>
                                        <option>자연경관</option>
                                        <option>쇼핑</option>
                                        <option>문화체험</option>
                                    </select>
                                </div>
                            </div>
                        </div>

                        <div className="space-y-2">
                            <label className="text-sm font-medium text-slate-700">주소</label>
                            <input type="text" className="w-full px-3 py-2 border border-slate-200 rounded-lg" placeholder="상세 주소를 입력하세요" />
                        </div>

                        <div className="space-y-2">
                            <label className="text-sm font-medium text-slate-700">소요시간 (예상)</label>
                            <div className="relative">
                                <Clock className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                                <input type="text" className="w-full pl-10 pr-4 py-2 border border-slate-200 rounded-lg" placeholder="예: 2시간" />
                            </div>
                        </div>

                        <div className="space-y-2">
                            <label className="text-sm font-medium text-slate-700">설명</label>
                            <textarea className="w-full px-3 py-2 border border-slate-200 rounded-lg h-32 resize-none" placeholder="관광지에 대한 설명을 입력하세요"></textarea>
                        </div>

                        <div className="space-y-2">
                            <label className="text-sm font-medium text-slate-700">대표 이미지</label>
                            <div className="border-2 border-dashed border-slate-200 rounded-lg p-8 flex flex-col items-center justify-center text-slate-400 hover:bg-slate-50 hover:border-primary-400 transition-colors cursor-pointer">
                                <ImageIcon size={32} className="mb-2" />
                                <span className="text-sm">이미지를 드래그하거나 클릭하여 업로드</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </PageContainer>
    );
};

export default AttractionCreatePage;
