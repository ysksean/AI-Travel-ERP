import React, { useState } from 'react';
import { ArrowLeft, Save, Upload, FileText, Trash2, Calendar, MapPin, Clock, Plane, Plus, Hotel, Camera, Image as ImageIcon } from 'lucide-react';
import { Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import PageContainer from '../components/PageContainer';

const ProductCreatePage = () => {
    const [isDragging, setIsDragging] = useState(false);
    const [files, setFiles] = useState([]);
    const [itinerary, setItinerary] = useState([
        { day: 1, title: '인천 출발' },
        { day: 2, title: '전일 자유 일정' },
        { day: 3, title: '인천 도착' }
    ]);

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = () => {
        setIsDragging(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);
        const droppedFiles = Array.from(e.dataTransfer.files);
        setFiles(prev => [...prev, ...droppedFiles]);
        setTimeout(() => {
            alert("AI 파싱 완료! 필드가 자동으로 입력되었습니다.");
        }, 1000);
    };

    return (
        <PageContainer>
            <div className="max-w-5xl mx-auto space-y-8 pb-20">
                {/* Header */}
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <Link to="/products" className="p-2 hover:bg-slate-100 rounded-full text-slate-500 transition-colors">
                            <ArrowLeft size={24} />
                        </Link>
                        <div>
                            <h1 className="text-2xl font-bold text-slate-900">신규 상품 등록</h1>
                            <p className="text-slate-500 text-sm">일정표 파일을 업로드하거나 직접 입력하세요.</p>
                        </div>
                    </div>
                    <button className="px-4 py-2 text-slate-600 font-medium hover:bg-slate-100 rounded-lg transition-colors">
                        PDF 내보내기
                    </button>
                    <button className="px-4 py-2 text-slate-600 font-medium hover:bg-slate-100 rounded-lg transition-colors">
                        임시 저장
                    </button>
                    <button className="px-6 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-medium shadow-sm shadow-primary-200 flex items-center gap-2">
                        <Save size={18} />
                        상품 게시
                    </button>
                </div>


                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    {/* Left Column: AI Input & Basic Info */}
                    <div className="lg:col-span-2 space-y-6">

                        {/* AI Drop Zone */}
                        <div
                            className={`relative border-2 border-dashed rounded-2xl p-8 transition-all duration-300 ${isDragging
                                ? 'border-primary-500 bg-primary-50 scale-[1.02]'
                                : 'border-slate-300 hover:border-primary-400 hover:bg-slate-50'
                                }`}
                            onDragOver={handleDragOver}
                            onDragLeave={handleDragLeave}
                            onDrop={handleDrop}
                        >
                            <div className="flex flex-col items-center justify-center text-center space-y-4">
                                <div className={`w-16 h-16 rounded-full flex items-center justify-center transition-colors ${isDragging ? 'bg-primary-100 text-primary-600' : 'bg-slate-100 text-slate-400'
                                    }`}>
                                    <Upload size={32} />
                                </div>
                                <div>
                                    <h3 className="text-lg font-semibold text-slate-900">AI 스마트 등록</h3>
                                    <p className="text-slate-500 text-sm mt-1">
                                        일정표 파일(Word, Excel, PDF)을 여기에 드래그하세요. <br />
                                        <span className="text-primary-600 font-medium">AI가 자동으로 내용을 분석하여 입력합니다.</span>
                                    </p>
                                </div>
                            </div>

                            {/* File List */}
                            <AnimatePresence>
                                {files.length > 0 && (
                                    <motion.div
                                        initial={{ opacity: 0, y: 10 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        className="mt-6 space-y-2"
                                    >
                                        {files.map((file, idx) => (
                                            <div key={idx} className="flex items-center justify-between p-3 bg-white rounded-lg border border-slate-200 shadow-sm">
                                                <div className="flex items-center gap-3">
                                                    <FileText className="text-blue-500" size={20} />
                                                    <span className="text-sm font-medium text-slate-700">{file.name}</span>
                                                </div>
                                                <button className="text-slate-400 hover:text-red-500">
                                                    <Trash2 size={16} />
                                                </button>
                                            </div>
                                        ))}
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </div>

                        {/* Basic Info Form */}
                        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-6">
                            <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                                <FileText size={20} className="text-slate-400" />
                                기본 정보
                            </h2>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">상품명</label>
                                    <input type="text" className="w-full px-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent" placeholder="예: 일본 프리미엄 골프 투어" />
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">상품코드</label>
                                    <input type="text" className="w-full px-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent" placeholder="자동 생성 또는 직접 입력" />
                                </div>

                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">행사기간</label>
                                    <div className="relative">
                                        <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                                        <input type="text" className="w-full pl-10 pr-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent" placeholder="YYYY/MM/DD ~ YYYY/MM/DD" />
                                    </div>
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">지역</label>
                                    <div className="relative">
                                        <MapPin className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                                        <select className="w-full pl-10 pr-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent appearance-none bg-white">
                                            <option>지역 선택</option>
                                            <option>일본</option>
                                            <option>동남아</option>
                                            <option>유럽</option>
                                        </select>
                                    </div>
                                </div>

                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">박수 (일정)</label>
                                    <div className="relative">
                                        <Clock className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                                        <input type="text" className="w-full pl-10 pr-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent" placeholder="예: 3박 5일" />
                                    </div>
                                </div>

                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">상품 구분</label>
                                    <div className="flex p-1 bg-slate-100 rounded-lg">
                                        <button className="flex-1 py-1.5 text-sm font-medium rounded-md bg-white text-slate-900 shadow-sm">패키지</button>
                                        <button className="flex-1 py-1.5 text-sm font-medium rounded-md text-slate-500 hover:text-slate-900">항공</button>
                                        <button className="flex-1 py-1.5 text-sm font-medium rounded-md text-slate-500 hover:text-slate-900">호텔</button>
                                    </div>
                                </div>

                                {/* Flight Info */}
                                <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-6">
                                    <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                                        <Plane size={20} className="text-slate-400" />
                                        항공 정보
                                    </h2>
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                        <div className="space-y-2">
                                            <label className="text-sm font-medium text-slate-700">항공사</label>
                                            <input type="text" className="w-full px-4 py-2 border border-slate-200 rounded-lg" placeholder="예: 대한항공" />
                                        </div>
                                        <div className="space-y-2">
                                            <label className="text-sm font-medium text-slate-700">편명</label>
                                            <input type="text" className="w-full px-4 py-2 border border-slate-200 rounded-lg" placeholder="예: KE123" />
                                        </div>
                                        <div className="space-y-2">
                                            <label className="text-sm font-medium text-slate-700">출발 시간</label>
                                            <input type="time" className="w-full px-4 py-2 border border-slate-200 rounded-lg" />
                                        </div>
                                        <div className="space-y-2">
                                            <label className="text-sm font-medium text-slate-700">도착 시간</label>
                                            <input type="time" className="w-full px-4 py-2 border border-slate-200 rounded-lg" />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Image Upload */}
                        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-6">
                            <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                                <ImageIcon size={20} className="text-slate-400" />
                                대표 이미지
                            </h2>
                            <div className="border-2 border-dashed border-slate-200 rounded-lg p-8 flex flex-col items-center justify-center text-slate-400 hover:bg-slate-50 hover:border-primary-400 transition-colors cursor-pointer">
                                <ImageIcon size={32} className="mb-2" />
                                <span className="text-sm">이미지를 드래그하거나 클릭하여 업로드</span>
                            </div>
                        </div>

                        {/* Itinerary Editor */}
                        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-6">
                            <div className="flex items-center justify-between">
                                <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                                    <MapPin size={20} className="text-slate-400" />
                                    일정표 타임라인
                                </h2>
                                <button className="text-sm text-primary-600 font-medium hover:underline">+ 일차 추가</button>
                            </div>

                            <div className="space-y-6 relative before:absolute before:left-4 before:top-4 before:bottom-4 before:w-0.5 before:bg-slate-200">
                                {itinerary.map((day, idx) => (
                                    <div key={idx} className="relative pl-10">
                                        <div className="absolute left-0 top-0 w-8 h-8 rounded-full bg-primary-100 text-primary-600 flex items-center justify-center font-bold text-sm border-4 border-white shadow-sm">
                                            {day.day}
                                        </div>

                                        <div className="bg-slate-50 rounded-lg border border-slate-200 p-4 space-y-3">
                                            <div className="flex items-center justify-between">
                                                <input
                                                    type="text"
                                                    value={day.title}
                                                    className="bg-transparent font-semibold text-slate-900 focus:outline-none border-b border-transparent focus:border-primary-500 transition-colors"
                                                />
                                                <div className="flex gap-2">
                                                    <button className="p-1 text-slate-400 hover:text-primary-600"><Plus size={16} /></button>
                                                    <button className="p-1 text-slate-400 hover:text-red-500"><Trash2 size={16} /></button>
                                                </div>
                                            </div>

                                            <div className="flex gap-2">
                                                <div className="flex-1 h-24 border-2 border-dashed border-slate-200 rounded-lg flex items-center justify-center text-slate-400 text-xs hover:border-primary-300 hover:bg-white transition-colors cursor-pointer">
                                                    사진 드래그
                                                </div>
                                                <div className="flex-1 space-y-2">
                                                    <div className="bg-white rounded-lg border border-slate-200 p-2 text-xs text-slate-500 flex items-center gap-2">
                                                        <Hotel size={14} className="text-slate-400" />
                                                        <input type="text" placeholder="호텔 검색..." className="w-full focus:outline-none" />
                                                    </div>
                                                    <div className="bg-white rounded-lg border border-slate-200 p-2 text-xs text-slate-500 flex items-center gap-2">
                                                        <Camera size={14} className="text-slate-400" />
                                                        <input type="text" placeholder="관광지 검색..." className="w-full focus:outline-none" />
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Right Column: Settings & Status */}
                    <div className="space-y-6">
                        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-6 sticky top-6">
                            <h3 className="font-bold text-slate-900">설정</h3>

                            <div className="space-y-4">
                                <div className="flex items-center justify-between">
                                    <span className="text-sm font-medium text-slate-700">판매 상태</span>
                                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                                        판매중
                                    </span>
                                </div>

                                <div className="flex items-center justify-between">
                                    <span className="text-sm font-medium text-slate-700">노출 여부</span>
                                    <div className="w-11 h-6 bg-primary-600 rounded-full relative cursor-pointer">
                                        <div className="absolute right-1 top-1 w-4 h-4 bg-white rounded-full shadow-sm"></div>
                                    </div>
                                </div>

                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">담당자</label>
                                    <select className="w-full px-3 py-2 border border-slate-200 rounded-lg text-sm">
                                        <option>김철수 (영업팀)</option>
                                        <option>이영희 (수배팀)</option>
                                    </select>
                                </div>

                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-slate-700">노출 순서</label>
                                    <input type="number" className="w-full px-3 py-2 border border-slate-200 rounded-lg text-sm" defaultValue={500} />
                                </div>
                            </div>

                            <div className="pt-4 border-t border-slate-100">
                                <button className="w-full py-2 bg-slate-100 text-slate-600 rounded-lg text-sm font-medium hover:bg-slate-200 transition-colors">
                                    PDF 미리보기
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </PageContainer >
    );
};

export default ProductCreatePage;
