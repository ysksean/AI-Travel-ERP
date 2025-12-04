import React, { useState } from 'react';
import { ArrowLeft, Printer, Download, Mail, FileText } from 'lucide-react';
import { useNavigate, useParams } from 'react-router-dom';
import PageContainer from '../components/PageContainer';

const DocumentPreviewPage = () => {
    const navigate = useNavigate();
    const { type, id } = useParams(); // type: invoice, quotation, land-request
    const [activeDoc, setActiveDoc] = useState('invoice');

    const handlePrint = () => {
        window.print();
    };

    return (
        <PageContainer>
            <div className="max-w-4xl mx-auto space-y-6 pb-20">
                {/* Header */}
                <div className="flex items-center justify-between print:hidden">
                    <div className="flex items-center gap-4">
                        <button onClick={() => navigate(-1)} className="p-2 hover:bg-slate-100 rounded-full text-slate-500 transition-colors">
                            <ArrowLeft size={24} />
                        </button>
                        <div>
                            <h1 className="text-2xl font-bold text-slate-900">문서 미리보기</h1>
                            <p className="text-slate-500 text-sm">자동 생성된 문서를 확인하고 출력하거나 발송할 수 있습니다.</p>
                        </div>
                    </div>
                    <div className="flex gap-3">
                        <button onClick={handlePrint} className="px-4 py-2 text-slate-600 font-medium hover:bg-slate-100 rounded-lg transition-colors flex items-center gap-2">
                            <Printer size={18} />
                            인쇄
                        </button>
                        <button className="px-4 py-2 text-slate-600 font-medium hover:bg-slate-100 rounded-lg transition-colors flex items-center gap-2">
                            <Download size={18} />
                            PDF 다운로드
                        </button>
                        <button className="px-6 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-medium shadow-sm flex items-center gap-2">
                            <Mail size={18} />
                            이메일 발송
                        </button>
                    </div>
                </div>

                {/* Tabs */}
                <div className="flex border-b border-slate-200 print:hidden">
                    <button
                        onClick={() => setActiveDoc('invoice')}
                        className={`px-6 py-3 text-sm font-medium border-b-2 transition-colors ${activeDoc === 'invoice' ? 'border-primary-600 text-primary-600' : 'border-transparent text-slate-500 hover:text-slate-700'}`}
                    >
                        인보이스 (청구서)
                    </button>
                    <button
                        onClick={() => setActiveDoc('quotation')}
                        className={`px-6 py-3 text-sm font-medium border-b-2 transition-colors ${activeDoc === 'quotation' ? 'border-primary-600 text-primary-600' : 'border-transparent text-slate-500 hover:text-slate-700'}`}
                    >
                        견적서 (고객용)
                    </button>
                    <button
                        onClick={() => setActiveDoc('land-request')}
                        className={`px-6 py-3 text-sm font-medium border-b-2 transition-colors ${activeDoc === 'land-request' ? 'border-primary-600 text-primary-600' : 'border-transparent text-slate-500 hover:text-slate-700'}`}
                    >
                        수배 요청서 (랜드사)
                    </button>
                </div>

                {/* Document Preview Area */}
                <div className="bg-white shadow-lg p-12 min-h-[800px] print:shadow-none print:p-0">
                    {/* Invoice Template */}
                    {activeDoc === 'invoice' && (
                        <div className="space-y-8">
                            <div className="flex justify-between items-start border-b border-slate-200 pb-8">
                                <div>
                                    <h1 className="text-4xl font-bold text-slate-900">INVOICE</h1>
                                    <p className="text-slate-500 mt-2">청구서 번호: #INV-2024-001</p>
                                    <p className="text-slate-500">발행일: 2024. 03. 21</p>
                                </div>
                                <div className="text-right">
                                    <h2 className="text-xl font-bold text-slate-900">트래블 에이전시</h2>
                                    <p className="text-slate-500 text-sm mt-1">서울시 강남구 테헤란로 123</p>
                                    <p className="text-slate-500 text-sm">Tel: 02-1234-5678</p>
                                    <p className="text-slate-500 text-sm">Email: accounting@travel.com</p>
                                </div>
                            </div>

                            <div className="grid grid-cols-2 gap-12">
                                <div>
                                    <h3 className="text-sm font-bold text-slate-400 uppercase mb-2">Bill To</h3>
                                    <p className="font-bold text-slate-900 text-lg">김철수 고객님</p>
                                    <p className="text-slate-600">010-1234-5678</p>
                                    <p className="text-slate-600">kim@example.com</p>
                                </div>
                                <div className="text-right">
                                    <h3 className="text-sm font-bold text-slate-400 uppercase mb-2">Total Amount</h3>
                                    <p className="font-bold text-primary-600 text-3xl">₩2,700,000</p>
                                    <p className="text-slate-500 text-sm mt-1">납부 기한: 2024. 03. 28</p>
                                </div>
                            </div>

                            <table className="w-full mt-8">
                                <thead className="bg-slate-50 border-y border-slate-200">
                                    <tr>
                                        <th className="py-3 text-left font-medium text-slate-500">내역</th>
                                        <th className="py-3 text-center font-medium text-slate-500">수량</th>
                                        <th className="py-3 text-right font-medium text-slate-500">단가</th>
                                        <th className="py-3 text-right font-medium text-slate-500">금액</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-100">
                                    <tr>
                                        <td className="py-4 font-medium text-slate-900">일본 프리미엄 골프 투어 (성인)</td>
                                        <td className="py-4 text-center text-slate-600">2</td>
                                        <td className="py-4 text-right text-slate-600">₩1,200,000</td>
                                        <td className="py-4 text-right font-medium text-slate-900">₩2,400,000</td>
                                    </tr>
                                    <tr>
                                        <td className="py-4 font-medium text-slate-900">유류할증료</td>
                                        <td className="py-4 text-center text-slate-600">2</td>
                                        <td className="py-4 text-right text-slate-600">₩150,000</td>
                                        <td className="py-4 text-right font-medium text-slate-900">₩300,000</td>
                                    </tr>
                                </tbody>
                                <tfoot className="border-t border-slate-200">
                                    <tr>
                                        <td colSpan="3" className="py-4 text-right font-bold text-slate-900">합계</td>
                                        <td className="py-4 text-right font-bold text-slate-900">₩2,700,000</td>
                                    </tr>
                                </tfoot>
                            </table>

                            <div className="bg-slate-50 p-6 rounded-lg mt-8">
                                <h3 className="font-bold text-slate-900 mb-2">입금 계좌 안내</h3>
                                <p className="text-slate-600">신한은행 110-123-456789 (예금주: 트래블에이전시)</p>
                            </div>
                        </div>
                    )}

                    {/* Quotation Template */}
                    {activeDoc === 'quotation' && (
                        <div className="space-y-8">
                            <div className="text-center border-b border-slate-200 pb-8">
                                <h1 className="text-3xl font-bold text-slate-900">여행 견적서</h1>
                                <p className="text-slate-500 mt-2">Quotation for Travel Services</p>
                            </div>

                            {/* ... Quotation Content ... */}
                            <div className="text-center py-20 text-slate-400">
                                <FileText size={48} className="mx-auto mb-4 opacity-50" />
                                <p>견적서 양식이 여기에 표시됩니다.</p>
                            </div>
                        </div>
                    )}

                    {/* Land Request Template */}
                    {activeDoc === 'land-request' && (
                        <div className="space-y-8">
                            <div className="border-b border-slate-200 pb-8">
                                <h1 className="text-3xl font-bold text-slate-900">수배 요청서</h1>
                                <p className="text-slate-500 mt-2">Land Operator Request Form</p>
                            </div>

                            {/* ... Land Request Content ... */}
                            <div className="text-center py-20 text-slate-400">
                                <FileText size={48} className="mx-auto mb-4 opacity-50" />
                                <p>수배 요청서 양식이 여기에 표시됩니다.</p>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </PageContainer>
    );
};

export default DocumentPreviewPage;
