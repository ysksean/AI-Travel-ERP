import React, { useState } from 'react';
import {
    LayoutDashboard,
    Package,
    CalendarDays,
    Users,
    FileText,
    Settings,
    Search,
    Bell,
    Menu,
    Plane,
    CreditCard,
    DollarSign,
    Hotel,
    MapPin,
    Handshake
} from 'lucide-react';
import { Link, useLocation } from 'react-router-dom';
import clsx from 'clsx';

const SidebarItem = ({ icon: Icon, label, to, active }) => {
    return (
        <Link
            to={to}
            className={clsx(
                "flex items-center gap-3 px-4 py-3 rounded-lg transition-colors text-sm font-medium",
                active
                    ? "bg-primary-50 text-primary-700"
                    : "text-slate-600 hover:bg-slate-50 hover:text-slate-900"
            )}
        >
            <Icon size={20} />
            <span>{label}</span>
        </Link>
    );
};

const MainLayout = ({ children }) => {
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    const location = useLocation();

    const navItems = [
        { icon: LayoutDashboard, label: '대시보드', to: '/' },
        { icon: Package, label: '상품 관리', to: '/products' },
        { icon: CalendarDays, label: '예약 관리', to: '/reservations' },
        { icon: FileText, label: '견적 관리', to: '/quotations' },
        { icon: CreditCard, label: '전자결제', to: '/payments' },
        { icon: DollarSign, label: '정산 관리', to: '/finance' },
        { icon: Plane, label: '항공 관리', to: '/flights' },
        { icon: Hotel, label: '호텔 관리', to: '/hotels' },
        { icon: MapPin, label: '관광지 관리', to: '/attractions' },
        { icon: Handshake, label: '파트너 관리', to: '/partners' },
        { icon: Users, label: '고객 관리', to: '/customers' },
        { icon: Settings, label: '설정', to: '/settings' },
    ];

    return (
        <div className="flex h-screen bg-slate-50 overflow-hidden font-sans">
            {/* Sidebar */}
            <aside
                className={clsx(
                    "bg-white border-r border-slate-200 flex-shrink-0 transition-all duration-300 flex flex-col",
                    isSidebarOpen ? "w-64" : "w-20"
                )}
            >
                <div className="h-16 flex items-center px-6 border-b border-slate-100">
                    <div className="flex items-center gap-2 text-primary-600 font-bold text-xl">
                        <Plane className="fill-current" />
                        {isSidebarOpen && <span>TravelERP</span>}
                    </div>
                </div>

                <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
                    {navItems.map((item) => (
                        <SidebarItem
                            key={item.label}
                            {...item}
                            active={location.pathname === item.to || (item.to !== '/' && location.pathname.startsWith(item.to))}
                        />
                    ))}
                </nav>

                <div className="p-4 border-t border-slate-100">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-slate-200 flex items-center justify-center text-slate-500 font-bold">
                            AD
                        </div>
                        {isSidebarOpen && (
                            <div className="text-sm">
                                <p className="font-medium text-slate-900">관리자</p>
                                <p className="text-slate-500 text-xs">admin@travel.com</p>
                            </div>
                        )}
                    </div>
                </div>
            </aside>

            {/* Main Content */}
            <div className="flex-1 flex flex-col min-w-0">
                {/* Header */}
                <header className="h-16 bg-white border-b border-slate-200 flex items-center justify-between px-6 flex-shrink-0">
                    <button
                        onClick={() => setIsSidebarOpen(!isSidebarOpen)}
                        className="p-2 hover:bg-slate-100 rounded-lg text-slate-500"
                    >
                        <Menu size={20} />
                    </button>

                    <div className="flex items-center gap-4">
                        <div className="relative hidden md:block">
                            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                            <input
                                type="text"
                                placeholder="예약, 상품 검색..."
                                className="pl-10 pr-4 py-2 bg-slate-100 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 w-64 transition-all"
                            />
                        </div>
                        <button className="p-2 hover:bg-slate-100 rounded-lg text-slate-500 relative">
                            <Bell size={20} />
                            <span className="absolute top-2 right-2 w-2 h-2 bg-red-500 rounded-full"></span>
                        </button>
                    </div>
                </header>

                {/* Page Content */}
                <main className="flex-1 overflow-y-auto p-6">
                    {children}
                </main>
            </div>
        </div>
    );
};

export default MainLayout;
