import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import MainLayout from './layouts/MainLayout';
import DashboardPage from './pages/DashboardPage';
import ProductListPage from './pages/ProductListPage';
import ProductCreatePage from './pages/ProductCreatePage';
import ProductDetailPage from './pages/ProductDetailPage';
import ReservationListPage from './pages/ReservationListPage';
import ReservationDetailPage from './pages/ReservationDetailPage';
import QuotationListPage from './pages/QuotationListPage';
import QuotationCreatePage from './pages/QuotationCreatePage';
import QuotationDetailPage from './pages/QuotationDetailPage';
import HotelListPage from './pages/HotelListPage';
import HotelCreatePage from './pages/HotelCreatePage';
import AttractionListPage from './pages/AttractionListPage';
import AttractionCreatePage from './pages/AttractionCreatePage';
import PartnerListPage from './pages/PartnerListPage';
import PartnerCreatePage from './pages/PartnerCreatePage';
import PaymentPage from './pages/PaymentPage';
import FlightPage from './pages/FlightPage';
import CustomerPage from './pages/CustomerPage';
import DocumentPreviewPage from './pages/DocumentPreviewPage';
import SettingsPage from './pages/SettingsPage';
import FinancePage from './pages/FinancePage';

function App() {
  return (
    <Router>
      <MainLayout>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/products" element={<ProductListPage />} />
          <Route path="/products/new" element={<ProductCreatePage />} />
          <Route path="/products/:id" element={<ProductDetailPage />} />
          <Route path="/reservations" element={<ReservationListPage />} />
          <Route path="/reservations/:id" element={<ReservationDetailPage />} />
          <Route path="/quotations" element={<QuotationListPage />} />
          <Route path="/quotations/new" element={<QuotationCreatePage />} />
          <Route path="/quotations/:id" element={<QuotationDetailPage />} />
          <Route path="/hotels" element={<HotelListPage />} />
          <Route path="/hotels/new" element={<HotelCreatePage />} />
          <Route path="/attractions" element={<AttractionListPage />} />
          <Route path="/attractions/new" element={<AttractionCreatePage />} />
          <Route path="/partners" element={<PartnerListPage />} />
          <Route path="/partners/new" element={<PartnerCreatePage />} />
          <Route path="/payments" element={<PaymentPage />} />
          <Route path="/flights" element={<FlightPage />} />
          <Route path="/customers" element={<CustomerPage />} />
          <Route path="/documents" element={<DocumentPreviewPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/finance" element={<FinancePage />} />
          {/* Add other routes as we build them */}
          <Route path="*" element={<div className="text-center mt-20 text-slate-400">Page under construction</div>} />
        </Routes>
      </MainLayout>
    </Router>
  );
}

export default App;
