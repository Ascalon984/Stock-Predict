// Complete list of Indonesian stocks (IDX) with market data
export interface Stock {
    ticker: string;
    name: string;
    sector?: string;
    marketCap?: number; // in billions IDR
    lastPrice?: number;
    change?: number; // percentage change
    volume?: number; // in millions
}

// Helper to generate dummy market data (Deterministic to avoid hydration errors)
function generateMarketData(basePrice: number, volatility: number = 0.03): { lastPrice: number; change: number; volume: number } {
    // Create a deterministic "random" based on the basePrice itself
    // so it's consistent between server and client
    const seed = (basePrice * 17.12354) % 1;
    const seed2 = (basePrice * 91.5321) % 1;

    const changePercent = (seed - 0.5) * 2 * volatility * 100;
    const volume = Math.round(5 + seed2 * 150); // in millions

    return {
        lastPrice: Math.round(basePrice * (1 + (seed - 0.5) * 0.1)),
        change: Math.round(changePercent * 100) / 100,
        volume
    };
}

// Popular blue chip stocks with detailed data
export const BLUE_CHIP_STOCKS: Stock[] = [
    { ticker: "BBCA.JK", name: "PT Bank Central Asia Tbk", sector: "Banking", marketCap: 1250000, ...generateMarketData(9850) },
    { ticker: "BBRI.JK", name: "PT Bank Rakyat Indonesia (Persero) Tbk", sector: "Banking", marketCap: 780000, ...generateMarketData(5125) },
    { ticker: "BMRI.JK", name: "PT Bank Mandiri (Persero) Tbk", sector: "Banking", marketCap: 620000, ...generateMarketData(6450) },
    { ticker: "TLKM.JK", name: "PT Telkom Indonesia (Persero) Tbk", sector: "Telecom", marketCap: 380000, ...generateMarketData(3820) },
    { ticker: "ASII.JK", name: "PT Astra International Tbk", sector: "Automotive", marketCap: 210000, ...generateMarketData(5275) },
    { ticker: "BBNI.JK", name: "PT Bank Negara Indonesia (Persero) Tbk", sector: "Banking", marketCap: 185000, ...generateMarketData(5550) },
    { ticker: "ICBP.JK", name: "PT Indofood CBP Sukses Makmur Tbk", sector: "Food", marketCap: 145000, ...generateMarketData(11250) },
    { ticker: "INDF.JK", name: "PT Indofood Sukses Makmur Tbk", sector: "Food", marketCap: 88000, ...generateMarketData(7175) },
    { ticker: "KLBF.JK", name: "PT Kalbe Farma Tbk", sector: "Pharma", marketCap: 82000, ...generateMarketData(1565) },
    { ticker: "ADRO.JK", name: "PT Alamtri Resources Indonesia Tbk", sector: "Energy", marketCap: 120000, ...generateMarketData(2980) },
];

// Hot/trending stocks
export const TRENDING_STOCKS: Stock[] = [
    { ticker: "AMMN.JK", name: "PT Amman Mineral Internasional Tbk", sector: "Mining", marketCap: 450000, ...generateMarketData(10500) },
    { ticker: "BREN.JK", name: "PT Barito Renewables Energy Tbk", sector: "Energy", marketCap: 380000, ...generateMarketData(8200) },
    { ticker: "GOTO.JK", name: "PT GoTo Gojek Tokopedia Tbk", sector: "Technology", marketCap: 125000, ...generateMarketData(82) },
    { ticker: "BUKA.JK", name: "PT Bukalapak.com Tbk", sector: "Technology", marketCap: 18000, ...generateMarketData(145) },
    { ticker: "PANI.JK", name: "PT Pantai Indah Kapuk Dua Tbk", sector: "Property", marketCap: 85000, ...generateMarketData(12500) },
];

// Complete list with basic data
export const STOCK_LIST: Stock[] = [
    // Blue chips (already with full data)
    ...BLUE_CHIP_STOCKS,
    ...TRENDING_STOCKS,

    // Banking
    { ticker: "BBTN.JK", name: "PT Bank Tabungan Negara (Persero) Tbk", sector: "Banking", marketCap: 28000, ...generateMarketData(1375) },
    { ticker: "BRIS.JK", name: "PT Bank Syariah Indonesia Tbk", sector: "Banking", marketCap: 95000, ...generateMarketData(2750) },
    { ticker: "ARTO.JK", name: "PT Bank Jago Tbk", sector: "Banking", marketCap: 68000, ...generateMarketData(2480) },
    { ticker: "BDMN.JK", name: "PT Bank Danamon Indonesia Tbk", sector: "Banking", marketCap: 42000, ...generateMarketData(2850) },
    { ticker: "BNGA.JK", name: "PT Bank CIMB Niaga Tbk", sector: "Banking", marketCap: 38000, ...generateMarketData(1590) },
    { ticker: "BNII.JK", name: "PT Bank Maybank Indonesia Tbk", sector: "Banking", marketCap: 24000, ...generateMarketData(340) },
    { ticker: "BNLI.JK", name: "PT Bank Permata Tbk", sector: "Banking", marketCap: 35000, ...generateMarketData(1420) },
    { ticker: "BJBR.JK", name: "PT Bank Pembangunan Daerah Jawa Barat dan Banten Tbk", sector: "Banking", marketCap: 15000, ...generateMarketData(1545) },
    { ticker: "BJTM.JK", name: "PT Bank Pembangunan Daerah Jawa Timur Tbk", sector: "Banking", marketCap: 12000, ...generateMarketData(720) },

    // Mining & Energy
    { ticker: "ANTM.JK", name: "PT Aneka Tambang Tbk", sector: "Mining", marketCap: 48000, ...generateMarketData(1820) },
    { ticker: "PTBA.JK", name: "PT Bukit Asam Tbk", sector: "Mining", marketCap: 52000, ...generateMarketData(2860) },
    { ticker: "INCO.JK", name: "PT Vale Indonesia Tbk", sector: "Mining", marketCap: 62000, ...generateMarketData(4820) },
    { ticker: "TINS.JK", name: "PT Timah Tbk", sector: "Mining", marketCap: 8500, ...generateMarketData(1050) },
    { ticker: "MDKA.JK", name: "PT Merdeka Copper Gold Tbk", sector: "Mining", marketCap: 75000, ...generateMarketData(2580) },
    { ticker: "MEDC.JK", name: "PT Medco Energi Internasional Tbk", sector: "Energy", marketCap: 38000, ...generateMarketData(1520) },
    { ticker: "PGAS.JK", name: "PT Perusahaan Gas Negara Tbk", sector: "Energy", marketCap: 42000, ...generateMarketData(1445) },
    { ticker: "INDY.JK", name: "PT Indika Energy Tbk", sector: "Energy", marketCap: 25000, ...generateMarketData(1890) },
    { ticker: "GEMS.JK", name: "PT Golden Energy Mines Tbk", sector: "Mining", marketCap: 18000, ...generateMarketData(2150) },
    { ticker: "HRUM.JK", name: "PT Harum Energy Tbk", sector: "Mining", marketCap: 12000, ...generateMarketData(1350) },
    { ticker: "BYAN.JK", name: "PT Bayan Resources Tbk", sector: "Mining", marketCap: 185000, ...generateMarketData(18500) },

    // Consumer & Retail
    { ticker: "UNVR.JK", name: "PT Unilever Indonesia Tbk", sector: "Consumer", marketCap: 95000, ...generateMarketData(2480) },
    { ticker: "MYOR.JK", name: "PT Mayora Indah Tbk", sector: "Food", marketCap: 68000, ...generateMarketData(2820) },
    { ticker: "GOOD.JK", name: "PT Garudafood Putra Putri Jaya Tbk", sector: "Food", marketCap: 15000, ...generateMarketData(980) },
    { ticker: "ULTJ.JK", name: "PT Ultra Jaya Milk Industry & Trading Company Tbk", sector: "Food", marketCap: 22000, ...generateMarketData(1920) },
    { ticker: "HMSP.JK", name: "PT HM Sampoerna Tbk", sector: "Tobacco", marketCap: 82000, ...generateMarketData(710) },
    { ticker: "GGRM.JK", name: "PT Gudang Garam Tbk", sector: "Tobacco", marketCap: 55000, ...generateMarketData(28500) },
    { ticker: "MAPI.JK", name: "PT Mitra Adiperkasa Tbk", sector: "Retail", marketCap: 28000, ...generateMarketData(1650) },
    { ticker: "ACES.JK", name: "PT Aspirasi Hidup Indonesia Tbk", sector: "Retail", marketCap: 18000, ...generateMarketData(780) },
    { ticker: "ERAA.JK", name: "PT Erajaya Swasembada Tbk", sector: "Retail", marketCap: 12000, ...generateMarketData(420) },
    { ticker: "RALS.JK", name: "PT Ramayana Lestari Sentosa Tbk", sector: "Retail", marketCap: 4500, ...generateMarketData(650) },

    // Property & Infrastructure
    { ticker: "BSDE.JK", name: "PT Bumi Serpong Damai Tbk", sector: "Property", marketCap: 32000, ...generateMarketData(1080) },
    { ticker: "CTRA.JK", name: "PT Ciputra Development Tbk", sector: "Property", marketCap: 28000, ...generateMarketData(1240) },
    { ticker: "SMRA.JK", name: "PT Summarecon Agung Tbk", sector: "Property", marketCap: 16000, ...generateMarketData(485) },
    { ticker: "PWON.JK", name: "PT Pakuwon Jati Tbk", sector: "Property", marketCap: 24000, ...generateMarketData(485) },
    { ticker: "JRPT.JK", name: "PT Jaya Real Property Tbk", sector: "Property", marketCap: 8500, ...generateMarketData(580) },
    { ticker: "TBIG.JK", name: "PT Tower Bersama Infrastructure Tbk", sector: "Infrastructure", marketCap: 48000, ...generateMarketData(2120) },
    { ticker: "TOWR.JK", name: "PT Sarana Menara Nusantara Tbk", sector: "Infrastructure", marketCap: 72000, ...generateMarketData(1085) },
    { ticker: "JSMR.JK", name: "PT Jasa Marga (Persero) Tbk", sector: "Infrastructure", marketCap: 38000, ...generateMarketData(5250) },
    { ticker: "WIKA.JK", name: "PT Wijaya Karya (Persero) Tbk", sector: "Infrastructure", marketCap: 12000, ...generateMarketData(485) },
    { ticker: "PTPP.JK", name: "PT PP (Persero) Tbk", sector: "Infrastructure", marketCap: 8500, ...generateMarketData(580) },
    { ticker: "ADHI.JK", name: "PT Adhi Karya (Persero) Tbk", sector: "Infrastructure", marketCap: 6500, ...generateMarketData(395) },

    // Cement & Materials
    { ticker: "SMGR.JK", name: "PT Semen Indonesia (Persero) Tbk", sector: "Cement", marketCap: 48000, ...generateMarketData(4120) },
    { ticker: "INTP.JK", name: "PT Indocement Tunggal Prakarsa Tbk", sector: "Cement", marketCap: 42000, ...generateMarketData(11500) },
    { ticker: "SMCB.JK", name: "PT Solusi Bangun Indonesia Tbk", sector: "Cement", marketCap: 8500, ...generateMarketData(420) },

    // Pharma & Healthcare
    { ticker: "SIDO.JK", name: "PT Industri Jamu dan Farmasi Sido Muncul Tbk", sector: "Pharma", marketCap: 25000, ...generateMarketData(850) },
    { ticker: "KAEF.JK", name: "PT Kimia Farma Tbk", sector: "Pharma", marketCap: 5500, ...generateMarketData(655) },
    { ticker: "MIKA.JK", name: "PT Mitra Keluarga Karyasehat Tbk", sector: "Healthcare", marketCap: 38000, ...generateMarketData(2750) },

    // Technology
    { ticker: "DCII.JK", name: "PT DCI Indonesia Tbk", sector: "Technology", marketCap: 48000, ...generateMarketData(42500) },
    { ticker: "EMTK.JK", name: "PT Elang Mahkota Teknologi Tbk", sector: "Media", marketCap: 28000, ...generateMarketData(510) },
    { ticker: "MNCN.JK", name: "PT Media Nusantara Citra Tbk", sector: "Media", marketCap: 8500, ...generateMarketData(810) },

    // Automotive
    { ticker: "AUTO.JK", name: "PT Astra Otoparts Tbk", sector: "Automotive", marketCap: 12500, ...generateMarketData(2580) },
    { ticker: "SMSM.JK", name: "PT Selamat Sempurna Tbk", sector: "Automotive", marketCap: 22000, ...generateMarketData(1620) },
    { ticker: "IMAS.JK", name: "PT Indomobil Sukses Internasional Tbk", sector: "Automotive", marketCap: 5800, ...generateMarketData(785) },

    // Chemical & Petrochemical
    { ticker: "BRPT.JK", name: "PT Barito Pacific Tbk", sector: "Chemical", marketCap: 125000, ...generateMarketData(1180) },
    { ticker: "TPIA.JK", name: "PT Chandra Asri Pacific Tbk", sector: "Chemical", marketCap: 185000, ...generateMarketData(8450) },

    // Agriculture
    { ticker: "AALI.JK", name: "PT Astra Agro Lestari Tbk", sector: "Agriculture", marketCap: 18000, ...generateMarketData(6850) },
    { ticker: "LSIP.JK", name: "PT PP London Sumatra Indonesia Tbk", sector: "Agriculture", marketCap: 8500, ...generateMarketData(985) },
    { ticker: "CPIN.JK", name: "PT Charoen Pokphand Indonesia Tbk", sector: "Agriculture", marketCap: 82000, ...generateMarketData(5020) },
    { ticker: "JPFA.JK", name: "PT Japfa Comfeed Indonesia Tbk", sector: "Agriculture", marketCap: 18000, ...generateMarketData(1520) },

    // Finance (non-bank)
    { ticker: "ADMF.JK", name: "PT Adira Dinamika Multi Finance Tbk", sector: "Finance", marketCap: 18000, ...generateMarketData(9250) },
    { ticker: "BFIN.JK", name: "PT BFI Finance Indonesia Tbk", sector: "Finance", marketCap: 15000, ...generateMarketData(1085) },

    // Transportation
    { ticker: "BIRD.JK", name: "PT Blue Bird Tbk", sector: "Transportation", marketCap: 4800, ...generateMarketData(1550) },
    { ticker: "ASSA.JK", name: "PT Adi Sarana Armada Tbk", sector: "Transportation", marketCap: 8500, ...generateMarketData(970) },
    { ticker: "GIAA.JK", name: "PT Garuda Indonesia (Persero) Tbk", sector: "Aviation", marketCap: 4200, ...generateMarketData(58) },

    // Telecom
    { ticker: "ISAT.JK", name: "PT Indosat Ooredoo Hutchison Tbk", sector: "Telecom", marketCap: 72000, ...generateMarketData(8450) },
    { ticker: "EXCL.JK", name: "PT XL Axiata Tbk", sector: "Telecom", marketCap: 38000, ...generateMarketData(2280) },

    // Others with generated data
    { ticker: "AADI.JK", name: "PT Adaro Andalan Indonesia Tbk", sector: "Energy", marketCap: 85000, ...generateMarketData(5200) },
    { ticker: "AKRA.JK", name: "PT AKR Corporindo Tbk", sector: "Trade", marketCap: 28000, ...generateMarketData(1385) },
    { ticker: "INKP.JK", name: "PT Indah Kiat Pulp & Paper Tbk", sector: "Paper", marketCap: 95000, ...generateMarketData(8650) },
    { ticker: "SRTG.JK", name: "PT Saratoga Investama Sedaya Tbk", sector: "Investment", marketCap: 32000, ...generateMarketData(2480) },
    { ticker: "DSSA.JK", name: "PT Dian Swastatika Sentosa Tbk", sector: "Energy", marketCap: 38000, ...generateMarketData(42000) },
    { ticker: "ELSA.JK", name: "PT Elnusa Tbk", sector: "Energy", marketCap: 5500, ...generateMarketData(385) },
    { ticker: "POWR.JK", name: "PT Cikarang Listrindo Tbk", sector: "Energy", marketCap: 12000, ...generateMarketData(1120) },
    { ticker: "ROTI.JK", name: "PT Nippon Indosari Corpindo Tbk", sector: "Food", marketCap: 8500, ...generateMarketData(1085) },
    { ticker: "CLEO.JK", name: "PT Sariguna Primatirta Tbk", sector: "Consumer", marketCap: 6800, ...generateMarketData(1150) },
    { ticker: "KRAS.JK", name: "PT Krakatau Steel (Persero) Tbk", sector: "Steel", marketCap: 8500, ...generateMarketData(385) },
];

// Popular stocks for quick access (ticker symbols only)
export const POPULAR_STOCKS = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK",
    "BBNI.JK", "ICBP.JK", "INDF.JK", "KLBF.JK", "ADRO.JK",
    "ANTM.JK", "GOTO.JK", "PGAS.JK", "PTBA.JK", "SMGR.JK"
];

import {
    Landmark, Pickaxe, Zap, Radio, Cpu, ShoppingBag, Building2,
    HardHat, Utensils, Pill, Car
} from 'lucide-react';

// Sector categories with icons
export const SECTORS = [
    { name: "Banking", icon: Landmark, color: "#3B82F6" },
    { name: "Mining", icon: Pickaxe, color: "#F59E0B" },
    { name: "Energy", icon: Zap, color: "#EF4444" },
    { name: "Telecom", icon: Radio, color: "#8B5CF6" },
    { name: "Technology", icon: Cpu, color: "#10B981" },
    { name: "Consumer", icon: ShoppingBag, color: "#EC4899" },
    { name: "Property", icon: Building2, color: "#6366F1" },
    { name: "Infrastructure", icon: HardHat, color: "#14B8A6" },
    { name: "Food", icon: Utensils, color: "#F97316" },
    { name: "Pharma", icon: Pill, color: "#06B6D4" },
    { name: "Automotive", icon: Car, color: "#84CC16" },
];

// Format market cap for display
export function formatMarketCap(marketCap: number): string {
    if (marketCap >= 1000000) {
        return `${(marketCap / 1000000).toFixed(1)}Q`;
    } else if (marketCap >= 1000) {
        return `${(marketCap / 1000).toFixed(1)}T`;
    }
    return `${marketCap}B`;
}

// Format price for display
export function formatPrice(price: number): string {
    return price.toLocaleString('id-ID');
}

// Search function with improved matching
export function searchStocks(query: string, limit: number = 20): Stock[] {
    const q = query.toLowerCase().trim();
    if (!q) return [];

    // Score-based matching for better relevance
    const scored = STOCK_LIST.map(stock => {
        let score = 0;
        const tickerClean = stock.ticker.replace('.JK', '').toLowerCase();
        const nameClean = stock.name.toLowerCase();
        const sectorClean = (stock.sector || '').toLowerCase();

        // Exact ticker match (highest priority)
        if (tickerClean === q) score += 100;
        // Ticker starts with query
        else if (tickerClean.startsWith(q)) score += 50;
        // Ticker contains query
        else if (tickerClean.includes(q)) score += 30;

        // Name contains query
        if (nameClean.includes(q)) score += 20;

        // Sector match
        if (sectorClean.includes(q)) score += 15;

        // Boost blue chips
        if (BLUE_CHIP_STOCKS.find(s => s.ticker === stock.ticker)) score += 10;

        // Boost by market cap
        if (stock.marketCap) {
            score += Math.min(stock.marketCap / 100000, 5);
        }

        return { stock, score };
    });

    return scored
        .filter(s => s.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, limit)
        .map(s => s.stock);
}

// Get stocks by sector
export function getStocksBySector(sector: string): Stock[] {
    return STOCK_LIST.filter(s => s.sector === sector);
}

// Get top gainers
export function getTopGainers(limit: number = 5): Stock[] {
    return [...STOCK_LIST]
        .filter(s => s.change !== undefined)
        .sort((a, b) => (b.change || 0) - (a.change || 0))
        .slice(0, limit);
}

// Get top losers
export function getTopLosers(limit: number = 5): Stock[] {
    return [...STOCK_LIST]
        .filter(s => s.change !== undefined)
        .sort((a, b) => (a.change || 0) - (b.change || 0))
        .slice(0, limit);
}

// Get most active (by volume)
export function getMostActive(limit: number = 5): Stock[] {
    return [...STOCK_LIST]
        .filter(s => s.volume !== undefined)
        .sort((a, b) => (b.volume || 0) - (a.volume || 0))
        .slice(0, limit);
}
