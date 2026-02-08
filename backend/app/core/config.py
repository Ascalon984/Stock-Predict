"""Application configuration and constants."""
from typing import List, Dict
import os

# ============================================
# Environment Configuration
# ============================================
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# ============================================
# API Configuration
# ============================================
API_VERSION = "2.0.0"
API_TITLE = "Stock Predictive Analytics API"
API_DESCRIPTION = """
## Hybrid Stock Prediction System

A sophisticated stock prediction API combining traditional statistical methods (SARIMA) 
with deep learning (LSTM with attention) for Indonesian Stock Exchange (IDX).

### Key Features

- **Real-time Data**: Live market data from Yahoo Finance
- **Auto-SARIMA**: Automatic parameter selection for optimal statistical forecasting
- **LSTM with Attention**: Deep learning model that focuses on important time patterns
- **Adaptive Ensemble**: Dynamic weight allocation based on model performance
- **Technical Analysis**: RSI, MACD, Bollinger Bands, Moving Averages, ATR
- **Sentiment Scoring**: Multi-signal sentiment analysis
- **WebSocket Support**: Real-time updates during market hours

### Prediction Horizons

- **1 Day**: Short-term intraday signals
- **7 Days**: Weekly trend direction
- **30 Days**: Monthly outlook

### Data Sources

- **Yahoo Finance API**: Real-time and historical OHLCV data
- **275+ Indonesian stocks** covering all major sectors

---

⚠️ **DISCLAIMER**: This system does NOT provide financial advice. 
All predictions are for informational purposes only. 
Past performance is not indicative of future results.
Always conduct your own research before making investment decisions.
"""

# ============================================
# Model Configuration
# ============================================
# SARIMA parameters (defaults, auto-ARIMA will optimize)
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 5)  # 5-day trading week seasonality
SARIMA_MAX_P = 3
SARIMA_MAX_Q = 3

# LSTM parameters
LSTM_SEQUENCE_LENGTH = 60
LSTM_UNITS = [128, 64]
LSTM_DENSE_UNITS = [32, 16]
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32
LSTM_DROPOUT = 0.2

# Ensemble configuration
DEFAULT_SARIMA_WEIGHT = 0.35
DEFAULT_LSTM_WEIGHT = 0.65

# Prediction horizons (days)
DEFAULT_PREDICTION_HORIZONS = [1, 7, 30]

# Data configuration
DEFAULT_LOOKBACK_PERIOD = "1y"
DEFAULT_LOOKBACK_DAYS = 365
MINIMUM_DATA_POINTS = 60

# Cache configuration
CACHE_TTL_MINUTES = 15
CACHE_MAX_SIZE = 100

# ============================================
# Stock Universe - Indonesian Stock Exchange (JK)
# ============================================
STOCK_UNIVERSE: List[str] = [
    # Banking
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "BBTN.JK", "BRIS.JK", "ARTO.JK",
    "BDMN.JK", "BNGA.JK", "BNII.JK", "BNLI.JK", "BJBR.JK", "BJTM.JK", "BBYB.JK",
    "BMAS.JK", "BINA.JK", "BTPN.JK", "AMAR.JK", "AGRO.JK", "AGRS.JK", "BABP.JK",
    "BACA.JK", "BBHI.JK", "BBKP.JK", "BBMD.JK", "BCIC.JK", "BEKS.JK", "BGTG.JK",
    "BNBA.JK", "BVIC.JK", "DNAR.JK", "MCOR.JK", "MEGA.JK", "NISP.JK", "NOBU.JK",
    "PNBN.JK", "SDRA.JK",
    
    # Mining & Energy
    "ADRO.JK", "ANTM.JK", "PTBA.JK", "INCO.JK", "TINS.JK", "MDKA.JK", "MEDC.JK",
    "PGAS.JK", "INDY.JK", "GEMS.JK", "HRUM.JK", "BYAN.JK", "BOSS.JK", "DEWA.JK",
    "DOID.JK", "ITMG.JK", "BRMS.JK", "CITA.JK", "AMMN.JK", "ARCI.JK", "ELSA.JK",
    "ENRG.JK", "FIRE.JK", "AADI.JK", "ADMR.JK", "AKSI.JK", "APEX.JK", "ARII.JK",
    "ARKO.JK", "ARTI.JK", "BIPI.JK", "BSSR.JK", "CBPE.JK", "DSSA.JK", "KKGI.JK",
    "NATO.JK", "POWR.JK", "PSAB.JK", "RAJA.JK", "SMMT.JK",
    
    # Consumer & Retail
    "ICBP.JK", "INDF.JK", "UNVR.JK", "MYOR.JK", "HMSP.JK", "GGRM.JK", "KLBF.JK",
    "MAPI.JK", "ACES.JK", "ERAA.JK", "RALS.JK", "GOOD.JK", "ULTJ.JK", "CLEO.JK",
    "SIDO.JK", "KAEF.JK", "MIKA.JK", "CAMP.JK", "DLTA.JK", "CEKA.JK", "AISA.JK",
    "ADES.JK", "ALTO.JK", "BATA.JK", "BEEF.JK", "BOBA.JK", "BUDI.JK", "CINT.JK",
    "DMND.JK", "DVLA.JK", "FAST.JK", "FISH.JK", "HOKI.JK", "KEJU.JK", "KICI.JK",
    "KINO.JK", "LSIP.JK", "MBTO.JK", "MLBI.JK", "MRAT.JK", "PEHA.JK", "PSDN.JK",
    "PYFA.JK", "ROTI.JK", "SKBM.JK", "SKLT.JK", "STTP.JK", "TCID.JK", "TSPC.JK",
    "WIIM.JK", "ZONE.JK",
    
    # Property & Infrastructure
    "BSDE.JK", "CTRA.JK", "SMRA.JK", "PWON.JK", "JRPT.JK", "PANI.JK", "LPKR.JK",
    "TBIG.JK", "TOWR.JK", "JSMR.JK", "WIKA.JK", "PTPP.JK", "ADHI.JK", "WSBP.JK",
    "DMAS.JK", "ASRI.JK", "APLN.JK", "BKSL.JK", "BEST.JK", "CMNP.JK", "COWL.JK",
    "DART.JK", "DILD.JK", "DGIK.JK", "EMDE.JK", "FMII.JK", "GAMA.JK", "KIJA.JK",
    "LPCK.JK", "MDLN.JK", "MKPI.JK", "MTLA.JK", "MTSM.JK", "NRCA.JK", "OMRE.JK",
    "PLIN.JK", "PPRO.JK", "SMDM.JK", "SSIA.JK", "TARA.JK", "TOTL.JK",
    
    # Telecom & Technology
    "TLKM.JK", "ISAT.JK", "EXCL.JK", "BUKA.JK", "GOTO.JK", "DCII.JK", "EMTK.JK",
    "MNCN.JK", "SCMA.JK", "IPTV.JK", "LINK.JK", "FREN.JK", "AREA.JK", "ASGR.JK",
    "ATIC.JK", "AXIO.JK", "CASH.JK", "DATA.JK", "DIVA.JK", "DOSS.JK", "EDGE.JK",
    "WIFI.JK", "ZBRA.JK",
    
    # Automotive
    "ASII.JK", "AUTO.JK", "GJTL.JK", "IMAS.JK", "SMSM.JK", "BRAM.JK", "GDYR.JK",
    "INDS.JK", "LPIN.JK", "MASA.JK", "NIPS.JK", "PRAS.JK",
    
    # Chemical & Manufacturing
    "BRPT.JK", "TPIA.JK", "SMGR.JK", "INTP.JK", "SMCB.JK", "INKP.JK", "TKIM.JK",
    "FASW.JK", "SRIL.JK", "ARNA.JK", "MARK.JK", "EKAD.JK", "DPNS.JK", "AGII.JK",
    "AKPI.JK", "AMFG.JK", "APLI.JK", "BTON.JK", "CTBN.JK", "IGAR.JK", "IMPC.JK",
    "JPFA.JK", "KBLI.JK", "MLIA.JK", "SRSN.JK", "TALF.JK", "TRST.JK", "UNIC.JK",
    "WTON.JK",
    
    # Agriculture
    "AALI.JK", "ANJT.JK", "BWPT.JK", "CPIN.JK", "DSNG.JK", "GZCO.JK", "JAWA.JK",
    "MAIN.JK", "PALM.JK", "SGRO.JK", "SIMP.JK", "SMAR.JK", "SSMS.JK", "TAPG.JK",
    "TBLA.JK", "UNSP.JK",
    
    # Finance (Non-Bank)
    "ADMF.JK", "BFIN.JK", "CFIN.JK", "MFIN.JK", "TIFA.JK", "VRNA.JK", "WOMF.JK",
    "APIC.JK", "ARKA.JK", "BCAP.JK", "BHIT.JK", "DEFI.JK", "PEGE.JK", "PNLF.JK",
    "SRTG.JK", "TRIM.JK",
    
    # Transportation & Logistics
    "BIRD.JK", "ASSA.JK", "GIAA.JK", "TAXI.JK", "SMDR.JK", "TMAS.JK", "IPCC.JK",
    "ALII.JK", "BLTA.JK", "BULL.JK", "DEAL.JK", "ELPI.JK", "GMFI.JK", "HITS.JK",
    "KARW.JK", "LEAD.JK", "LRNA.JK", "MBSS.JK", "NELY.JK", "SAFE.JK", "SAPX.JK",
    "SHIP.JK", "SOCI.JK", "TPMA.JK", "WEHA.JK",
    
    # Healthcare
    "CARE.JK", "HEAL.JK", "MTRA.JK", "PRDA.JK", "SILO.JK", "SAME.JK",
    
    # Others
    "FILM.JK", "KREN.JK", "LPGI.JK", "MDIA.JK", "PDES.JK", "POOL.JK", "PSKT.JK",
    "PURA.JK", "SCNP.JK", "SRAJ.JK", "STAR.JK"
]

# Stock name mapping (commonly used stocks)
STOCK_NAMES: Dict[str, str] = {
    # Banking
    "BBCA.JK": "PT Bank Central Asia Tbk",
    "BBRI.JK": "PT Bank Rakyat Indonesia (Persero) Tbk",
    "BMRI.JK": "PT Bank Mandiri (Persero) Tbk",
    "BBNI.JK": "PT Bank Negara Indonesia (Persero) Tbk",
    "BBTN.JK": "PT Bank Tabungan Negara (Persero) Tbk",
    "BRIS.JK": "PT Bank Syariah Indonesia Tbk",
    "ARTO.JK": "PT Bank Jago Tbk",
    "BDMN.JK": "PT Bank Danamon Indonesia Tbk",
    "BNGA.JK": "PT Bank CIMB Niaga Tbk",
    "PNBN.JK": "PT Bank Pan Indonesia Tbk",
    
    # Mining & Energy
    "ADRO.JK": "PT Alamtri Resources Indonesia Tbk",
    "ANTM.JK": "PT Aneka Tambang Tbk",
    "PTBA.JK": "PT Bukit Asam Tbk",
    "INCO.JK": "PT Vale Indonesia Tbk",
    "TINS.JK": "PT Timah Tbk",
    "MDKA.JK": "PT Merdeka Copper Gold Tbk",
    "MEDC.JK": "PT Medco Energi Internasional Tbk",
    "PGAS.JK": "PT Perusahaan Gas Negara Tbk",
    "AMMN.JK": "PT Amman Mineral Internasional Tbk",
    "BYAN.JK": "PT Bayan Resources Tbk",
    
    # Consumer
    "ICBP.JK": "PT Indofood CBP Sukses Makmur Tbk",
    "INDF.JK": "PT Indofood Sukses Makmur Tbk",
    "UNVR.JK": "PT Unilever Indonesia Tbk",
    "MYOR.JK": "PT Mayora Indah Tbk",
    "HMSP.JK": "PT HM Sampoerna Tbk",
    "GGRM.JK": "PT Gudang Garam Tbk",
    "KLBF.JK": "PT Kalbe Farma Tbk",
    "SIDO.JK": "PT Industri Jamu dan Farmasi Sido Muncul Tbk",
    
    # Telecom
    "TLKM.JK": "PT Telkom Indonesia (Persero) Tbk",
    "ISAT.JK": "PT Indosat Ooredoo Hutchison Tbk",
    "EXCL.JK": "PT XL Axiata Tbk",
    
    # Technology
    "GOTO.JK": "PT GoTo Gojek Tokopedia Tbk",
    "BUKA.JK": "PT Bukalapak.com Tbk",
    "DCII.JK": "PT DCI Indonesia Tbk",
    "EMTK.JK": "PT Elang Mahkota Teknologi Tbk",
    
    # Automotive
    "ASII.JK": "PT Astra International Tbk",
    "AUTO.JK": "PT Astra Otoparts Tbk",
    "GJTL.JK": "PT Gajah Tunggal Tbk",
    
    # Property
    "BSDE.JK": "PT Bumi Serpong Damai Tbk",
    "CTRA.JK": "PT Ciputra Development Tbk",
    "SMRA.JK": "PT Summarecon Agung Tbk",
    "PANI.JK": "PT Pantai Indah Kapuk Dua Tbk",
    
    # Infrastructure
    "TBIG.JK": "PT Tower Bersama Infrastructure Tbk",
    "TOWR.JK": "PT Sarana Menara Nusantara Tbk",
    "JSMR.JK": "PT Jasa Marga (Persero) Tbk",
    "WIKA.JK": "PT Wijaya Karya (Persero) Tbk",
    
    # Cement
    "SMGR.JK": "PT Semen Indonesia (Persero) Tbk",
    "INTP.JK": "PT Indocement Tunggal Prakarsa Tbk",
    "SMCB.JK": "PT Solusi Bangun Indonesia Tbk",
    
    # Chemical
    "BRPT.JK": "PT Barito Pacific Tbk",
    "TPIA.JK": "PT Chandra Asri Pacific Tbk",
}

# Sector classifications
SECTORS: Dict[str, List[str]] = {
    "Banking": ["BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "BBTN.JK", "BRIS.JK", "ARTO.JK"],
    "Mining": ["ADRO.JK", "ANTM.JK", "PTBA.JK", "INCO.JK", "TINS.JK", "MDKA.JK", "AMMN.JK"],
    "Energy": ["MEDC.JK", "PGAS.JK", "BYAN.JK", "ELSA.JK", "POWR.JK"],
    "Consumer": ["ICBP.JK", "INDF.JK", "UNVR.JK", "MYOR.JK", "KLBF.JK", "SIDO.JK"],
    "Telecom": ["TLKM.JK", "ISAT.JK", "EXCL.JK"],
    "Technology": ["GOTO.JK", "BUKA.JK", "DCII.JK", "EMTK.JK"],
    "Automotive": ["ASII.JK", "AUTO.JK", "GJTL.JK", "IMAS.JK"],
    "Property": ["BSDE.JK", "CTRA.JK", "SMRA.JK", "PANI.JK", "LPKR.JK"],
    "Infrastructure": ["TBIG.JK", "TOWR.JK", "JSMR.JK", "WIKA.JK", "PTPP.JK"],
    "Cement": ["SMGR.JK", "INTP.JK", "SMCB.JK"],
}

# Popular/Blue chip stocks for quick access
POPULAR_STOCKS: List[str] = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK",
    "BBNI.JK", "ICBP.JK", "INDF.JK", "KLBF.JK", "ADRO.JK",
    "ANTM.JK", "GOTO.JK", "PGAS.JK", "PTBA.JK", "SMGR.JK"
]

# ============================================
# Utility Functions
# ============================================

def get_stock_name(ticker: str) -> str:
    """Get display name for a stock ticker."""
    return STOCK_NAMES.get(ticker, ticker.replace(".JK", ""))

def get_sector(ticker: str) -> str:
    """Get sector for a stock ticker."""
    for sector, tickers in SECTORS.items():
        if ticker in tickers:
            return sector
    return "Other"

def is_valid_ticker(ticker: str) -> bool:
    """Check if ticker is in our universe."""
    return ticker in STOCK_UNIVERSE
