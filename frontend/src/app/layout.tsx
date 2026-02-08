import type { Metadata } from "next";
import "./globals.css";
import QueryProvider from "../providers/QueryProvider";

export const metadata: Metadata = {
  title: "StockPredict.AI | Hybrid SARIMA-LSTM Predictive Analytics",
  description: "Advanced Stock Market Predictive Analytics System using hybrid SARIMA and LSTM models. Analyze 275 Indonesian stocks with real-time technical indicators, multi-horizon forecasts, and sentiment analysis.",
  keywords: "stock prediction, SARIMA, LSTM, deep learning, technical analysis, IDX, Indonesia Stock Exchange, AI trading, machine learning, stock forecast",
  authors: [{ name: "StockPredict.AI" }],
  robots: "index, follow",
  openGraph: {
    title: "StockPredict.AI | AI-Powered Stock Prediction",
    description: "Hybrid SARIMA-LSTM model for accurate stock price forecasting. Analyze 275 Indonesian stocks with comprehensive technical analysis.",
    type: "website",
    locale: "id_ID",
  },
  icons: {
    icon: '/logo-app.jpeg',
    apple: '/logo-app.jpeg',
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="id" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <meta name="theme-color" content="#0a0a0f" />
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=5" />
      </head>
      <body className="antialiased bg-[#0a0a0f] text-neutral-200">
        <QueryProvider>
          {children}
        </QueryProvider>
      </body>
    </html>
  );
}
