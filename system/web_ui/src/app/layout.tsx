import type { Metadata } from "next";
import { Montserrat, Cormorant_Garamond } from "next/font/google";
import Providers from "./providers";
import "./globals.css";

const montserrat = Montserrat({
  subsets: ["latin"],
  variable: "--font-body",
});
const cormorant = Cormorant_Garamond({
  subsets: ["latin"],
  variable: "--font-display",
  weight: ["400", "600", "700"],
});

export const metadata: Metadata = {
  title: "UNLZ AI Studio",
  description: "Next Gen AI Platform",
  icons: {
    icon: "/favicon.ico",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`app-body ${montserrat.variable} ${cormorant.variable}`}>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
