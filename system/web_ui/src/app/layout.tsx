import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
    title: "UNLZ AI Studio",
    description: "Next Gen AI Platform",
};

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en">
            <body className="antialiased min-h-screen bg-black text-white selection:bg-primary selection:text-white">
                {children}
            </body>
        </html>
    );
}
