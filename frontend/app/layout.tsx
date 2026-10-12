import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";
import AppHeader from "@/components/AppHeader";
import "./globals.css";

const plexSans = IBM_Plex_Sans({ subsets: ["latin"], weight: ["300", "400", "500", "600"] });
const plexMono = IBM_Plex_Mono({ subsets: ["latin"], weight: ["400", "500"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: { default: "Autotune", template: "%s — Autotune" },
  description: "Train, benchmark, and serve models on your own data: upload, fine-tune, compare, and deploy.",
  applicationName: "Autotune",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${plexSans.className} ${plexMono.variable} flex min-h-screen flex-col`}>
        <AppHeader />
        <main className="app-main">{children}</main>
      </body>
    </html>
  );
}
