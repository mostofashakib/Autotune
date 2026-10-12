"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Cloud, FlaskConical, Gauge, Server, Settings } from "lucide-react";
import AutotuneMark from "@/components/AutotuneMark";

const NAV_ITEMS = [
  { label: "Training", href: "/training", icon: Gauge },
  { label: "Benchmark", href: "/benchmark", icon: FlaskConical },
  { label: "Serving", href: "/serving", icon: Server },
  { label: "Cloud", href: "/cloud", icon: Cloud },
  { label: "Settings", href: "/settings", icon: Settings },
];

function isActivePath(pathname: string, href: string): boolean {
  return pathname === href || pathname.startsWith(`${href}/`);
}

export default function AppHeader() {
  const pathname = usePathname();

  return (
    <header className="app-header">
      <div className="app-header__rail">
        <Link href="/" className="brand-mark" aria-label="Autotune home">
          <span className="brand-mark__icon">
            <AutotuneMark />
          </span>
          <span className="brand-mark__word">AUTOTUNE</span>
        </Link>
        <nav className="app-nav" aria-label="Primary navigation">
          {NAV_ITEMS.map((item, index) => {
            const active = isActivePath(pathname, item.href);
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                aria-current={active ? "page" : undefined}
                className={`app-nav__link ${active ? "app-nav__link--active" : ""}`}
              >
                <span className="app-nav__index">0{index + 1}</span>
                <Icon size={15} strokeWidth={1.8} aria-hidden="true" />
                <span className="app-nav__label">{item.label}</span>
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
