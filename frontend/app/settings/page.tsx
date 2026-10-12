"use client";

import { ErrorNotice, PageHero, Panel } from "@/components/ui";
import { API_BASE } from "@/lib/api";
import type { CredentialStatus } from "@/lib/types";
import { usePolling } from "@/lib/usePolling";

export default function SettingsPage() {
  const credentials = usePolling<CredentialStatus[]>("/api/settings/credentials", 15000, () => true);

  return (
    <>
      <PageHero eyebrow="05 / Settings" title={<>Keys and <em>connections</em>.</>}>
        Credentials live in the API&apos;s .env file. This page shows whether each one is set. It never shows a value.
      </PageHero>
      <ErrorNotice message={credentials.error} title="API unavailable" />

      <div className="grid-2">
        <Panel index="A" title="Credentials">
          {credentials.data ? (
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th>Variable</th><th>Needed for</th><th>Status</th></tr></thead>
                <tbody>
                  {credentials.data.map((credential) => (
                    <tr key={credential.name}>
                      <td className="mono">{credential.name}</td>
                      <td>{credential.purpose}</td>
                      <td>
                        <span className={`status status--${credential.configured ? "ok" : "warn"}`}>
                          <i aria-hidden="true" />
                          {credential.configured ? "configured" : "not configured"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
          <div className="panel__body">
            <div className="notice notice--info">
              <strong>How to set one</strong>
              Put <code>NAME=value</code> in <code>{credentials.data?.[0]?.env_file ?? ".env"}</code> at the Autotune repo root, then restart the API. A value left at <code>mock_key</code> counts as not configured.
            </div>
          </div>
        </Panel>

        <Panel index="B" title="Connection">
          <div className="facts">
            <div><span>API</span><strong>{API_BASE}</strong></div>
            <div><span>Set with</span><strong>NEXT_PUBLIC_API_URL</strong></div>
          </div>
          <div className="panel__body">
            <p className="text-sm text-muted-foreground">
              The API only accepts requests from this UI&apos;s origin. If you serve the UI elsewhere, set <code className="mono">AUTOTUNE_UI_ORIGIN</code> for the API to match.
            </p>
          </div>
        </Panel>
      </div>
    </>
  );
}
