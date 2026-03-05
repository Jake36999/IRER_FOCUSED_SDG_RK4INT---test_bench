import React, { useState, useEffect, useCallback } from 'react';
import { TelemetryChart } from './TelemetryChart';
import { useTelemetry } from '@/lib/hooks/useTelemetry'; // Assumed hook
import { useKelSummary } from '@/lib/hooks/useKelSummary'; // Assumed hook
import { SignalLayer } from './SignalLayer';
import { useScanReport } from '@/lib/hooks/useScanReport';


import { ErrorBoundary } from './ErrorBoundary';

export const MonitorContainer: React.FC = () => {
  // 1. Hydrate static scan report before connecting SSE
  const { fileTree, loading: reportLoading, error: reportError } = useScanReport();
  const [isHydrated, setIsHydrated] = useState(false);
  const [sseReady, setSseReady] = useState(false);
  const { events, latest, status: connectionStatus } = useTelemetry(sseReady);
  const { summary: kelSummary } = useKelSummary();
  const [isCritical, setIsCritical] = useState(false);

  useEffect(() => {
    if (!reportLoading && !reportError) {
      setIsHydrated(true);
    }
  }, [reportLoading, reportError]);

  useEffect(() => {
    if (isHydrated) {
      setSseReady(true);
    }
  }, [isHydrated]);

  useEffect(() => {
    if (latest?.rho_max && latest.rho_max > 0.95) {
      setIsCritical(true);
    } else {
      setIsCritical(false);
    }
  }, [latest]);

  const handleChartClick = useCallback((timestamp: number) => {
    console.log(`[Auditor] Deep dive requested at t=${timestamp}`);
    // Trigger Telescope Modal logic here
  }, []);

  if (reportLoading) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-slate-900">
        <span className="text-amber-500 font-mono animate-pulse">
          [System] Hydrating state from KEL...
        </span>
      </div>
    );
  }

  if (reportError) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-slate-900">
        <span className="text-red-500 font-mono">[Error] {reportError}</span>
      </div>
    );
  }

  return (
    <ErrorBoundary>
      <div className="relative w-full h-full bg-slate-900 overflow-hidden flex flex-col">
        {/* Overlay Layer for Alarms/Ghost Tracer */}
        <SignalLayer 
          isCritical={isCritical} 
          activePrimitives={latest?.active_primitives || []} 
        />

        {/* Pure Chart Rendering */}
        <div className="flex-grow relative z-10">
          <ErrorBoundary>
            <TelemetryChart 
              data={events.map(e => ({
                timestamp: Date.parse(e.timestamp),
                h_norm: e.h_norm_l2,
                rho_max: e.rho_max ?? 0,
                step: e.step,
              }))}
              width={800}
              height={600}
              onDataPointClick={handleChartClick}
              colorMode={isCritical ? 'CRITICAL' : 'NORMAL'}
            />
          </ErrorBoundary>
        </div>

        {/* Status Footer (ISA-101 Gray-Scale) */}
        <div className="h-8 bg-slate-800 border-t border-slate-700 flex items-center px-4 space-x-4">
          <span className={`text-xs font-mono ${connectionStatus === 'CONNECTED' ? 'text-slate-400' : 'text-amber-500'}`}>
            SOCKET: {connectionStatus}
          </span>
          <span className="text-xs font-mono text-slate-500">
             KEL INDEX: {kelSummary?.total_vectors || 0}
          </span>
        </div>
      </div>
    </ErrorBoundary>
  );
};