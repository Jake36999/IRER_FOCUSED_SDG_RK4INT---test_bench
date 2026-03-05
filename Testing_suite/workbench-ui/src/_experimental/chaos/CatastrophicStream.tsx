import React, { useEffect, useRef, useState } from 'react';

// Mock stream of rapid errors
const MOCK_ERRORS = [
  "[FATAL] ConcretizationTypeError: Abstract tracer value encountered in jax.lax.scan",
  "[WARN] NaN detected in gradient pass 4042",
  "[FATAL] rho_max exceeded physical bounds (rho > 1.0e12)",
  "[SYS] Memory slab allocation failed on GPU 0",
  "[FATAL] Vacuum collapse geometry detected at coordinates [12, 44, 91]",
];

export const CatastrophicStream: React.FC = () => {
  const [logs, setLogs] = useState<string[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Simulate high-frequency incoming error logs
  useEffect(() => {
    const interval = setInterval(() => {
      const randomError = MOCK_ERRORS[Math.floor(Math.random() * MOCK_ERRORS.length)];
      const timestamp = new Date().toISOString().split('T')[1].replace('Z', '');
      
      setLogs(prev => [...prev.slice(-100), `[${timestamp}] ${randomError}`]); // Keep last 100
    }, 150); // Very fast updates

    return () => clearInterval(interval);
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="h-full w-full flex flex-col relative group">
      {/* Stream Header */}
      <div className="bg-red-950/40 px-4 py-2 border-b border-red-900/50 flex justify-between items-center">
        <span className="text-xs font-bold text-red-500 tracking-wider">RAW EXCEPTION FEED</span>
        <span className="text-[10px] text-red-400 bg-red-900/30 px-2 py-0.5 rounded border border-red-800 animate-pulse">
          LIVE
        </span>
      </div>

      {/* The Log Terminal */}
      <div 
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-4 font-mono text-[11px] leading-relaxed custom-scrollbar relative z-10"
      >
        {logs.map((log, i) => (
          <div key={i} className="text-red-400/80 hover:bg-red-900/20 hover:text-red-300 px-1 -mx-1 rounded transition-colors break-words">
            {log}
          </div>
        ))}
      </div>

      {/* CRT Scanline Effect (Overlay) */}
      <div className="absolute inset-0 pointer-events-none z-20 bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] bg-[length:100%_4px,3px_100%] opacity-20" />
    </div>
  );
};