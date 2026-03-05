import React from 'react';

interface DiagnosticBarProps {
  jaxVersion: string;
  vectorsLoaded: number;
  unindexedCount: number;
  onSearch: (query: string) => void;
  similarityThreshold: number;
}

export const DiagnosticContextBar: React.FC<DiagnosticBarProps> = ({
  jaxVersion,
  vectorsLoaded,
  unindexedCount,
  onSearch,
  similarityThreshold
}) => {
  return (
    <header className="h-12 bg-slate-900/95 backdrop-blur border-b border-slate-800 flex items-center justify-between px-4 sticky top-0 z-50">
      
      {/* Left: System Vitals */}
      <div className="flex items-center space-x-6 font-mono text-xs">
        <div className="flex items-center space-x-2">
          <span className="text-slate-500">JAX_VER:</span>
          {/* ISA-101: Green if matching target, Amber if mismatched */}
          <span className={jaxVersion === '0.4.20' ? 'text-green-500' : 'text-amber-500'}>
            {jaxVersion}
          </span>
        </div>

        <div className="flex items-center space-x-2">
          <span className="text-slate-500">VECTORS:</span>
          <div className="w-24 h-2 bg-slate-800 rounded-full overflow-hidden">
            <div 
              className="h-full bg-blue-600" 
              style={{ width: `${Math.min(vectorsLoaded / 1000 * 100, 100)}%` }} 
            />
          </div>
          <span className="text-slate-300">{vectorsLoaded}</span>
        </div>

        {unindexedCount > 0 && (
          <div className="flex items-center space-x-2 px-2 py-1 bg-red-900/20 border border-red-900 rounded">
            <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
            <span className="text-red-400">UNINDEXED: {unindexedCount}</span>
          </div>
        )}
      </div>

      {/* Center: Search Console */}
      <div className="flex-1 max-w-xl mx-4">
        <div className="relative">
          <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-600 font-mono">
            $ query
          </span>
          <input 
            type="text" 
            placeholder="Search Remedial Manifold..."
            className="w-full bg-black/50 border border-slate-700 rounded-md py-1.5 pl-20 pr-4 text-xs font-mono text-slate-200 focus:outline-none focus:border-blue-500 transition-colors"
            onChange={(e) => onSearch(e.target.value)}
          />
        </div>
      </div>

      {/* Right: Sensitivity Control */}
      <div className="flex items-center space-x-3 w-48">
        <span className="text-[10px] text-slate-500 font-mono uppercase">Similarity Thresh.</span>
        <input 
          type="range" 
          min="0.4" 
          max="0.99" 
          step="0.01" 
          defaultValue={similarityThreshold}
          className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
        />
        <span className="text-xs font-mono text-blue-400">{similarityThreshold}</span>
      </div>
    </header>
  );
};