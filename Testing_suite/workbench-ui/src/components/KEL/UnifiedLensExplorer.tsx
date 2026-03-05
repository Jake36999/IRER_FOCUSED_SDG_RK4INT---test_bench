import React, { useEffect, useRef } from "react";
import { ErrorBoundary } from '../monitor/ErrorBoundary';
import SelectionArea from '@viselect/vanilla';
import { useSelection } from '../../lib/context/SelectionContext';
import { Tabs, TabsList, TabsTrigger, TabsContent } from "../lightswind/tabs";
import { Activity, FolderTree, Zap } from "lucide-react";

interface FrictionCard {
  id: string;
  label: string;
  message: string;
  timestamp: string;
}

interface UnifiedLensExplorerProps {
  frictionCards?: FrictionCard[];
}

export const UnifiedLensExplorer: React.FC<UnifiedLensExplorerProps> = ({ frictionCards = [] }) => {
  const { setSelection } = useSelection();
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const selection = new SelectionArea({
      selectables: ['.anomaly-card'],
      boundaries: [containerRef.current],
    });
    return () => selection.destroy();
  }, []);

  return (
    <ErrorBoundary>
      <div ref={containerRef} className="h-full w-full flex flex-col bg-[#0a0a0a] border-r border-slate-800">
        <Tabs defaultValue="friction" className="flex flex-col h-full w-full">
          {/* Tab Header */}
          <div className="px-4 py-3 border-b border-slate-800 bg-[#0a0a0a]/50 backdrop-blur-sm">
            <TabsList className="grid w-full grid-cols-3 bg-slate-900/50 p-1 h-9">
              <TabsTrigger value="friction" className="data-[state=active]:bg-slate-800 data-[state=active]:text-white">
                <Zap className="w-3.5 h-3.5 mr-2" />
                Friction
              </TabsTrigger>
              <TabsTrigger value="filesystem" className="data-[state=active]:bg-slate-800 data-[state=active]:text-white">
                <FolderTree className="w-3.5 h-3.5 mr-2" />
                Files
              </TabsTrigger>
              <TabsTrigger value="telemetry" className="data-[state=active]:bg-slate-800 data-[state=active]:text-white">
                <Activity className="w-3.5 h-3.5 mr-2" />
                Signals
              </TabsTrigger>
            </TabsList>
          </div>

          {/* Tab Contents (The Stream) */}
          <div className="flex-1 overflow-y-auto custom-scrollbar p-2">
            <TabsContent value="friction" className="mt-0 h-full">
              <div className="space-y-2">
                {frictionCards.map(card => (
                  <div
                    key={card.id}
                    className="anomaly-card p-3 rounded border border-slate-800 bg-slate-900/30 hover:bg-slate-800/50 transition-colors cursor-pointer group"
                    onClick={() => setSelection({ type: 'error', payload: { id: card.id } })}
                  >
                    <div className="flex justify-between items-start mb-1">
                      <span className="text-xs font-mono text-red-400">{card.label}</span>
                      <span className="text-[10px] text-slate-500">{card.timestamp}</span>
                    </div>
                    <p className="text-xs text-slate-300 line-clamp-2">
                      {card.message}
                    </p>
                  </div>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="filesystem" className="mt-0">
              <div className="p-4 text-center text-xs text-slate-500">
                Filesystem Tree Visualization Placeholder
              </div>
            </TabsContent>

            <TabsContent value="telemetry" className="mt-0">
               <div className="p-4 text-center text-xs text-slate-500">
                Active Stream Signals Placeholder
              </div>
            </TabsContent>

          </div>
        </Tabs>
      </div>
    </ErrorBoundary>
  );
};