import React from 'react';
import { SessionTimeLine } from './SessionTimeLine';
import { AxiomInspector } from '../axioms/AxiomInspector';

interface ForensicsDockProps {
  sessionEvents: any[];
  axiomStates: any[];
}

export const ForensicsDock: React.FC<ForensicsDockProps> = ({ sessionEvents, axiomStates }) => {
  return (
    <div className="flex flex-row w-full h-64 border-t border-slate-800 bg-slate-950/90 z-20">
      {/* Timeline takes up the majority of the dock */}
      <div className="flex-grow border-r border-slate-800 overflow-hidden relative">
        <SessionTimeLine events={sessionEvents} />
      </div>
      {/* Inspector stays pinned to the right side of the dock */}
      <div className="w-1/3 min-w-[300px] overflow-y-auto">
        <AxiomInspector axioms={axiomStates} />
      </div>
    </div>
  );
};
