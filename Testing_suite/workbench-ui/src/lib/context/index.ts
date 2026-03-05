import React, { createContext, useContext, useState, ReactNode } from 'react';

// Selection Context
export interface SelectionState {
  selectedFile: string | null;
  selectedError: string | null;
}

const SelectionContext = createContext<{
  selection: SelectionState;
  setSelection: (s: SelectionState) => void;
} | undefined>(undefined);

export const SelectionProvider = ({ children }: { children: ReactNode }) => {
  const [selection, setSelection] = useState<SelectionState>({ selectedFile: null, selectedError: null });
  return (
    <SelectionContext.Provider value={{ selection, setSelection }}>
      {children}
    </SelectionContext.Provider>
  );
};

export function useSelection() {
  const ctx = useContext(SelectionContext);
  if (!ctx) throw new Error('useSelection must be used within SelectionProvider');
  return ctx;
}

// Timeline Context
export interface TimelineState {
  currentStep: number;
  setStep: (step: number) => void;
}

const TimelineContext = createContext<TimelineState | undefined>(undefined);

export const TimelineProvider = ({ children }: { children: ReactNode }) => {
  const [currentStep, setStep] = useState(0);
  return (
    <TimelineContext.Provider value={{ currentStep, setStep }}>
      {children}
    </TimelineContext.Provider>
  );
};

export function useTimeline() {
  const ctx = useContext(TimelineContext);
  if (!ctx) throw new Error('useTimeline must be used within TimelineProvider');
  return ctx;
}

// Fix State Context
export interface FixState {
  fixStatus: 'idle' | 'pending' | 'applied' | 'error';
  setFixStatus: (s: 'idle' | 'pending' | 'applied' | 'error') => void;
}

const FixStateContext = createContext<FixState | undefined>(undefined);

export const FixStateProvider = ({ children }: { children: ReactNode }) => {
  const [fixStatus, setFixStatus] = useState<'idle' | 'pending' | 'applied' | 'error'>('idle');
  return (
    <FixStateContext.Provider value={{ fixStatus, setFixStatus }}>
      {children}
    </FixStateContext.Provider>
  );
};

export function useFixState() {
  const ctx = useContext(FixStateContext);
  if (!ctx) throw new Error('useFixState must be used within FixStateProvider');
  return ctx;
}
