import React, { createContext, useContext, useState, ReactNode } from 'react';

// Types for selection (e.g., file, error, etc.)
export type Selection = {
  type: 'file' | 'error' | 'timeline' | null;
  payload?: any;
};

interface SelectionContextType {
  selection: Selection;
  setSelection: (sel: Selection) => void;
}

const SelectionContext = createContext<SelectionContextType | undefined>(undefined);

export const useSelection = () => {
  const ctx = useContext(SelectionContext);
  if (!ctx) throw new Error('useSelection must be used within a SelectionProvider');
  return ctx;
};

export const SelectionProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [selection, setSelection] = useState<Selection>({ type: null });
  return (
    <SelectionContext.Provider value={{ selection, setSelection }}>
      {children}
    </SelectionContext.Provider>
  );
};
