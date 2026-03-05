describe('UnifiedLensExplorer', () => {
import { render, screen } from '@testing-library/react';
import React from 'react';
jest.mock('../../lib/context/SelectionContext', () => ({ useSelection: () => ({ setSelection: jest.fn() }) }));
import { UnifiedLensExplorer } from '../UnifiedLensExplorer';

describe('UnifiedLensExplorer', () => {
  it('renders friction cards from props', () => {
    const frictionCards = [
      { id: 'err1', label: 'Error 1', message: 'Test error', timestamp: 'now' }
    ];
    render(<UnifiedLensExplorer frictionCards={frictionCards} />);
    expect(screen.getByText(/Error 1/i)).toBeInTheDocument();
    expect(screen.getByText(/Test error/i)).toBeInTheDocument();
  });

  it('renders empty state gracefully', () => {
    render(<UnifiedLensExplorer frictionCards={[]} />);
    expect(screen.getByText(/Friction/i)).toBeInTheDocument();
  });
});
