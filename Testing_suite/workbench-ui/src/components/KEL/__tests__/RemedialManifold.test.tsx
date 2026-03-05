import { render, screen } from '@testing-library/react';
import React from 'react';
jest.mock('../../lib/context', () => ({ useSelection: () => ({ selection: {} }) }));
jest.mock('../../lib/hooks/useRemedialQuery', () => ({ useRemedialQuery: () => ({ result: null, loading: false, error: null, queryKel: jest.fn() }) }));
import RemedialManifold from '../RemedialManifold';

describe('RemedialManifold', () => {
  it('renders prompt when no error selected', () => {
    render(<RemedialManifold />);
    expect(screen.getByText(/select an error/i)).toBeInTheDocument();
  });
});
