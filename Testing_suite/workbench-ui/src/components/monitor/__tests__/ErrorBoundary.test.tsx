import { render, screen } from '@testing-library/react';
import React from 'react';
import { ErrorBoundary } from '../ErrorBoundary';

describe('ErrorBoundary', () => {
  it('renders children when no error', () => {
    render(
      <ErrorBoundary>
        <span>Safe Child</span>
      </ErrorBoundary>
    );
    expect(screen.getByText(/safe child/i)).toBeInTheDocument();
  });

  it('renders error UI on error', () => {
    // Component that throws
    const ProblemChild = () => { throw new Error('Test error!'); };
    render(
      <ErrorBoundary>
        <ProblemChild />
      </ErrorBoundary>
    );
    expect(screen.getByText(/component error/i)).toBeInTheDocument();
    expect(screen.getByText(/test error/i)).toBeInTheDocument();
  });
});
