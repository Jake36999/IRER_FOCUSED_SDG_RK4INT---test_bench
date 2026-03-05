import { render, screen } from '@testing-library/react';
import { ForensicsDock } from '../ForensicsDock';

describe('ForensicsDock', () => {
  it('renders SessionTimeLine and AxiomInspector', () => {
    render(<ForensicsDock sessionEvents={['event1']} axiomStates={[{ id: 'a1', label: 'Axiom', description: 'desc' }]} />);
    expect(screen.getByText(/Axiom Inspector/i)).toBeInTheDocument();
    expect(screen.getByText(/SessionTimeLine/i)).toBeInTheDocument(); // If SessionTimeLine renders a label
  });
});
