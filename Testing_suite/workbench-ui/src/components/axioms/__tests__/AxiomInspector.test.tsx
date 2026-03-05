import { render, screen } from '@testing-library/react';
import { AxiomInspector } from '../AxiomInspector';

describe('AxiomInspector', () => {
  it('renders provided axioms', () => {
    const axioms = [
      { id: 'a1', label: 'Test Axiom', description: 'Axiom description', verified: 'test' }
    ];
    render(<AxiomInspector axioms={axioms} />);
    expect(screen.getByText(/Test Axiom/i)).toBeInTheDocument();
    expect(screen.getByText(/Axiom description/i)).toBeInTheDocument();
    expect(screen.getByText(/Verified: test/i)).toBeInTheDocument();
  });

  it('renders empty state gracefully', () => {
    render(<AxiomInspector axioms={[]} />);
    expect(screen.getByText(/Axiom Inspector/i)).toBeInTheDocument();
  });
});
