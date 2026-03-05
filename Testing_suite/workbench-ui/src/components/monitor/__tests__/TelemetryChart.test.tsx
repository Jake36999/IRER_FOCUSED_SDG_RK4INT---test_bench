import { render, screen } from '@testing-library/react';
import { TelemetryChart } from '../TelemetryChart';

describe('TelemetryChart', () => {
  const data = [
    { timestamp: 1, h_norm: 0.1, rho_max: 0.2, step: 1 },
    { timestamp: 2, h_norm: 0.2, rho_max: 0.3, step: 2 }
  ];

  it('renders chart with data', () => {
    render(<TelemetryChart data={data} />);
    expect(screen.getByText(/step/i)).toBeInTheDocument();
  });

  it('shows loading state', () => {
    render(<TelemetryChart data={[]} loading={true} />);
    expect(screen.getByText(/acquiring stream/i)).toBeInTheDocument();
  });
});
