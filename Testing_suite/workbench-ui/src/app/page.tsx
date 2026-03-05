import BinaryGuardExplorer from '../components/binary-guard/BinaryGuardExplorer';
import MonitorDashboard from '../components/monitor/MonitorDashboard';

export default function HomePage() {
  return (
    <div className="space-y-8">
      <h1 className="text-2xl font-bold">IRER Test Bench: God View</h1>
      <MonitorDashboard />
      <BinaryGuardExplorer />
    </div>
  );
}
