import React from 'react';
import { SmokeyBackground } from '../lightswind/smokey-background'; // Assumes component from lightswind
import { ParticlesBackground } from '../lightswind/particles-background';

interface EntropyCanvasProps {
  intensity: number; // 0.0 to 1.0 mapping to h_norm spike
}

export const EntropyCanvas: React.FC<EntropyCanvasProps> = ({ intensity }) => {
  return (
    <div className="absolute inset-0 w-full h-full">
      {/* Base Layer: Smokey/Hellish WebGL Shader */}
      <SmokeyBackground 
        color="#ef4444" // ISA-101 Critical Red
        density={intensity * 10} 
        speed={intensity * 5}
        className="absolute inset-0"
      />
      
      {/* Overlay Layer: Fast moving particles representing escaping tensors/memory leaks */}
      <ParticlesBackground 
        particleColor="#fca5a5"
        particleCount={Math.floor(intensity * 500)}
        speed={intensity * 10}
        direction="up" // Simulating explosion/evaporation
        className="absolute inset-0 opacity-50"
      />
    </div>
  );
};