// Shannon entropy calculation for a string
export function calculateEntropy(str: string): number {
  const map: Record<string, number> = {};
  for (const char of str) {
    map[char] = (map[char] || 0) + 1;
  }
  const len = str.length;
  let entropy = 0;
  for (const k in map) {
    const p = map[k] / len;
    entropy -= p * Math.log2(p);
  }
  return entropy;
}
