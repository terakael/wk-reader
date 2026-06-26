export function l2normalize(arr) {
  let norm = 0;
  for (const v of arr) norm += v * v;
  norm = Math.sqrt(norm);
  if (norm === 0) return arr instanceof Float32Array ? arr : new Float32Array(arr);
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) out[i] = arr[i] / norm;
  return out;
}
