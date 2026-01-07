// file-utils.ts
export interface ValidateXrayResponse {
  is_xray: boolean;
  score: number;
  top_label?: string;
  all_probs?: Record<string, number>;
  error?: string;
}

export function isDicomFile(file: File): boolean {
  const name = file.name.toLowerCase();
  const t = (file.type || '').toLowerCase();
  return (
    name.endsWith('.dcm') ||
    name.endsWith('.dicom') ||
    t === 'application/dicom' ||
    t === 'application/dicom+json' ||
    t === 'application/dicom+xml'
  );
}

// Convierte ArrayBuffer -> base64 (en chunks para evitar overflow)
export function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000; // 32KB
  let binary = '';
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...Array.from(chunk));
  }
  return btoa(binary);
}

// Lee un File como ArrayBuffer (para DICOM)
export function readAsArrayBuffer(file: File): Promise<ArrayBuffer> {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result as ArrayBuffer);
    fr.onerror = () => reject(fr.error);
    fr.readAsArrayBuffer(file);
  });
}

// Lee un File como DataURL (para imágenes)
export function readAsDataURL(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result as string);
    fr.onerror = () => reject(fr.error);
    fr.readAsDataURL(file);
  });
}
