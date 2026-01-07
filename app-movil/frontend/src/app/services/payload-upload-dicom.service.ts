// upload.service.ts
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, from } from 'rxjs';
import { switchMap } from 'rxjs/operators';
import {
  arrayBufferToBase64,
  isDicomFile,
  readAsArrayBuffer,
  readAsDataURL,
  ValidateXrayResponse,
} from '../shared/resources/files-utils';

@Injectable({ providedIn: 'root' })
export class UploadService {
  
  private readonly API_URL = 'http://localhost:5002/validate-xray';

  constructor(private http: HttpClient) {}

  /**
   * Prepara el payload { image: <base64> } a partir de un File.
   * - DICOM => lee como ArrayBuffer, convierte a base64 CRUDO (sin dataURL)
   * - Imagen => lee como DataURL y extrae el payload (después de la coma)
   */
  buildPayload(file: File): Promise<{ image: string; previewDataUrl?: string; isDicom: boolean }> {
    if (isDicomFile(file)) {
      return readAsArrayBuffer(file).then(ab => ({
        image: arrayBufferToBase64(ab),
        isDicom: true,
        previewDataUrl: undefined, // no hay preview para DICOM
      }));
    } else if ((file.type || '').startsWith('image/')) {
      return readAsDataURL(file).then(dataUrl => {
        const base64 = dataUrl.split(',')[1] || '';
        return { image: base64, isDicom: false, previewDataUrl: dataUrl };
      });
    } else {
      return Promise.reject(new Error('Formato no soportado. Sube una imagen o un DICOM (.dcm).'));
    }
  }

  /**
   * Envía el payload al endpoint /validate-xray
   */
  validateXray(payload: { image: string }): Observable<ValidateXrayResponse> {
    return this.http.post<ValidateXrayResponse>(this.API_URL, payload);
  }

  /**
   * Flujo completo: de File -> request -> response
   */
  analyzeFile(file: File): Observable<ValidateXrayResponse> {
    return from(this.buildPayload(file)).pipe(
      switchMap(({ image }) => this.validateXray({ image }))
    );
  }
}
