import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { map, Observable } from 'rxjs';

/**
 * ValidateImageService
 *
 * 🇪🇸 Servicio encargado de verificar si la imagen enviada por el usuario corresponde
 * a una radiografía de tórax válida, utilizando un modelo de validación (e.g. CLIP).
 *
 * 🇺🇸 Service responsible for verifying whether the submitted image is a valid chest X-ray,
 * using a validation model (e.g., CLIP).
 */
@Injectable({
  providedIn: 'root',
})
export class ValidateImageService {
  // URL del endpoint de validación por modelo CLIP
  private readonly apiUrl = 'http://localhost:5002/validate-xray';
  private readonly previewUrl = 'http://localhost:5002/preview-image';

  constructor( private http: HttpClient) {}

  /**
   * validateImage
   *
   * 🇪🇸 Valida si una imagen codificada en base64 es una radiografía de tórax,
   * enviándola a un backend que utiliza un modelo CLIP.
   *
   * 🇺🇸 Validates whether a base64-encoded image is a chest X-ray,
   * by sending it to a backend using a CLIP model.
   *
   * @param base64Image - 🇪🇸 Imagen en base64 a validar | 🇺🇸 Base64 image to validate
   * @returns Promise<boolean> - 🇪🇸 `true` si es una radiografía válida | 🇺🇸 `true` if it is a valid X-ray
   * @throws Error - En caso de error en la conexión o respuesta del backend
   */
  async validateImage(base64Image: string): Promise<boolean> {
    try {
      const response = await fetch(this.apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64Image }),
      });

      if (!response.ok) {
        const error = await response.text();
        console.error('Error en respuesta CLIP:', error);
        throw new Error('Respuesta no válida del validador CLIP');
      }

      const result = await response.json();
      console.log('Respuesta CLIP:', result);

      return result?.is_xray === true;
    } catch (err) {
      console.error('Error en validación CLIP:', err);
      throw err;
    }
  }

  /**
   * Genera una vista previa PNG (dataURL) desde el backend.
   * Sirve para DICOM (y también funciona con imágenes normales).
   *
   * @param base64Image base64 CRUDO (sin data URL)
   * @returns Observable<string> con un dataURL: "data:image/png;base64,..."
   */
  getPreview(base64Image: string): Observable<string> {
    const headers = new HttpHeaders({ 'Content-Type': 'application/json' });
    return this.http
      .post<{ data_url: string }>(this.previewUrl, { image: base64Image }, { headers })
      .pipe(map((res) => res.data_url));
  }

}
