import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

/**
 * ImageService
 *
 * 🇪🇸 Servicio encargado de manejar la imagen seleccionada o capturada por el usuario
 * y enviarla al backend para su análisis mediante un modelo de IA.
 *
 * 🇺🇸 Service responsible for handling the user-selected or captured image
 * and sending it to the backend for analysis using an AI model.
 */
@Injectable({
  providedIn: 'root'
})
export class ImageService {
  // URL del backend para el análisis de imágenes
  // Backend URL for image analysis
  private apiUrl = 'http://localhost:10000/analyze-image';

  private imageData: string | null = null;

  constructor(private http: HttpClient) {}

  /**
   * sendImage
   *
   * 🇪🇸 Envía la imagen en base64 al backend para su análisis.
   * 🇺🇸 Sends the base64 image to the backend for analysis.
   *
   * @param base64Image - 🇪🇸 Imagen codificada en base64 | 🇺🇸 Base64-encoded image
   * @returns Observable con la respuesta del backend | Observable with backend response
   */
  sendImage(base64Image: string): Observable<any> {
    const headers = new HttpHeaders({ 'Content-Type': 'application/json' });
    const body = JSON.stringify({ image: base64Image });
    return this.http.post(this.apiUrl, body, { headers });
  }

  /**
   * setImage
   *
   * 🇪🇸 Guarda temporalmente la imagen seleccionada o capturada.
   * 🇺🇸 Temporarily stores the selected or captured image.
   *
   * @param data - Imagen en formato base64 | Base64 image
   */
  setImage(data: string) {
    this.imageData = data;
  }

  /**
   * getImage
   *
   * 🇪🇸 Obtiene la imagen almacenada.
   * 🇺🇸 Retrieves the stored image.
   *
   * @returns string | null
   */
  getImage(): string | null {
    return this.imageData;
  }

  /**
   * clearImage
   *
   * 🇪🇸 Elimina la imagen almacenada del servicio.
   * 🇺🇸 Clears the stored image from the service.
   */
  clearImage() {
    this.imageData = null;
  }
}
