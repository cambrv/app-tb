import { Injectable } from '@angular/core';

export interface DiagnosisContext {
  probability: number;        // 0..1 (ej. 0.82 == 82%)
  imageUrl?: string;          // opcional: preview o referencia
  patientId?: string;         // opcional
  timestamp?: string;         // ISO string opcional
}

/**
 * ChatContextService
 *
 * 🇪🇸 Servicio global utilizado para compartir el contexto del diagnóstico
 * entre el componente que analiza la imagen y el módulo de chat de recomendaciones.
 *
 * 🇺🇸 Global service used to share diagnostic context
 * between the image analysis component and the recommendation chat module.
 *
 * Proporciona / Provides:
 * - 🇪🇸 Almacenamiento temporal de probabilidad de tuberculosis e imagen analizada.
 * - 🇺🇸 Temporary storage of TB probability and preview image.
 *
 * Facilita / Facilitates:
 * - 🇪🇸 Comunicación entre componentes sin necesidad de @Input ni rutas.
 * - 🇺🇸 Component communication without relying on @Input or routing.
 */
@Injectable({
  providedIn: 'root',
})
export class ChatContextService {

    private ctx?: DiagnosisContext;

  /** Setea el contexto de diagnóstico cuando llegas al modal de chat */
  setDiagnosisContext(ctx: DiagnosisContext) {
    this.ctx = { ...ctx };
  }

  /** Devuelve el contexto actual; si no existe, genera uno "vacío" seguro */
  getDiagnosisContext(): DiagnosisContext {
    return this.ctx ?? { probability: 0 };
  }

  /**
   * Construye el mensaje inicial del asistente según la probabilidad.
   * Ajusta los umbrales a tu criterio clínico (ej.: <0.3 bajo, 0.3–0.7 medio, >0.7 alto).
   */
  buildIntroMessage(): string {
    const { probability } = this.getDiagnosisContext();
    const p = Math.max(0, Math.min(1, probability));
    const pct = Math.round(p * 100);

    if (p < 0.3) {
      return [
        `**Análisis inicial:** Probabilidad estimada de tuberculosis: **${pct}%** (bajo).`,
        '',
        `Puedo darte recomendaciones preventivas y orientarte sobre signos de alarma.`,
        `Nota: Esto **no sustituye** una evaluación médica.`,
      ].join('\n');
    }

    if (p < 0.7) {
      return [
        `**Análisis inicial:** Probabilidad estimada de tuberculosis: **${pct}%** (moderado).`,
        '',
        `Te sugiero **consultar** con un profesional de salud para ampliar estudios.`,
        `Puedo explicarte qué pruebas suelen solicitarse y cómo prepararte.`,
        `Nota: Esto **no constituye** un diagnóstico definitivo.`,
      ].join('\n');
    }

    return [
      `**Análisis inicial:** Probabilidad estimada de tuberculosis: **${pct}%** (alto).`,
      '',
      `Mi recomendación es **acudir cuanto antes** a un centro de salud para evaluación clínica.`,
      `Puedo guiarte sobre los próximos pasos y medidas de cuidado mientras tanto.`,
      `Importante: Este resultado **no reemplaza** una consulta médica.`,
    ].join('\n');
  }

  /** Limpia el contexto */
  clear() {
    this.ctx = {
      probability: 0,
      imageUrl: "",
      patientId: "",
      timestamp: "",
    };
  }
}
