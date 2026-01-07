import { Component, ElementRef, HostListener, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import {
  IonContent,
  IonHeader,
  IonTitle,
  IonToolbar,
  IonButtons,
  IonIcon,
  IonButton,
  LoadingController,
  ToastController,
  IonFabButton,
  ModalController,
  IonItem,
  IonLabel,
  IonBadge,
  IonPopover,
  IonFooter,
} from '@ionic/angular/standalone';
import { addIcons } from 'ionicons';
import {
  camera,
  imageOutline,
  homeOutline,
  refreshOutline,
  chatbubblesOutline,
  closeOutline,
  send,
  syncOutline,
  closeCircleOutline,
  checkmarkCircleOutline,
  alertCircleOutline,
  helpCircleOutline,
  informationCircleOutline,
} from 'ionicons/icons';
import { ImageService } from '../services/image.service';
import { ChatService } from '../services/chat.service';
import { ChatRecommendationPage } from '../chat-recommendation/chat-recommendation.page';
import { ChatContextService } from '../services/chat-context.service';
import { CircleProgressModule } from '../shared/circle-progress/circle-progress.module';
import { ValidateImageService } from '../services/validate-image.service';
import { lastValueFrom } from 'rxjs';
import { ValidateXrayResponse } from '../shared/resources/files-utils';
import { UploadService } from '../services/payload-upload-dicom.service';
import { HttpClient } from '@angular/common/http';
import { ChatStateService } from '../services/chat-state.service';
@Component({
  selector: 'app-home',
  templateUrl: './home.page.html',
  styleUrls: ['./home.page.scss'],
  standalone: true,
  imports: [
    IonIcon,
    IonContent,
    IonHeader,
    IonToolbar,
    CommonModule,
    FormsModule,
    IonIcon,
    IonButton,
    CircleProgressModule,
    IonFabButton,
    IonBadge,
  ],
})
export class HomePage {
  @ViewChild('fileInput') fileInput!: any;
  @ViewChild('tip') tipRef!: ElementRef<HTMLSpanElement>;
  tipVisible = false;
  tipBelow = false;

  hasSelectedImage = false;
  isUploading = false;
  isValidating = false;
  validationStatus: 'idle' | 'loading' | 'valid' | 'invalid' = 'idle';
  currentStep = ''; // 'validating' | 'analyzing'

  // Diagnosis
  showResults = false;
  xrayImage: string | null = null;
  tbProbability = 0;
  circleKey = true;
  generatedRecommendation: string = '';
  isLoadingRecommendation = false;

  imageDataUrl?: string | null;
  loading = false;
  errorMsg?: string;
  result?: ValidateXrayResponse;

  dicomSelectedName?: string;
  selectedFile?: File;
  payloadBase64?: string;
  isDicomSelected = false;

  constructor(
    private imageService: ImageService,
    private loadingController: LoadingController,
    private toastController: ToastController,
    private chatService: ChatService,
    private modalController: ModalController,
    private chatContext: ChatContextService,
    private validatorService: ValidateImageService,
    private upload: UploadService,
    private http: HttpClient,
    private chatState: ChatStateService
  ) {
    addIcons({
      imageOutline,
      helpCircleOutline,
      refreshOutline,
      informationCircleOutline,
      chatbubblesOutline,
      alertCircleOutline,
      syncOutline,
      checkmarkCircleOutline,
      closeCircleOutline,
      closeOutline,
      send,
      camera,
      homeOutline,
    });
  }

  // Manejo de archivo seleccionado
  async onFileChange(evt: Event) {
    this.errorMsg = undefined;
    this.result = undefined;
    this.imageDataUrl = null;
    this.dicomSelectedName = undefined;

    const input = evt.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;

    const MAX_MB = 50;
    if (file.size > MAX_MB * 1024 * 1024) {
      this.errorMsg = `El archivo excede ${MAX_MB} MB.`;
      return;
    }

    this.loading = true;
    try {
      this.selectedFile = file;
      const payload = await this.upload.buildPayload(file);

      this.payloadBase64 = payload.image;
      this.isDicomSelected = !!payload.isDicom;

      if (this.isDicomSelected) {
        this.dicomSelectedName = file.name;
        // pedir preview al backend
        this.validatorService.getPreview(this.payloadBase64!).subscribe({
          next: (dataUrl) => (this.imageDataUrl = dataUrl),
          error: (err) => {
            this.imageDataUrl = null;
            this.errorMsg =
              err?.error?.error || 'No se pudo generar la vista previa.';
          },
        });
      } else {
        // preview nativa para imágenes
        this.imageDataUrl = payload.previewDataUrl ?? null;
      }

      this.hasSelectedImage = true;
    } catch (e: any) {
      this.errorMsg = e?.message || 'Error al leer el archivo.';
    } finally {
      this.loading = false;
    }
  }

  placeTip(ev: MouseEvent | FocusEvent) {
    const tip = this.tipRef?.nativeElement;
    const trigger = ev.currentTarget as HTMLElement;
    if (!tip || !trigger) return;

    this.tipVisible = true;
    tip.style.position = 'fixed';

    setTimeout(() => {
      const vv = (window as any).visualViewport;
      const vw = vv?.width ?? window.innerWidth;
      const vh = vv?.height ?? window.innerHeight;
      const offsetX = vv?.offsetLeft ?? 0;
      const offsetY = vv?.offsetTop ?? 0;

      const margin = 12;
      const desiredW = Math.min(200, vw - 2 * margin);

      tip.style.width = desiredW + 'px';

      const btn = trigger.getBoundingClientRect();
      const tipRect = tip.getBoundingClientRect();
      const centerX = btn.left + btn.width / 2;

      let left = centerX - desiredW / 2;
      let top = btn.top - tipRect.height - 10;

      const minLeft = offsetX + margin;
      const maxLeft = offsetX + vw - margin - desiredW;
      if (left < minLeft) left = minLeft;
      if (left > maxLeft) left = maxLeft;

      if (top < offsetY + margin) {
        top = btn.bottom + 10;
        this.tipBelow = true;
      } else {
        this.tipBelow = false;
      }

      tip.style.left = `${left}px`;
      tip.style.top = `${top}px`;
    }, 0);
  }

  hideTip() {
    this.tipVisible = false;
  }

  @HostListener('window:scroll') onScroll() {
    this.hideTip();
  }
  @HostListener('window:resize') onResize() {
    this.hideTip();
  }

  getRiskClass(prob: number): string {
    if (prob > 70) return 'high-risk';
    if (prob > 30) return 'medium-risk';
    return 'low-risk';
  }

  getRiskColor(prob: number): string {
    if (prob > 70) return 'tomato';
    if (prob > 30) return 'orange';
    return 'limegreen';
  }

  async openChatRecommendation() {
    if (this.isDicomSelected && !this.imageDataUrl && this.payloadBase64) {
      try {
        const dataUrl = await lastValueFrom(
          this.validatorService.getPreview(this.payloadBase64)
        );
        this.imageDataUrl = dataUrl;
      } catch (e) {
        console.warn('No se pudo obtener la preview DICOM antes del chat', e);
      }
    }

    this.chatContext.setDiagnosisContext({
      probability: Math.max(0, Math.min(1, this.tbProbability / 100)),
      imageUrl: this.imageDataUrl ?? undefined,
    });

    const modal = await this.modalController.create({
      component: ChatRecommendationPage,
    });
    await modal.present();
  }

  async showLoading(message: string) {
    const loading = await this.loadingController.create({
      message,
      spinner: 'circles',
      cssClass: 'custom-loading',
      translucent: true,
      animated: true,
      backdropDismiss: false,
    });
    await loading.present();
    return loading;
  }

  async analyzeImage() {
    if (!this.payloadBase64 || this.isValidating) return;

    this.isValidating = true;
    this.currentStep = 'validating';
    this.validationStatus = 'loading';

    const loading = await this.showLoading('Validando imagen...');
    try {
      const valRes = await lastValueFrom(
        this.upload.validateXray({ image: this.payloadBase64 })
      );
      await loading.dismiss();

      if (!valRes?.is_xray) {
        this.validationStatus = 'invalid';
        this.presentToast('La imagen no parece una radiografía de tórax.');
        this.currentStep = '';
        this.isValidating = false;
        return;
      }

      this.validationStatus = 'valid';
      this.currentStep = 'analyzing';

      const analyzing = await this.showLoading('Analizando radiografía...');

      this.imageService.sendImage(this.payloadBase64).subscribe(
        async (res) => {
          this.tbProbability = Math.max(0.01, res.probability);
          this.circleKey = false;
          setTimeout(() => (this.circleKey = true), 0);
          this.xrayImage = this.isDicomSelected
            ? null
            : this.imageDataUrl ?? null;

          this.showResults = true;
          analyzing.dismiss();
          this.isValidating = false;
          this.currentStep = '';
          this.isLoadingRecommendation = true;
          try {
            this.generatedRecommendation = await this.getDynamicRecommendation(
              this.tbProbability
            );
          } catch {
            this.generatedRecommendation = this.getFallbackRecommendation(
              this.tbProbability
            );
          } finally {
            this.isLoadingRecommendation = false;
          }
        },
        (err) => {
          console.error('Error en el diagnóstico', err);
          analyzing.dismiss();
          this.isValidating = false;
          this.currentStep = '';
          this.presentToast('Error al analizar la radiografía.');
        }
      );
    } catch (err) {
      await loading.dismiss();
      this.presentToast('Error al validar la imagen. Intenta nuevamente.');
      this.isValidating = false;
      this.currentStep = '';
    }
  }

  getFixedProbability(): number {
    return this.tbProbability < 1 ? 1 : +this.tbProbability.toFixed(2);
  }

  getDisplayedProbability(): string {
    return this.tbProbability < 1 ? '< 1' : `${this.tbProbability.toFixed(2)}`;
  }

  resetImage() {
    this.selectedFile = undefined;
    this.payloadBase64 = undefined;
    this.isDicomSelected = false;
    this.dicomSelectedName = undefined;

    this.imageDataUrl = null;
    this.xrayImage = null;

    this.hasSelectedImage = false;
    this.errorMsg = undefined;
  }

  resetPage() {
    this.resetImage();
    this.tbProbability = 0;
    this.showResults = false;
    this.generatedRecommendation = '';
    this.validationStatus = 'idle';
    this.currentStep = '';

    this.chatContext.setDiagnosisContext({ probability: 0 }); // opcional: resetear contexto
    this.chatState.clear();
  }

  getProbabilityDescription(): string {
    if (this.tbProbability > 70) {
      return 'Alta probabilidad de tuberculosis.';
    } else if (this.tbProbability > 30) {
      return 'Probabilidad moderada de tuberculosis.';
    } else {
      return 'Baja probabilidad de tuberculosis.';
    }
  }

  cleanRecommendation(text: string): string {
    return text
      .replace(/ImageLayout.*$/gi, '')
      .replace(/Recomendaci[oó]n:?.*/gi, '')
      .trim();
  }

  async getDynamicRecommendation(prob: number): Promise<string> {
    const question = `
Eres un asistente médico. Según una imagen de rayos X torácica, se obtuvo una probabilidad de tuberculosis de ${prob.toFixed(
      2
    )}%.

Devuelve únicamente una **frase corta en español** que brinde una recomendación médica profesional relacionada con ese valor.

La respuesta debe:
- Ser **concisa** (una sola línea).
- No incluir títulos, etiquetas, encabezados, aclaraciones ni explicaciones.
- No mencionar el hecho de que estás respondiendo.
- No incluir frases como "Respuesta:", "Recomendación:", ni instrucciones técnicas.
- No agregar texto adicional, marcadores, códigos ni layout.

Si la probabilidad es muy baja (<5%), simplemente indica que **no se requiere tratamiento** y se puede hacer seguimiento de rutina.
`;

    try {
      const markdown = await this.chatService.askRecommendation(question, {
        probability: Math.max(0, Math.min(1, prob / 100)),
      });

      return this.cleanRecommendation(markdown);
    } catch (err) {
      console.error('Error generando recomendación dinámica', err);
      return this.getFallbackRecommendation(prob);
    }
  }

  getFallbackRecommendation(prob: number): string {
    if (prob > 70) {
      return 'Consulta médica urgente. La probabilidad de TB es alta.';
    } else if (prob > 30) {
      return 'Se recomienda valoración médica adicional.';
    } else {
      return 'Mantén seguimiento regular. Riesgo bajo de TB.';
    }
  }

  async presentToast(message: string) {
    const toast = await this.toastController.create({
      cssClass: 'toast-white',
      message: message,
      duration: 3000,
      position: 'bottom',
    });
    await toast.present();
  }
}
