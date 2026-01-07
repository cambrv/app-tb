/**
 * ChatRecommendationPage
 *
 * Componente que permite la interacción del usuario con un chatbot basado en IA,
 * el cual ofrece recomendaciones clínicas según la probabilidad de tuberculosis estimada.
 *
 * Funcionalidades principales:
 * - Recupera el contexto (probabilidad + imagen) desde `ChatContextService`.
 * - Muestra un mensaje inicial basado en la probabilidad de tuberculosis.
 * - Permite enviar preguntas al backend usando `ChatService` y mostrar las respuestas del modelo.
 * - Aplica un prompt dinámico ajustado al nivel de riesgo para mejorar la calidad de respuesta.
 * - Las respuestas del chatbot están estructuradas con Markdown y formateadas en el frontend.
 * - Opción de cerrar el modal que contiene la conversación.
 */

/**
 * ChatRecommendationPage
 *
 * Component for user interaction with an AI-powered chatbot that provides
 * clinical recommendations based on the estimated probability of tuberculosis.
 *
 * Main functionalities:
 * - Retrieves diagnosis context (probability + image) from `ChatContextService`.
 * - Displays an initial system message based on the TB risk level.
 * - Sends user input to the backend using `ChatService` and displays the model's response.
 * - Uses a dynamic prompt adjusted to the risk level to optimize AI output.
 * - Responses are expected in Markdown and rendered using a custom pipe.
 * - Modal can be closed via a dedicated action.
 */

import { Component, inject, OnInit, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import {
  IonContent,
  IonTitle,
  IonToolbar,
  IonButtons,
  IonItem,
  IonInput,
  IonButton,
  IonIcon,
  ModalController,
} from '@ionic/angular/standalone';
import { ChatService } from '../services/chat.service';
import { ChatContextService } from '../services/chat-context.service';
import { MarkdownFormatPipe } from 'src/app/pipes/markdown-format.pipe';
import { lastValueFrom } from 'rxjs';
import { TypewriterService } from '../services/typewriter.service';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { ChatMessage, ChatMsg } from 'src/models/chat.model';
import { marked } from 'marked';
import { ChatStateService } from '../services/chat-state.service';

@Component({
  selector: 'app-chat-recommendation',
  templateUrl: './chat-recommendation.page.html',
  styleUrls: ['./chat-recommendation.page.scss'],
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    IonContent,
    IonToolbar,
    IonTitle,
    IonButtons,
    IonItem,
    IonInput,
    IonButton,
    IonIcon,
    MarkdownFormatPipe,
  ],
})
export class ChatRecommendationPage implements OnInit {
  tbProbability: number = 0;
  imagePreview: string | null = null;
  isDicom = false;
  fileName?: string;

chatMessages: ChatMsg[] = [];
  currentMessage = '';

  //Typewritting
  @ViewChild(IonContent) content!: IonContent;
  private cancelTyping?: () => void;

  private typewriter = inject(TypewriterService);
  private sanitizer = inject(DomSanitizer);
  private chatSvc = inject(ChatService);
  private ctx = inject(ChatContextService);

  messages: ChatMessage[] = [];

  constructor(
    private chatService: ChatService,
    private chatContext: ChatContextService,
    private modalController: ModalController,
    private chatState: ChatStateService
  ) {}

  ngOnInit() {
    // Si ya tienes estos datos en ChatContextService, podrías recuperarlos aquí
    // y mostrarlos arriba (preview + probabilidad)
    const ctx = this.chatContext.getDiagnosisContext?.();
    if (ctx?.probability != null) {
      this.tbProbability = Math.round((ctx.probability || 0) * 100);
    }
    this.imagePreview = (ctx as any)?.imageUrl ?? this.imagePreview;

    // Recupera mensajes guardados si existen
  const cached = this.chatState.getAll();

  if (cached.length > 0) {
    this.chatMessages = [...cached];
  } else {
    // Si no hay historial, creamos el intro
    const intro: ChatMsg = {
      isUser: false,
      text: this.chatContext.buildIntroMessage(),
      renderedText: '',
      isTyping: true,
    };
    this.chatMessages.push(intro);

    this.cancelTyping = this.typewriter.animateMsg(intro, {
      typingSpeedMs: 14,
      smartPauses: true,
      onTick: () => this.scrollToBottom(),
    });
  }
  this.chatState.setAll(this.chatMessages);
  }

  async sendMessage() {
    const q = (this.currentMessage || '').trim();
    if (!q) return;

    // 1) Mensaje del usuario (sin animación)
    this.chatMessages.push({
      isUser: true,
      text: q,
      renderedText: q,
    });
    this.chatState.setAll(this.chatMessages);
    this.currentMessage = '';
    this.scrollToBottom();

    // 2) Placeholder del bot con tipeo
    const bot: ChatMsg = {
      isUser: false,
      text: '',
      renderedText: '',
      isTyping: true,
    };
    this.chatMessages.push(bot);
    this.chatState.setAll(this.chatMessages);
    this.scrollToBottom();

    try {
      // 3) Llamada al backend
      const ctx = this.chatContext.getDiagnosisContext();
      const markdown = await this.chatService.askRecommendation(q, ctx);

      // Si hay una animación previa corriendo, cancelarla
      this.cancelTyping?.();

      // 4) Asignar texto completo y animar renderedText
      bot.text = markdown || '(sin respuesta)';
      this.cancelTyping = this.typewriter.animateMsg(bot, {
        typingSpeedMs: 12,
        smartPauses: true,
        onTick: () => this.scrollToBottom(),
      });
      this.chatState.setAll(this.chatMessages);

    } catch (e) {
      // Error: mostramos un mensaje estático
      this.cancelTyping?.();
      bot.isTyping = false;
      bot.text = '**Error:** no pude obtener respuesta. Intenta de nuevo.';
      bot.renderedText = bot.text;
      this.chatState.setAll(this.chatMessages);

    }
  }

  async scrollToBottom() {
    try {
      await this.content?.scrollToBottom(200);
    } catch {}
  }

  closeModal() {
    this.cancelTyping?.();
    this.modalController.dismiss();
  }
}
