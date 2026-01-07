import { Injectable } from '@angular/core';
import { ChatMsg } from 'src/models/chat.model';



@Injectable({ providedIn: 'root' })
export class ChatStateService {
  private messages: ChatMsg[] = [];

  getAll(): ChatMsg[] {
    return this.messages;
  }

  setAll(msgs: ChatMsg[]) {
    // clonar para evitar referencias raras
    this.messages = [...msgs];
  }

  clear() {
    this.messages = [];
  }
}
